"""
    Script for running Flux Kontext on the paragraph_level.json dataset.
    Adapted for InfoDet.
"""
import os
import sys
import json
import math
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

# Add repo root to sys.path to allow importing the 'kontext' package.
sys.path.append(str(Path(__file__).resolve().parent))

# If needed for debugging
if os.environ.get("DEBUGPY", "0") == "1":
    import debugpy
    port = 5679 + int(os.environ.get("LOCAL_RANK", 0))
    debugpy.listen(("0.0.0.0", port))
    print(f"[rank={os.environ.get('RANK')}] debugpy listening on port {port}", flush=True)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        debugpy.wait_for_client()

import torch
from tqdm import tqdm
from loguru import logger
from torchvision.transforms import functional as F
import random
import datasets

from kontext.pipeline_flux_kontext import FluxKontextPipeline
from kontext.pipeline_utils import create_larger_latent_bboxes
from kontext.transformer_flux import FluxTransformer2DModel
from kontext.attention import get_attention_processor

# Set seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _parse_positive_int(value, default):
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return default


_SAVE_RESULTS_MAX_WORKERS = _parse_positive_int(os.environ.get("SAVE_RESULTS_MAX_WORKERS"), 2)
_SAVE_RESULTS_MAX_PENDING_TASKS = _parse_positive_int(os.environ.get("SAVE_RESULTS_MAX_PENDING_TASKS"), max(4, _SAVE_RESULTS_MAX_WORKERS * 2))
_SAVE_RESULTS_EXECUTOR = ThreadPoolExecutor(max_workers=_SAVE_RESULTS_MAX_WORKERS)
_SAVE_RESULTS_SEMAPHORE = threading.BoundedSemaphore(_SAVE_RESULTS_MAX_PENDING_TASKS)


def _shutdown_save_results_executor():
    _SAVE_RESULTS_EXECUTOR.shutdown(wait=True)


atexit.register(_shutdown_save_results_executor)

def load_dataset_from_json(json_path: Path, image_dir: Path, target_lang: str):
    """
    Generator function to yield examples for Hugging Face Dataset.
    """
    json_path = Path(json_path)
    image_dir = Path(image_dir)

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    translation_key = f"text_{target_lang.lower()}"
    if translation_key not in data[0]["paragraphs"][0] and "translation" in data[0]["paragraphs"][0]: # Fallback check
         translation_key = "translation"

    for img_idx, item in enumerate(data):
        file_name = item.get('file_name')
        image_path = image_dir / file_name
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}, skipping...")
            continue
            
        # Validation check (lightweight)
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}, skipping...")
            continue

        bbox_list = []
        description_list = []
        translation_list = []
        label_list = [] # 2 for text
        annotation_id_list = []

        for p_idx, para in enumerate(item.get('paragraphs', [])):
            text = para.get('text', '')
            trans = para.get(translation_key, '')
            bbox = para.get('bbox')
            
            # Simple check for valid paragraph
            if text and bbox and len(bbox) == 4:
                bbox_list.append(bbox)
                description_list.append(text)
                
                # If no translation found, use original text or handle as needed. 
                # For inference we usually need the translation.
                # If missing, script logic later handles it or skips. 
                translation_list.append(trans if trans else None) 
                
                label_list.append(2) # 2 = text label in original script logic
                annotation_id_list.append(f"{img_idx}_{p_idx}")

        yield {
            "image_id": item.get("image_id", img_idx),
            "file_name": file_name,
            "image_path": str(image_path), # Pass path instead of object
            "bbox": bbox_list,
            "description": description_list,
            "language_translations": translation_list,
            "label": label_list,
            "annotation_id": annotation_id_list,
            "target_language": target_lang
        }

@torch.no_grad()
def _load_lora_weights_safe(pipe, lora_path):
    """
    Load LoRA weights, stripping incompatible config keys if needed.
    
    Handles version mismatch between training (newer PEFT) and inference (older PEFT).
    """
    import safetensors.torch
    from safetensors import safe_open
    import json
    import tempfile
    
    # List of supported LoraConfig parameters in installed PEFT version
    # Determined by inspecting peft.LoraConfig.__init__ signature
    SUPPORTED_LORA_KEYS = {
        'task_type', 'peft_type', 'auto_mapping', 'base_model_name_or_path',
        'revision', 'inference_mode', 'r', 'target_modules', 'exclude_modules',
        'lora_alpha', 'lora_dropout', 'fan_in_fan_out', 'bias', 'use_rslora',
        'modules_to_save', 'init_lora_weights', 'layers_to_transform',
        'layers_pattern', 'rank_pattern', 'alpha_pattern', 'megatron_config',
        'megatron_core', 'trainable_token_indices', 'loftq_config', 'eva_config',
        'corda_config', 'use_dora', 'layer_replication', 'runtime_config', 'lora_bias'
    }
    
    # Try direct load first
    try:
        pipe.load_lora_weights(lora_path)
        return
    except TypeError as e:
        # Check for the wrapper error message diffusers raises
        if "LoraConfig" not in str(e) and "could not be instantiated" not in str(e):
            raise
        logger.warning(f"LoRA load failed (config incompatibility), attempting workaround: {e}")
    
    # Fallback: Load safetensors, sanitize metadata, and load manually
    lora_files = list(Path(lora_path).glob("*.safetensors"))
    if not lora_files:
         lora_file = Path(lora_path) / "pytorch_lora_weights.safetensors"
    else:
         lora_file = lora_files[0]

    if not lora_file.exists():
        raise FileNotFoundError(f"Could not find LoRA weights file in: {lora_path}")
    
    # Load the safetensors file to get metadata and state dict
    with safe_open(str(lora_file), framework="pt") as f:
        metadata = dict(f.metadata() or {})
    
    # Sanitize lora_adapter_metadata if present
    if "lora_adapter_metadata" in metadata:
        lora_meta = json.loads(metadata["lora_adapter_metadata"])
        
        # Filter to only supported keys for each adapter
        removed_keys = []
        for key in list(lora_meta.keys()):
            # Extract the base key (remove "transformer." prefix)
            base_key = key.split('.')[-1]
            if base_key not in SUPPORTED_LORA_KEYS:
                lora_meta.pop(key)
                removed_keys.append(key)
        
        if removed_keys:
            logger.info(f"Removed unsupported LoRA config keys: {removed_keys}")
        
        # Update metadata with sanitized config
        metadata["lora_adapter_metadata"] = json.dumps(lora_meta)
    
    # Load weights
    state_dict = safetensors.torch.load_file(str(lora_file))
    
    # Write sanitized version to temp directory and load
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_lora_file = Path(tmpdir) / "pytorch_lora_weights.safetensors"
        safetensors.torch.save_file(state_dict, str(tmp_lora_file), metadata=metadata)
        pipe.load_lora_weights(tmpdir)
        logger.info(f"Successfully loaded sanitized LoRA")

@torch.no_grad()
def infer_kontext(dataset, result_save_path, lora_path, rl_lora_path, pretrained_model_name_or_path, args, disable_inner_pbar: bool = False, use_bridge: bool = False):
    device = 'cuda'
    prompt_input_type = 'original_translated'
    prompt_settings = args.prompt_settings
    attention_setting = args.attention_setting
    bring_area_to_1024_squared = args.bring_area_to_1024_squared
    crop_height_fraction = None
    result_save_path.mkdir(parents=True, exist_ok=True)

    # Load models
    pipe = FluxKontextPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=disable_inner_pbar)
    
    # Load LoRA logic (kept from original)
    if lora_path is not None and lora_path.strip() != '' and lora_path != 'None':
        logger.info(f"Loading LoRA weights from: {lora_path}")
        try:
            _load_lora_weights_safe(pipe, lora_path)
            if rl_lora_path is not None and rl_lora_path.strip() != '' and rl_lora_path != 'None':
                logger.info(f"Loading RL LoRA weights from: {rl_lora_path}")
                pipe.fuse_lora()
                from peft import PeftModel
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, rl_lora_path, is_trainable=False)
                pipe.fuse_lora()
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            raise

    attention_processor = get_attention_processor(attention_setting)
    pipe.transformer.set_attn_processor(attention_processor)

    pbar = tqdm(dataset, desc="Processing Images")
    for sample in pbar:
        # CRITICAL: Clear attention masks from previous sample to avoid using wrong masks
        if hasattr(attention_processor, 'clear_cached_masks'):
            attention_processor.clear_cached_masks()
        
        torch.cuda.empty_cache()
        try:
            target_lang = sample.get('target_language', 'unknown')
            file_stem = Path(sample['file_name']).stem
            image_prefix = f"{sample['image_id']}_{file_stem}_english_{target_lang}"
            
            if (result_save_path / f"{image_prefix}_result.png").exists():
                logger.debug(f"Skipping {file_stem}, already processed.")
                continue

            # Original script logic adaptation
            # Data from HF dataset comes as lists
            
            # Filter for text (label == 2)
            label_arr = np.array(sample['label'])
            text_mask = label_arr == 2
            
            if not np.any(text_mask):
                continue
                
            text_bboxes = np.array(sample['bbox'])[text_mask]
            
            # Extract descriptions and translations based on mask
            descriptions = np.array(sample['description'])
            translations = np.array(sample['language_translations'])
            ids = np.array(sample['annotation_id'])
            
            ref_texts = descriptions[text_mask]
            gen_texts = translations[text_mask]
            annotation_ids = ids[text_mask]

            # Filter valid translations (not None/Empty)
            valid_mask = np.array([t is not None and len(t) > 0 for t in gen_texts])
            if not np.all(valid_mask):
                text_bboxes = text_bboxes[valid_mask]
                ref_texts = ref_texts[valid_mask]
                gen_texts = gen_texts[valid_mask]
                annotation_ids = annotation_ids[valid_mask]

            if len(gen_texts) == 0:
                continue

            # BBox enlargement logic requires lengths
            characters_source = [len(t) for t in ref_texts]
            characters_target = [len(t) for t in gen_texts]
            
            # Prepare image
            try:
                source_image = Image.open(sample['image_path']).convert('RGB')
            except Exception as e:
                 logger.error(f"Failed to open image {sample['image_path']}: {e}")
                 continue

            original_w, original_h = source_image.size
            
            if bring_area_to_1024_squared:
                aspect_ratio = original_w / original_h   
                width = round((1024 * 1024 * aspect_ratio) ** 0.5)
                height = round((1024 * 1024 / aspect_ratio) ** 0.5)
            else:
                width = original_w
                height = original_h

            multiple_of = pipe.vae_scale_factor * 2
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of
            source_image_resized = source_image.resize((width, height))
            width_scale = original_w / width
            height_scale = original_h / height   

            prompt = ''
            prompts = []
            instance_bboxes_xyxy_normalized = []
            final_annotation_ids = []

            for i, (text_bbox, ref_text, gen_text, ann_id) in enumerate(zip(text_bboxes, ref_texts, gen_texts, annotation_ids)):
                x, y, w, h = text_bbox
                x1 = math.floor(x / width_scale)
                y1 = math.floor(y / height_scale)
                x2 = math.ceil((x + w) / width_scale)
                y2 = math.ceil((y + h) / height_scale)

                if prompt_input_type == 'original_translated':
                    p_text = f'Change "{ref_text}" to "{gen_text}". '
                    prompt += p_text
                    prompts.append(p_text)
                
                instance_bboxes_xyxy_normalized.append([x1 / width, y1 / height, x2 / width, y2 / height])
                final_annotation_ids.append(ann_id)

            prompt = prompt.strip()
            global_prompt = ' '
            prompts.insert(0, global_prompt)
            
            # Run inference
            generator = torch.Generator()
            generator.manual_seed(0)

            # Argument preparation
            hard_binding = args.hard_image_attribute_binding_list
            if args.use_typo_boxes:
                 instance_bboxes_xyxy_normalized_larger = create_larger_latent_bboxes(instance_bboxes_xyxy_normalized, characters_source, characters_target)
            else:
                 instance_bboxes_xyxy_normalized_larger = instance_bboxes_xyxy_normalized

            prompt_2 = '$BREAKFLAG$'.join(prompts)

            images_list = pipe(
                image=source_image_resized,
                prompt=global_prompt,
                prompt_2=prompt_2,
                guidance_scale=4.0,
                height=height,
                width=width,
                _auto_resize=False,
                num_inference_steps=28,
                num_hard_control_steps=28,
                instance_bboxes_xyxy_normalized=instance_bboxes_xyxy_normalized_larger,
                hard_image_attribute_binding_list=hard_binding,
                use_bridge=use_bridge,
                generator=generator,
                use_global_prompt=True,
                use_pos_embeds_from_orig=True,
                use_context_bridge_tokens=False,
                pad_to_square=False,
                prompt_settings=prompt_settings,
                attention_setting=attention_setting,
                larger_latent_boxes=False,
            ).images

            main_result = images_list[0][0]
            
            # Save results
            save_name = result_save_path / f"{image_prefix}_result.png"
            main_result.save(save_name)
            
            # Also save source for comparison if needed (not strictly in original full script but good practice)
            # source_image.save(result_save_path / f"{image_prefix}_source.png")

        except Exception as e:
            logger.exception(f"Failed to process image {sample.get('file_name', 'unknown')}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Run Flux Kontext on paragraph_level.json")
    # New args
    parser.add_argument("--json-path", type=Path, default=None, help="Path to specific paragraph_level.json")
    parser.add_argument("--json-dir", type=Path, default=None, help="Base directory searching for language files (if --languages used)")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--target-language", type=str, default=None, help="Specific target language (if using --json-path)")
    parser.add_argument("--languages", nargs='+', default=['French', 'German', 'Italian', 'Spanish'], help="List of languages to process")
    
    # Inherited args
    parser.add_argument("--base-model-path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="Path or HF id of the base FLUX.1-Kontext-dev model.")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA path")
    parser.add_argument("--rl-lora-path", type=str, default=None, help="RL LoRA path")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name for output dir")
    parser.add_argument("--save-dir", type=Path, default=Path("./results"), help="Base save directory")

    parser.add_argument("--bring-area-to-1024-squared", action="store_true", default=False)
    parser.add_argument("--prompt-settings", type=str, default='outer_local_prompts')
    parser.add_argument("--attention-setting", type=str, default='APITA', choices=['full', 'APITA'])
    parser.add_argument("--use-bridges", action="store_true", default=False)
    parser.add_argument("--use-typo-boxes", action="store_true", default=False)
    parser.add_argument("--hard-image-attribute-binding-list", type=str, default="0,57")

    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)

    args = parser.parse_args()
    
    # Parse binding list
    def str2list(string):
        return [int(item) for item in string.split(',')]
    args.hard_image_attribute_binding_list = str2list(args.hard_image_attribute_binding_list)
    if len(args.hard_image_attribute_binding_list) > 0:
        args.hard_image_attribute_binding_list = list(range(args.hard_image_attribute_binding_list[0], args.hard_image_attribute_binding_list[1]))
    else:
        args.hard_image_attribute_binding_list = list(range(0, 57))

    # Language mapping for file discovery
    LANG_CODES = {
        "French": "fr", "German": "de", "Italian": "it", "Spanish": "es",
        "English": "en", "Chinese": "zh", "Russian": "ru", "Japanese": "ka", 
        "Korean": "ko", "Portuguese": "pt"
    }

    from datasets import concatenate_datasets, Dataset

    # Handle multiple languages
    all_datasets = []
    
    # If explicit json path provided, treat as single language (legacy support or specific file)
    if args.json_path and args.json_path.exists():
        logger.info(f"Loading single file from {args.json_path} for {args.target_language}")
        ds = Dataset.from_generator(
            load_dataset_from_json, 
            gen_kwargs={
                "json_path": str(args.json_path),
                "image_dir": str(args.image_dir),
                "target_lang": args.target_language
            }
        )
        all_datasets.append(ds)
    
    # If languages list provided, try to find files
    elif args.languages:
        if not args.json_dir:
            logger.error("Must provide --json-dir when using --languages list")
            return

        for lang in args.languages:
            code = LANG_CODES.get(lang, lang.lower()[:2])
            
            # Search patterns for the file
            # 1. In language subdirectory
            # 2. In root of json_dir
            candidates = [
                args.json_dir / "translations" / lang.lower() / f"paragraph_level_{code}_merged.json",
                args.json_dir / f"paragraph_level_{code}_merged.json",
                args.json_dir / f"paragraph_level_{code}.json",
            ]
            
            found_path = None
            for p in candidates:
                if p.exists():
                    found_path = p
                    break
            
            if found_path:
                logger.info(f"Loading {lang} from {found_path}")
                # We need a closure for generator to capture variables correctly
                ds = Dataset.from_generator(
                    load_dataset_from_json,
                    gen_kwargs={
                        "json_path": str(found_path),
                        "image_dir": str(args.image_dir),
                        "target_lang": lang
                    }
                )
                # Add language column if not present (our generator adds it to item dict, but good to be explicit meta)
                all_datasets.append(ds)
            else:
                logger.warning(f"Could not find merged dataset for {lang}. Checked: {[str(c) for c in candidates]}")

    if not all_datasets:
        logger.error("No datasets loaded!")
        return

    if len(all_datasets) > 1:
        hf_dataset = concatenate_datasets(all_datasets)
        logger.info(f"Concatenated {len(all_datasets)} datasets. Total samples: {len(hf_dataset)}")
    else:
        hf_dataset = all_datasets[0]
        logger.info(f"Loaded single dataset. Total samples: {len(hf_dataset)}")
    
    # Sharding
    if args.num_shards is not None and args.shard_index is not None:
        hf_dataset = hf_dataset.shard(num_shards=args.num_shards, index=args.shard_index)
        logger.info(f"Sharded to {len(hf_dataset)} samples (shard {args.shard_index}/{args.num_shards})")

    # Limit samples
    if args.num_samples is not None:
        hf_dataset = hf_dataset.select(range(min(args.num_samples, len(hf_dataset))))
    
    # Output dir
    lang_suffix = args.target_language if args.target_language else "multilingual"
    run_name = f'paragraph_level_{lang_suffix}'
    final_save_dir = args.save_dir / run_name / f'flux_kontext_{args.exp_name}'
    final_save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {final_save_dir}")
    
    infer_kontext(
        dataset=hf_dataset,
        result_save_path=final_save_dir,
        lora_path=args.lora_path,
        rl_lora_path=args.rl_lora_path,
        pretrained_model_name_or_path=args.base_model_path,
        args=args,
        use_bridge=args.use_bridges
    )
    
    logger.info("Inference complete!")

if __name__ == "__main__":
    main()
