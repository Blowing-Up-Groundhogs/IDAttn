"""
    Script for running Flux Kontext on the LoMOE-Bench dataset (v2).
    Uses specific prompt format: "Replace {source_object} with {target_object}"
    and empty global prompt.
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
import shlex

# Ensure repo root and script directory are on sys.path for module resolution under Slurm.
REPO_ROOT = Path(__file__).resolve().parent
REPO_PARENT = REPO_ROOT.parent
for p in (REPO_ROOT, REPO_PARENT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from kontext.pipeline_utils import create_larger_latent_bboxes
if os.environ.get("DEBUGPY", "0") == "1":
    import debugpy
    port = 5679 + int(os.environ.get("LOCAL_RANK", 0))
    debugpy.listen(("0.0.0.0", port))
    print(f"[rank={os.environ.get('RANK')}] debugpy listening on port {port}", flush=True)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        debugpy.wait_for_client()

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from loguru import logger
from torchvision.transforms import functional as F
import random

from kontext.pipeline_flux_kontext import FluxKontextPipeline
from kontext.transformer_flux import FluxTransformer2DModel
from kontext.attention import get_attention_processor

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LoMOEDatasetV2(Dataset):
    def __init__(self, json_path, mask_orig_prompts_path, base_dir, split='test'):
        self.base_dir = Path(base_dir)
        with open(json_path) as f:
            self.data = json.load(f)
        
        # Load source mappings
        # The line index in mask_orig_prompts.txt corresponds to the image index (0, 1, ...).
        # We will retrieve the correct line based on the folder index found in 'image_path'.
        with open(mask_orig_prompts_path, 'r') as f:
            lines = f.readlines()

        # To be robust, let's load all lines into a list
        self.raw_source_prompts = [l.strip() for l in lines]
        self.keys = sorted(list(self.data.keys()))
        
        # We need a robust way to map key "00" (image 0) -> line 0 of text file?
        # Key "10" (image 10) -> line 10 of text file?
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        
        # Extract image index from path images/X/init_image.png -> X
        try:
            image_idx_str = item['image_path'].split('/')[1] # images/0/init_image.png -> 0
            image_idx = int(image_idx_str)
        except Exception:
            # Fallback if path structure is different
            # Try to parse key? "00" -> 0. "10" -> 10.
            if key.isdigit():
                 image_idx = int(key)
            else:
                 # Last resort, use idx but this is risky if keys are not 0..N
                 image_idx = idx
        
        # Get source objects from text file line
        if image_idx < len(self.raw_source_prompts):
            line_content = self.raw_source_prompts[image_idx]
            # Parse shlex because it's quoted: "obj1" "obj2"
            try:
                source_objects = shlex.split(line_content)
            except:
                source_objects = [line_content]
        else:
            logger.warning(f"Index {image_idx} out of bounds for source prompts (len {len(self.raw_source_prompts)})")
            source_objects = []

        
        # Load image
        img_path = self.base_dir / item['image_path']
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise e
            
        w, h = image.size
        
        # Parse prompts and masks
        fg_prompts = shlex.split(item["fg_prompt"])
        mask_paths = shlex.split(item["mask_path"])
        
        # Ensure we have enough source objects
        if len(source_objects) < len(fg_prompts):
             # Pad with last object or generic?
             # Or maybe the source_objects list exactly matches mask_paths len?
             # Let's fill with "object" if missing
             source_objects += ["object"] * (len(fg_prompts) - len(source_objects))
        
        text_sources = source_objects[:len(fg_prompts)]
        text_targets = fg_prompts
        
        bboxes = []
        for mp in mask_paths:
            full_mp = self.base_dir / mp
            # Load mask safely
            try:
                mask_pil = Image.open(full_mp)
                mask = np.array(mask_pil)
            except Exception as e:
                logger.error(f"Error loading mask {full_mp}: {e}")
                bboxes.append([0.0, 0.0, 0.01, 0.01]) # Fallback dummy
                continue

            if mask.ndim == 3:
                mask = mask[:, :, 0]
            
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                logger.warning(f"Empty mask for {mp} in {key}; using dummy bbox")
                bboxes.append([0.0, 0.0, 0.01, 0.01])
                continue
                
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            bboxes.append([
                float(x_min) / w,
                float(y_min) / h,
                float(x_max + 1) / w,
                float(y_max + 1) / h
            ])

        return {
            'id': key,
            'image_source': image,
            'bboxes_xyxy_normalized': bboxes,
            'text_source': text_sources,
            'text_target': text_targets,
            'global_prompt': "", # Empty global prompt as requested
            'src_language': 'en',
            'tgt_language': 'en'
        }


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


def _load_lora_weights_safe(pipe, lora_path):
    import safetensors.torch
    from safetensors import safe_open
    
    SUPPORTED_LORA_KEYS = {
        'task_type', 'peft_type', 'auto_mapping', 'base_model_name_or_path',
        'revision', 'inference_mode', 'r', 'target_modules', 'exclude_modules',
        'lora_alpha', 'lora_dropout', 'fan_in_fan_out', 'bias', 'use_rslora',
        'modules_to_save', 'init_lora_weights', 'layers_to_transform',
        'layers_pattern', 'rank_pattern', 'alpha_pattern', 'megatron_config',
        'megatron_core', 'trainable_token_indices', 'loftq_config', 'eva_config',
        'corda_config', 'use_dora', 'layer_replication', 'runtime_config', 'lora_bias'
    }
    
    try:
        pipe.load_lora_weights(lora_path)
        return
    except TypeError as e:
        if "LoraConfig" not in str(e) and "could not be instantiated" not in str(e):
            raise
        logger.warning(f"LoRA load failed (config incompatibility), attempting workaround: {e}")
    
    lora_file = Path(lora_path) / "pytorch_lora_weights.safetensors"
    if not lora_file.exists():
        raise FileNotFoundError(f"Could not find LoRA weights file: {lora_file}")
    
    with safe_open(str(lora_file), framework="pt") as f:
        metadata = dict(f.metadata() or {})
    
    if "lora_adapter_metadata" in metadata:
        lora_meta = json.loads(metadata["lora_adapter_metadata"])
        removed_keys = []
        for key in list(lora_meta.keys()):
            base_key = key.split('.')[-1]
            if base_key not in SUPPORTED_LORA_KEYS:
                lora_meta.pop(key)
                removed_keys.append(key)
        
        if removed_keys:
            logger.info(f"Removed unsupported LoRA config keys: {removed_keys}")
        metadata["lora_adapter_metadata"] = json.dumps(lora_meta)
    
    state_dict = safetensors.torch.load_file(str(lora_file))
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_lora_file = Path(tmpdir) / "pytorch_lora_weights.safetensors"
        safetensors.torch.save_file(state_dict, str(tmp_lora_file), metadata=metadata)
        pipe.load_lora_weights(tmpdir)
        logger.info(f"Successfully loaded sanitized LoRA")

@torch.no_grad()
def infer_kontext(dataset, result_save_path, lora_path, rl_lora_path, pretrained_model_name_or_path, args, disable_inner_pbar: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs are available")
    
    device = 'cuda'
    prompt_settings = args.prompt_settings
    attention_setting = args.attention_setting
    bring_area_to_1024_squared = args.bring_area_to_1024_squared
    result_save_path.mkdir(parents=True, exist_ok=True)

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
    
    if lora_path is not None and lora_path.strip() != '' and lora_path != 'None':
        logger.info(f"Loading SFT LoRA weights from: {lora_path}")
        _load_lora_weights_safe(pipe, lora_path)
            
        if rl_lora_path is not None and rl_lora_path.strip() != '' and rl_lora_path != 'None':
            pipe.fuse_lora()
            logger.info(f"Loading RL LoRA weights from: {rl_lora_path}")
            try:
                _load_lora_weights_safe(pipe, rl_lora_path)
            except Exception as e1:
                try:
                    from peft import PeftModel
                    pipe.transformer = PeftModel.from_pretrained(
                        pipe.transformer,
                        rl_lora_path,
                        is_trainable=False
                    )
                except Exception as e2:
                    raise e2
            pipe.fuse_lora()
    else:
        logger.info("No LoRA path provided - using base model only")

    attention_processor = get_attention_processor(attention_setting)
    pipe.transformer.set_attn_processor(attention_processor)

    pbar = tqdm(dataset, desc="Processing Images")
    for idx, sample in enumerate(pbar):
        if hasattr(attention_processor, 'clear_cached_masks'):
            attention_processor.clear_cached_masks()
        
        torch.cuda.empty_cache()
        try:
            sample_id = sample['id']
            image_prefix = f"{sample_id}_{sample['src_language']}_{sample['tgt_language']}_{idx:06d}"
            
            if (result_save_path / f"{image_prefix}_result.png").exists():
                logger.debug(f"Skipping sample {idx} (ID: {sample_id}) as it has already been processed.")
                continue

            instance_bboxes_xyxy_normalized = sample['bboxes_xyxy_normalized']
            if not instance_bboxes_xyxy_normalized:
                continue

            text_sources = sample['text_source']
            text_targets = sample['text_target']
            
            global_prompt = sample['global_prompt']
            if not global_prompt:
                 global_prompt = " "

            source_image = sample['image_source']
            source_image_w, source_image_h = source_image.size
            
            if bring_area_to_1024_squared:
                aspect_ratio = source_image_w / source_image_h   
                width = round((1024 * 1024 * aspect_ratio) ** 0.5)
                height = round((1024 * 1024 / aspect_ratio) ** 0.5)
            else:
                width = source_image_w
                height = source_image_h

            multiple_of = pipe.vae_scale_factor * 2
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of
            
            if width != source_image_w or height != source_image_h:
                source_image = source_image.resize((width, height))

            # NEW LOGIC: Correct Prompt Construction
            prompts = []
            for text_source, text_target in zip(text_sources, text_targets):
                prompts.append(f"Replace {text_source} with {text_target}")
            
            prompts.insert(0, global_prompt)
            prompt_2 = '$BREAKFLAG$'.join(prompts)

            num_inference_steps = 28
            num_hard_control_steps = 28
            hard_image_attribute_binding_list = args.hard_image_attribute_binding_list
            use_bridge = False
            use_global_prompt = True
            use_pos_embeds_from_orig = True
            use_context_bridge_tokens = False
            
            generator = torch.Generator()
            generator.manual_seed(0)

            images_list = pipe(
                image=source_image,
                prompt=global_prompt,
                prompt_2=prompt_2,
                guidance_scale=4.0,
                height=height,
                width=width,
                _auto_resize=False,
                num_inference_steps=num_inference_steps,
                num_hard_control_steps=num_hard_control_steps,
                instance_bboxes_xyxy_normalized=instance_bboxes_xyxy_normalized,
                hard_image_attribute_binding_list=hard_image_attribute_binding_list,
                use_bridge=use_bridge,
                generator=generator,
                use_global_prompt=use_global_prompt,
                use_pos_embeds_from_orig=use_pos_embeds_from_orig,
                use_context_bridge_tokens=use_context_bridge_tokens,
                pad_to_square=False,
                prompt_settings=prompt_settings,
                attention_setting=attention_setting,
                larger_latent_boxes=False,
            ).images

            main_result = images_list[0][0]
            save_partial_results_for_metrics(
                result_save_path,
                source_image,
                main_result,
                image_prefix,
                instance_bboxes_xyxy_normalized,
                text_sources,
                async_save=True,
            )
            del images_list
        except Exception as e:
            logger.exception(f"Failed to process sample {idx} (ID: {sample.get('id', '<unknown>')}): {e}; continuing with next sample.")
            continue


def _save_partial_results_for_metrics(result_save_path, source_image, main_result, image_prefix, instance_bboxes_xyxy_normalized, text_sources):
    main_result.save(result_save_path / f"{image_prefix}_result.png")
    main_result_w, main_result_h = main_result.size

    partial_dir = result_save_path / f'{image_prefix}_partial'
    partial_dir.mkdir(exist_ok=True, parents=True)

    for text_idx, (text_bbox, text_source) in enumerate(zip(instance_bboxes_xyxy_normalized, text_sources)):
        x1, y1, x2, y2 = text_bbox
        x1 = int(x1 * main_result_w)
        y1 = int(y1 * main_result_h)
        x2 = int(x2 * main_result_w)
        y2 = int(y2 * main_result_h)

        partial_result = main_result.crop((x1, y1, x2, y2))
        partial_result.save(partial_dir / f"{text_idx:02d}_generated_text_rgb.png")

        source_image_cropped = source_image.crop((x1, y1, x2, y2))
        source_image_cropped.save(partial_dir / f"{text_idx:02d}_style_image.png")


def save_partial_results_for_metrics(result_save_path, source_image, main_result, image_prefix, instance_bboxes_xyxy_normalized, text_sources, async_save=False):
    if not async_save:
        _save_partial_results_for_metrics(
            result_save_path,
            source_image,
            main_result,
            image_prefix,
            instance_bboxes_xyxy_normalized,
            text_sources,
        )
        return None

    _SAVE_RESULTS_SEMAPHORE.acquire()

    def _task():
        try:
            _save_partial_results_for_metrics(
                result_save_path,
                source_image,
                main_result,
                image_prefix,
                instance_bboxes_xyxy_normalized,
                text_sources,
            )
        except Exception:
            logger.exception("Failed to save partial results for %s", image_prefix)
            raise
        finally:
            _SAVE_RESULTS_SEMAPHORE.release()

    try:
        return _SAVE_RESULTS_EXECUTOR.submit(_task)
    except Exception:
        _SAVE_RESULTS_SEMAPHORE.release()
        raise
    

def main():
    parser = argparse.ArgumentParser(description="Run Flux Kontext APITA inference on the LoMOE-Bench dataset.")
    parser.add_argument("--base-model-path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="Path or HF id of the base FLUX.1-Kontext-dev model.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory where results are written.")
    parser.add_argument("--bring_area_to_1024_squared", action="store_true", default=False, help="Bring the area of the images to 1024 squared")
    parser.add_argument("--prompt_settings", type=str, default='outer_local_prompts', choices=['base', 'inner_local_prompts', 'outer_local_prompts', 'outer_local_prompts_smart'], help="Prompt settings to use.")
    parser.add_argument("--attention_setting", type=str, default='APITA', choices=['full', 'APITA'], help="Attention setting to use.")
    parser.add_argument("--num_samples", type=int, required=False, default=None, help="Number of samples to process.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name.")
    parser.add_argument("--lora_path", type=str, default=None, required=False, help="Path to the LoRA checkpoint. If not provided, uses base model only.")
    parser.add_argument("--rl_lora_path", type=str, default=None, required=False, help="Path to the RL LoRA checkpoint. If not provided, uses base model only.")
    parser.add_argument("--job-index", type=int, required=False, default=None, help="The index of the current job for sharding.")
    parser.add_argument("--total-jobs", type=int, required=False, default=None, help="The total number of jobs for sharding.")
    parser.add_argument("--hard_image_attribute_binding_list", type=str, default="0,57", help="List of image attribute binding steps.")
    parser.add_argument("--use_typo_boxes", action="store_true", default=False, help="Whether to use typo boxes in the training data.")

    # LoMOE-Bench dataset paths
    parser.add_argument("--lomoe_json_path", type=str, required=True, help="Path to the LoMOE-Bench JSON file (LoMOE.json).")
    parser.add_argument("--mask_orig_prompts_path", type=str, required=True, help="Path to the LoMOE mask_orig_prompts.txt file.")
    parser.add_argument("--lomoe_base_dir", type=str, required=True, help="Base directory for the LoMOE-Bench dataset images.")

    args = parser.parse_args()

    def str2list(string):
        return [int(item) for item in string.split(',')]
    args.hard_image_attribute_binding_list = str2list(args.hard_image_attribute_binding_list)
    if len(args.hard_image_attribute_binding_list) > 0:
        args.hard_image_attribute_binding_list = list(range(args.hard_image_attribute_binding_list[0], args.hard_image_attribute_binding_list[1]))
    else:
        args.hard_image_attribute_binding_list = list(range(0, 57))

    # Load LoMOE dataset
    logger.info("Loading LoMOE dataset v2...")
    dataset = LoMOEDatasetV2(
        json_path=getattr(args, 'lomoe_json_path'),
        mask_orig_prompts_path=getattr(args, 'mask_orig_prompts_path'),
        base_dir=getattr(args, 'lomoe_base_dir')
    )
    
    if args.num_samples is not None:
        from torch.utils.data import Subset
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = Subset(dataset, indices)
        logger.info(f"Limited to {len(dataset)} samples")
    
    if args.job_index is not None and args.total_jobs is not None:
        total_samples = len(dataset)
        samples_per_job = (total_samples + args.total_jobs - 1) // args.total_jobs
        start_idx = args.job_index * samples_per_job
        end_idx = min(start_idx + samples_per_job, total_samples)
        
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            original_indices = dataset.indices[start_idx:end_idx]
            dataset = Subset(dataset.dataset, original_indices)
        else:
            indices = list(range(start_idx, end_idx))
            dataset = Subset(dataset, indices)
        
        logger.info(f"Job {args.job_index}/{args.total_jobs-1}: processing {len(dataset)} samples (indices {start_idx}-{end_idx-1}) from LoMOE")
    else:
        logger.info(f"Processing {len(dataset)} samples from LoMOE")

    run_name = f'lomoe_test_{args.exp_name}'
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {save_dir}")

    infer_kontext(dataset, save_dir, args.lora_path, args.rl_lora_path, args.base_model_path, args, disable_inner_pbar=False)
    
    logger.info('Finished!')


if __name__ == '__main__':
    main()
