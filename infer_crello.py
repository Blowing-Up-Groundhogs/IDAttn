"""
    Script for running Flux Kontext on the Crello test dataset.
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

# Ensure repo root is on sys.path for module resolution under Slurm.
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
from tqdm import tqdm
from loguru import logger
from torchvision.transforms import functional as F
import random

from kontext.pipeline_flux_kontext import FluxKontextPipeline
from kontext.transformer_flux import FluxTransformer2DModel
from kontext.attention import get_attention_processor
from local_datasets.dataset_crello_from_json import CrelloDatasetFromJson

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


def _load_lora_weights_safe(pipe, lora_path):
    """
    Load LoRA weights, stripping incompatible config keys if needed.
    
    Handles version mismatch between training (newer PEFT) and inference (older PEFT).
    """
    import safetensors.torch
    from safetensors import safe_open
    
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
    lora_file = Path(lora_path) / "pytorch_lora_weights.safetensors"
    if not lora_file.exists():
        raise FileNotFoundError(f"Could not find LoRA weights file: {lora_file}")
    
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
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_lora_file = Path(tmpdir) / "pytorch_lora_weights.safetensors"
        safetensors.torch.save_file(state_dict, str(tmp_lora_file), metadata=metadata)
        pipe.load_lora_weights(tmpdir)
        logger.info(f"Successfully loaded sanitized LoRA")

@torch.no_grad()
def infer_kontext(crello_dataset, result_save_path, lora_path, rl_lora_path, pretrained_model_name_or_path, args, disable_inner_pbar: bool = False):
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error(f"CUDA not available! torch.cuda.is_available()={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
        logger.error(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        raise RuntimeError("No CUDA GPUs are available")
    
    device = 'cuda'
    prompt_settings = args.prompt_settings
    attention_setting = args.attention_setting
    bring_area_to_1024_squared = args.bring_area_to_1024_squared
    result_save_path.mkdir(parents=True, exist_ok=True)

    # Load models with custom transformer.
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
    
    # Load LoRA weights only if path is provided and valid
    if lora_path is not None and lora_path.strip() != '' and lora_path != 'None':
        logger.info(f"Loading SFT LoRA weights from: {lora_path}")
        try:
            # Load SFT LoRA (diffusers format) with auto-sanitization
            _load_lora_weights_safe(pipe, lora_path)
            logger.info("Successfully loaded SFT LoRA weights")
            
            # If RL LoRA path is provided, fuse SFT and load RL LoRA
            if rl_lora_path is not None and rl_lora_path.strip() != '' and rl_lora_path != 'None':
                logger.info("Fusing SFT LoRA into base model...")
                pipe.fuse_lora()
                logger.info(f"Loading RL LoRA weights from: {rl_lora_path}")
                
                # Try loading as diffusers format first, then PEFT format
                try:
                    _load_lora_weights_safe(pipe, rl_lora_path)
                    logger.info("Successfully loaded RL LoRA as diffusers format")
                except Exception as e1:
                    logger.info(f"Failed to load as diffusers format ({e1}), trying PEFT format...")
                    try:
                        from peft import PeftModel
                        # Load PEFT LoRA onto the transformer
                        pipe.transformer = PeftModel.from_pretrained(
                            pipe.transformer,
                            rl_lora_path,
                            is_trainable=False
                        )
                        logger.info("Successfully loaded RL LoRA as PEFT format")
                    except Exception as e2:
                        logger.warning(f"Failed to load RL LoRA in both formats: diffusers ({e1}), PEFT ({e2})")
                        raise e2
                
                logger.info("Fusing RL LoRA into model...")
                pipe.fuse_lora()
                logger.info("Successfully loaded and fused both SFT and RL LoRAs")
        except Exception as e:
            logger.warning(f"Failed to load LoRA weights: {e}")
            raise e
    else:
        logger.info("No LoRA path provided - using base model only")

    attention_processor = get_attention_processor(attention_setting)
    pipe.transformer.set_attn_processor(attention_processor)

    pbar = tqdm(crello_dataset, desc="Processing Images")
    for idx, sample in enumerate(pbar):
        # CRITICAL: Clear attention masks from previous sample to avoid using wrong masks
        if hasattr(attention_processor, 'clear_cached_masks'):
            attention_processor.clear_cached_masks()
        
        torch.cuda.empty_cache()
        try:
            # Use index and ID for filename
            sample_id = sample['id']
            image_prefix = f"{sample_id}_{sample['src_language']}_{sample['tgt_language']}_{idx:06d}"
            
            if (result_save_path / f"{image_prefix}_result.png").exists():
                logger.debug(f"Skipping sample {idx} (ID: {sample_id}) as it has already been processed.")
                continue

            # Get bounding boxes and text
            instance_bboxes_xyxy_normalized = sample['bboxes_xyxy_normalized']
            if not instance_bboxes_xyxy_normalized:
                logger.debug(f"Skipping sample {idx} (ID: {sample_id}) as it has no text annotations.")
                continue

            text_sources = sample['text_source']
            text_targets = sample['text_target']
            global_prompt = sample['global_prompt']

            # Get source image (PIL Image from dataset)
            source_image = sample['image_source']
            
            # Calculate character lengths for bbox adjustment
            characters_source = [len(text) for text in text_sources]
            characters_target = [len(text) for text in text_targets]

            # Get image dimensions
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
            
            # Resize if needed
            if width != source_image_w or height != source_image_h:
                source_image = source_image.resize((width, height))

            # Build prompts
            prompts = []
            for text_source, text_target in zip(text_sources, text_targets):
                prompts.append(f'Change "{text_source}" to "{text_target}".')
            
            # Insert global prompt at the beginning
            global_prompt = " "
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

            # FOR DEBUG ----
            if os.environ.get("DEBUGPY", "0") == "1":
                mask_idx = f'{image_prefix}_crello_sample'
                Path(f'{mask_idx}').mkdir(exist_ok=True)
                for image_idx, image in enumerate(images_list[:len(images_list)//2]):
                    image[0].save(f'{mask_idx}/{mask_idx}_{image_idx}.png') 
                for image_idx, image in enumerate(images_list[len(images_list)//2:]):
                    image[0].save(f'{mask_idx}/{mask_idx}_context_{image_idx}.png') 

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


# Saving partial results can be slow on networked storage, so we optionally offload it to a background thread
def _save_partial_results_for_metrics(result_save_path, source_image, main_result, image_prefix, instance_bboxes_xyxy_normalized, text_sources):
    # Saves the result in the format required for metrics evaluation
    
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
    parser = argparse.ArgumentParser(description="Run Flux Kontext APITA inference on the Crello test dataset.")
    parser.add_argument("--base-model-path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="Path or HF id of the base FLUX.1-Kontext-dev model.")
    parser.add_argument("--crello-json-dir", type=str, required=True, help="Directory with the processed Crello JSON files (expects test_dataset.json).")
    parser.add_argument("--crello-images-dir", type=str, required=True, help="Directory with the Crello source images.")
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
    parser.add_argument("--use_filtered_version_test", action="store_true", default=False, help="Use the filtered version of the dataset for sampling.")
    args = parser.parse_args()

    def str2list(string):
        return [int(item) for item in string.split(',')]
    args.hard_image_attribute_binding_list = str2list(args.hard_image_attribute_binding_list)
    if len(args.hard_image_attribute_binding_list) > 0:
        args.hard_image_attribute_binding_list = list(range(args.hard_image_attribute_binding_list[0], args.hard_image_attribute_binding_list[1]))
    else:
        args.hard_image_attribute_binding_list = list(range(0, 57))
    # Load Crello test dataset
    logger.info("Loading Crello test dataset...")
    crello_dataset = CrelloDatasetFromJson(
        json_directory=Path(args.crello_json_dir),
        split='test',
        images_directory=Path(args.crello_images_dir),
        load_tgt_image=False,
        load_only_valid_boxes=False,
        en_src_only=True,
        enable_bucketing=not args.bring_area_to_1024_squared,
        bring_area_to_1024_squared=args.bring_area_to_1024_squared,
        return_type='pil',
        use_typo_boxes=args.use_typo_boxes,
        use_filtered_version_test=args.use_filtered_version_test,
    )
    
    if args.num_samples is not None:
        # Create a subset of the dataset
        from torch.utils.data import Subset
        indices = list(range(min(args.num_samples, len(crello_dataset))))
        crello_dataset = Subset(crello_dataset, indices)
        logger.info(f"Limited to {len(crello_dataset)} samples")
    
    # Auto-detect SLURM environment if arguments are not provided
    if args.job_index is None and "SLURM_PROCID" in os.environ:
        args.job_index = int(os.environ["SLURM_PROCID"])
        logger.info(f"Auto-detected job index from SLURM_PROCID: {args.job_index}")
    
    if args.total_jobs is None:
        if "SLURM_NTASKS" in os.environ:
             args.total_jobs = int(os.environ["SLURM_NTASKS"])
             logger.info(f"Auto-detected total jobs from SLURM_NTASKS: {args.total_jobs}")
        elif "SLURM_NPROCS" in os.environ:
             args.total_jobs = int(os.environ["SLURM_NPROCS"]) 
             logger.info(f"Auto-detected total jobs from SLURM_NPROCS: {args.total_jobs}")

    # Shard dataset for parallel processing if job-index and total-jobs are provided
    if args.job_index is not None and args.total_jobs is not None:
        # For HuggingFace datasets, use .shard method
        if hasattr(crello_dataset, 'shard'):
            crello_dataset = crello_dataset.shard(num_shards=args.total_jobs, index=args.job_index)
            logger.info(f"Job {args.job_index}/{args.total_jobs-1}: processing {len(crello_dataset)} samples from Crello test set")
        else:
            # For Subset or other dataset types, manually shard by selecting indices
            total_samples = len(crello_dataset)
            samples_per_job = (total_samples + args.total_jobs - 1) // args.total_jobs
            start_idx = args.job_index * samples_per_job
            end_idx = min(start_idx + samples_per_job, total_samples)
            
            from torch.utils.data import Subset
            if isinstance(crello_dataset, Subset):
                # If already a Subset, we need to subset the subset
                original_indices = crello_dataset.indices[start_idx:end_idx]
                crello_dataset = Subset(crello_dataset.dataset, original_indices)
            else:
                indices = list(range(start_idx, end_idx))
                crello_dataset = Subset(crello_dataset, indices)
            
            logger.info(f"Job {args.job_index}/{args.total_jobs-1}: processing {len(crello_dataset)} samples (indices {start_idx}-{end_idx-1}) from Crello test set")
    else:
        logger.info(f"Processing {len(crello_dataset)} samples from Crello test set")

    run_name = f'crello_test_{args.exp_name}'
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {save_dir}")

    infer_kontext(crello_dataset, save_dir, args.lora_path, args.rl_lora_path, args.base_model_path, args, disable_inner_pbar=False)
    
    logger.info('Finished!')


if __name__ == '__main__':
    main()

