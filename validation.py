from contextlib import nullcontext

import numpy as np
import torch
from accelerate.logging import get_logger
from diffusers.training_utils import free_memory
from diffusers.utils import check_min_version, is_wandb_available
from PIL import Image
from PIL import ImageDraw
import random
import os
import time
from torchvision.transforms import functional as F
import math

if is_wandb_available():
    import wandb

from kontext.attention import get_attention_processor
from kontext.pipeline_utils import create_larger_latent_bboxes

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)


def process_crello_batch(validation_batch):
    instance_bboxes_xyxy_normalized = validation_batch["bboxes_xyxy_normalized"][0]
    image_source = validation_batch["image_source"][0]
    image_target = validation_batch["image_target"][0]
    global_prompts = validation_batch["global_prompt"]
    text_sources = validation_batch["text_source"]
    text_targets = validation_batch["text_target"] 

    # bucket_idx = validation_batch["bucket_idx"][0]

    # Use empty prompts to preserve background (text editing, not image generation)
    prompts = [""] * len(global_prompts)
    prompts2 = []   # len(prompts2) = batch_size
    for global_prompt, text_source, text_target in zip(global_prompts, text_sources, text_targets):
        local_prompts = [f'Change "{source}" to "{target}."' for source, target in zip(text_source, text_target)]
        # Empty global prompt = "edit only, don't regenerate"
        prompt2_global_prompt = ""
        prompt2 = '$BREAKFLAG$'.join([prompt2_global_prompt] + local_prompts)
        prompts2.append(prompt2)

    return instance_bboxes_xyxy_normalized, image_source, image_target, text_targets, prompts, prompts2


@torch.no_grad()
def log_validation(
    pipeline,
    args,
    accelerator,
    valid_dataloader,
    epoch,
    torch_dtype,
    prompt_settings,
    attention_setting,
    num_validation_images,
    hard_image_attribute_binding_list,
    is_final_validation=False,
    global_step=None,
    dataset_name=None,
    return_payload_only=False,
):
    logger.info(f"Running validation... \n Generating {num_validation_images} images.")

    array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    # Stagger by 0.2 seconds per task ID + 0.05 seconds per process rank
    delay = array_task_id * 0.2 + accelerator.process_index * 0.05
    logger.info(f"[Array Task {array_task_id}, Rank {accelerator.process_index}] Waiting {delay:.2f}s before validation to avoid concurrent file access...")
    time.sleep(delay)

    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)
    attention_processor = get_attention_processor(attention_setting)
    pipeline.transformer.set_attn_processor(attention_processor)
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    validation_images_list = []
    validation_prompts_list = []

    # Retry/backoff to handle occasional PermissionError (e.g., Lustre contention)
    max_retries = 5
    retry_delay = 0.5
    local_rng = random.Random(array_task_id * 1000 + accelerator.process_index)
    dataset_iter = iter(valid_dataloader)
    sample_idx = 0
    dataset_exhausted = False

    while len(validation_images_list) < num_validation_images and not dataset_exhausted:
        validation_batch = None
        for attempt in range(max_retries):
            try:
                validation_batch = next(dataset_iter)
                sample_idx += 1
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt) + local_rng.uniform(0, 0.5)
                    logger.warning(f"PermissionError accessing sample {sample_idx}, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"PermissionError persisted after {max_retries} attempts, skipping validation")
                    raise
            except StopIteration:
                logger.warning(f"Reached end of dataset after {sample_idx} samples")
                dataset_exhausted = True
                break

        if dataset_exhausted or validation_batch is None:
            break
        
        if hasattr(attention_processor, 'clear_cached_masks'):
            attention_processor.clear_cached_masks()

        if dataset_name == "crello":
            instance_bboxes_xyxy_normalized, image_source, image_target, text_targets, prompts, prompts2 = process_crello_batch(validation_batch)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        # pre-calculate prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
        prompt_embeds, pooled_prompt_embeds, text_ids, batched_instance_text_index_lst, seq_len = pipeline.encode_prompt(
            prompt_settings,
            prompts, prompts2,
            device=accelerator.device,
            max_sequence_length=args.max_sequence_length)
        
        height, width = image_source.height, image_source.width
        
        with autocast_ctx:
            images_list = pipeline(
                image=image_source,
                guidance_scale=4.0,
                height=height,
                width=width,
                _auto_resize=False,
                num_inference_steps=20,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                instance_bboxes_xyxy_normalized=instance_bboxes_xyxy_normalized,
                text_ids=text_ids,
                batched_instance_text_index_lst=batched_instance_text_index_lst,
                seq_len=seq_len,
                generator=generator,
                prompt_settings=prompt_settings,
                use_bridge=False,
                attention_setting=attention_setting,
                pad_to_square=False,
                use_context_bridge_tokens=False,
                use_pos_embeds_from_orig=True,
                use_global_prompt=True,
                num_hard_control_steps=20,
                hard_image_attribute_binding_list=hard_image_attribute_binding_list,
            ).images

            generated_image = images_list[0][0]
            # here there are all results for the current batch, current image, eventual bridges, context, eventual context bridges
            # I will only keep the actual result but keeping all for debugging purposes
            
            image_for_boxes_to_edit = image_source.copy()
            draw = ImageDraw.Draw(image_for_boxes_to_edit)
            for bbox in instance_bboxes_xyxy_normalized:
                x1, y1, x2, y2 = bbox
                x1 = int(x1 * image_source.width)
                y1 = int(y1 * image_source.height)
                x2 = int(x2 * image_source.width)
                y2 = int(y2 * image_source.height)
                # Generate a random color for each box
                color = tuple(random.randint(64, 255) for _ in range(3))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                # Optionally, draw a semi-transparent fill
                fill_color = color + (10,)
                draw.rectangle([x1, y1, x2, y2], outline=color, fill=fill_color, width=4)

            image_for_comparison_log_width = image_source.width + generated_image.width + image_for_boxes_to_edit.width
            if dataset_name == "crello":
                image_for_comparison_log_width += image_target.width

            image_for_comparison_log = Image.new('RGB', (image_for_comparison_log_width, height), color=(255, 255, 255))
            image_for_comparison_log.paste(image_source, (0, 0))
            image_for_comparison_log.paste(generated_image, (image_source.width, 0))

            if dataset_name == "crello":
                image_for_comparison_log.paste(image_target, (image_source.width + generated_image.width, 0))
                image_for_comparison_log.paste(image_for_boxes_to_edit, (image_source.width + generated_image.width + image_target.width, 0))
            else:
                image_for_comparison_log.paste(image_for_boxes_to_edit, (image_source.width + generated_image.width, 0))
            validation_images_list.append(image_for_comparison_log)
            validation_prompts_list.append(prompts2)

    # Return payload for external aggregation/logging if requested
    if return_payload_only:
        phase_name = f"test_{dataset_name}" if is_final_validation else f"validation_{dataset_name}"
        payload = []
        for i, image in enumerate(validation_images_list):
            caption = (
                f"{i}: source | result | target | boxes | prompt: {validation_prompts_list[i]}"
                if dataset_name == "crello"
                else f"{i}: source | result | boxes | prompt: {validation_prompts_list[i]}"
            )
            payload.append(
                {
                    "phase": phase_name,
                    "step": global_step if global_step is not None else epoch,
                    "image": image,
                    "caption": caption,
                }
            )
        del pipeline
        free_memory()
        return payload

    for tracker in accelerator.trackers:
        phase_name = f"test_{dataset_name}" if is_final_validation else f"validation_{dataset_name}"
        step_value = global_step if global_step is not None else epoch
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in validation_images_list])
            tracker.writer.add_images(phase_name, np_images, step_value, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(
                            image, 
                            caption=(
                                f"{i}: source | result | target | boxes | prompt: {validation_prompts_list[i]}"
                                if dataset_name == "crello"
                                else f"{i}: source | result | boxes | prompt: {validation_prompts_list[i]}"
                            ),
                        ) for i, image in enumerate(validation_images_list)
                    ]
                },
                step=step_value,
            )

    del pipeline
    free_memory()

    return validation_images_list