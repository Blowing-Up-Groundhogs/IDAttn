import os
import statistics
from datetime import datetime
import torch
from accelerate.logging import get_logger
from diffusers.utils import check_min_version
from transformers import PretrainedConfig
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator, DistributedType, init_empty_weights, load_checkpoint_and_dispatch
import transformers
import logging
from accelerate.logging import get_logger

from kontext.pipeline_utils import  find_inner_sentence_token_span_t5
from kontext.pipeline_flux_kontext import _get_t5_prompt_embeds

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")
logger = get_logger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    prompt_2: str,
    max_sequence_length,
    prompt_settings: str,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds, instance_text_index_lst, seq_len = _get_t5_prompt_embeds(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt_2,
        prompt_settings=prompt_settings,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids, instance_text_index_lst, seq_len

def unwrap_model(accelerator, model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def sequentially_load_text_encoders(accelerator, logger, models, pretrained_model_name_or_path, revision, variant, load_dtype):
    # Load text encoders sequentially
    for i in range(accelerator.num_processes):
        if i == accelerator.process_index:
            logger.info(
                "Process %s (global_rank=%s, local_rank=%s, hostname=%s): Loading text encoders...",
                i,
                accelerator.process_index,
                accelerator.local_process_index,
                os.environ.get("HOSTNAME", "unknown"),
            )
            text_encoder_one, text_encoder_two = load_text_encoders(
                pretrained_model_name_or_path,
                revision,
                variant,
                models[0],
                models[1],
                torch_dtype=load_dtype,
            )
            logger.info(
                "Process %s (global_rank=%s, local_rank=%s, hostname=%s): Text encoders loaded",
                i,
                accelerator.process_index,
                accelerator.local_process_index,
                os.environ.get("HOSTNAME", "unknown"),
            )
        accelerator.wait_for_everyone()
    return text_encoder_one, text_encoder_two


def sequentially_load_vae(accelerator, logger, model_class, pretrained_model_name_or_path, revision, variant, load_dtype):
     # Load VAE sequentially
    for i in range(accelerator.num_processes):
        if i == accelerator.process_index:
            logger.info(
                "Process %s (global_rank=%s, local_rank=%s, hostname=%s): Loading VAE...",
                i,
                accelerator.process_index,
                accelerator.local_process_index,
                os.environ.get("HOSTNAME", "unknown"),
            )
            vae = model_class.from_pretrained(
                pretrained_model_name_or_path, 
                subfolder="vae", 
                revision=revision, 
                variant=variant,
                low_cpu_mem_usage=True,
                device_map=None,  # We'll move to device manually after loading
                torch_dtype=load_dtype
            )
            logger.info(
                "Process %s (global_rank=%s, local_rank=%s, hostname=%s): VAE loaded",
                i,
                accelerator.process_index,
                accelerator.local_process_index,
                os.environ.get("HOSTNAME", "unknown"),
            )
        accelerator.wait_for_everyone()
    return vae

def sequentially_load_transformer(accelerator, logger, model_class, pretrained_model_name_or_path, revision, variant, offload_folder, load_dtype):
    # Load transformer (the largest model) sequentially
    for i in range(accelerator.num_processes):
        if i == accelerator.process_index:
            logger.info(
                "Process %s (global_rank=%s, local_rank=%s, hostname=%s): Loading Transformer...",
                i,
                accelerator.process_index,
                accelerator.local_process_index,
                os.environ.get("HOSTNAME", "unknown"),
            )
            with init_empty_weights():
                transformer = model_class.from_config(model_class.load_config(pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant))
            transformer = load_checkpoint_and_dispatch(
                transformer,
                checkpoint=os.path.join(pretrained_model_name_or_path, "transformer"),
                device_map={"": str(accelerator.device)},  # Convert torch.device to string
                no_split_module_classes=getattr(transformer, "_no_split_modules", None),
                offload_folder=offload_folder,
                dtype=load_dtype,
            )
            logger.info(
                "Process %s (global_rank=%s, local_rank=%s, hostname=%s): Transformer loaded",
                i,
                accelerator.process_index,
                accelerator.local_process_index,
                os.environ.get("HOSTNAME", "unknown"),
            )
        accelerator.wait_for_everyone()
    return transformer

def load_text_encoders(pretrained_model_name_or_path, revision, variant, class_one, class_two, torch_dtype=None):
    loader_kwargs = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "revision": revision,
        "variant": variant,
        "low_cpu_mem_usage": True,
        "device_map": None,
    }
    if torch_dtype is not None:
        loader_kwargs["torch_dtype"] = torch_dtype

    text_encoder_one = class_one.from_pretrained(subfolder="text_encoder", **loader_kwargs,)
    text_encoder_two = class_two.from_pretrained(subfolder="text_encoder_2", **loader_kwargs,)
    return text_encoder_one, text_encoder_two


def build_wandb_tags(args):
    """Build wandb tags list based on training arguments."""
    tags = []
    if args.en_src_only:
        tags.append("en_src_only")
    if args.rank is not None:
        tags.append(f"lora_rank_{args.rank}")
    if type(args.num_instances_cap) == int:
        tags.append(f"num_instances_cap_{args.num_instances_cap}")
    if args.num_instances_cap is None:
        tags.append("no_num_instances_cap")
    if args.attention_setting:
        tags.append(args.attention_setting)
    if args.lora_layers == "":
        tags.append("all_lora")
    if args.lora_layers == '"attn.to_q,attn.to_k,attn.to_v,attn.add_q_proj,attn.add_k_proj,attn.add_v_proj,attn.to_out.0,attn.to_add_out"':
        tags.append("attn_lora")
    if args.use_typo_boxes:
        tags.append("use_typo_boxes")
    if args.prompt_settings:
        tags.append(args.prompt_settings)
    if args.prompt_settings == "outer_local_prompts_smart":
        tags.append("outer_local_prompts_smart")
    return tags


def build_run_name(args, include_wandb_id=None):
    """Build run name string for output directory based on training arguments.
    
    Args:
        args: Training arguments
        include_wandb_id: Optional wandb run id to include in the name (e.g., 'wx7ua1w0')
    """
    timestamp = datetime.now().strftime("%y_%m_%d")
    layers_str = "default" if args.lora_layers == "" else args.lora_layers.replace(",", "-")
    num_instances_cap_str = f"num_instances_cap_{args.num_instances_cap}" if args.num_instances_cap is not None else "no_num_instances_cap"
    use_typo_boxes_str = "use_typo_boxes" if args.use_typo_boxes else "no_use_typo_boxes"
    en_src_only_str = "en_src_only" if args.en_src_only else "no_en_src_only"
    prompt_settings_str = args.prompt_settings if args.prompt_settings else "base"
    # add soft masking  layers range
    hard_image_attribute_binding_list_str = "-".join([str(args.hard_image_attribute_binding_list[0]), str(args.hard_image_attribute_binding_list[-1])])

    base_name = f"{timestamp}_rank{args.rank}_layers{layers_str}_attn{args.attention_setting}_lr{args.learning_rate}_{num_instances_cap_str}_{use_typo_boxes_str}_{en_src_only_str}_soft{hard_image_attribute_binding_list_str}_{prompt_settings_str}"
    
    if include_wandb_id:
        return f"{base_name}_{include_wandb_id}"
    return base_name


def create_optimizer(args, params_to_optimize, logger):
    """Create and return the optimizer based on training arguments."""
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include [adamW, prodigy]. Defaulting to adamW")
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was set to {args.optimizer.lower()}")

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning("Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0")

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def build_joint_attention_kwargs(
    attention_setting,
    instance_position_mask_list,
    instance_text_index_lst,
    seq_len,
    instance_bboxes_xyxy_normalized,
    image_w_instance_token_index_list,
    image_w_instance_token_H_list,
    image_w_instance_token_W_list,
    context_image_w_instance_token_index_list,
    hard_image_attribute_binding_list,
    is_training=False,
):
    """Build joint attention kwargs dict based on attention setting.

    Args:
        attention_setting: Type of attention masking
        instance_position_mask_list: Spatial masks for each instance
        instance_text_index_lst: Text token indices for each prompt segment
        seq_len: Total text sequence length
        instance_bboxes_xyxy_normalized: Normalized bounding boxes
        image_w_instance_token_index_list: Image token indices
        image_w_instance_token_H_list: Image token heights
        image_w_instance_token_W_list: Image token widths
        context_image_w_instance_token_index_list: Context image token indices
        hard_image_attribute_binding_list: Hard binding step list
        is_training: Whether in training mode
    """
    if attention_setting == 'full':
        return {}
    elif attention_setting == 'APITA':
        kwargs = {
            "instance_position_mask_list": instance_position_mask_list,
            "pos_instance_text_index_lst": instance_text_index_lst[0],
            "pos_seq_len": seq_len,
            "instance_bboxes_xyxy_normalized": instance_bboxes_xyxy_normalized,
            "use_bridge": False,
            "use_global_prompt": True,
            "use_context_bridge_tokens": False,
            "image_w_instance_token_index_list": image_w_instance_token_index_list,
            "image_w_instance_token_H_list": image_w_instance_token_H_list,
            "image_w_instance_token_W_list": image_w_instance_token_W_list,
            "context_image_w_instance_token_index_list": context_image_w_instance_token_index_list,
            "is_conditional": True,
            'hard_image_attribute_binding_list': hard_image_attribute_binding_list,
            'num_inference_steps': 20,
        }
        return kwargs
    else:
        raise NotImplementedError(f"Attention setting {attention_setting} is not supported in training.")


def compute_step_summary(step_data, max_grad_norm=None):
    """Compute summary statistics for a batch of steps. Returns dict of metrics."""
    summary = {
        "step_summary/mask_ratio_mean": statistics.mean(step_data["mask_ratios"]),
        "step_summary/mask_ratio_std": statistics.stdev(step_data["mask_ratios"]) if len(step_data["mask_ratios"]) > 1 else 0.0,
        "step_summary/num_instances_mean": statistics.mean(step_data["num_instances"]),
        "step_summary/masked_loss_mean": statistics.mean(step_data["masked_losses"]),
        "step_summary/masked_loss_std": statistics.stdev(step_data["masked_losses"]) if len(step_data["masked_losses"]) > 1 else 0.0,
        "step_summary/background_loss_mean": statistics.mean(step_data["background_losses"]),
        "step_summary/background_loss_std": statistics.stdev(step_data["background_losses"]) if len(step_data["background_losses"]) > 1 else 0.0,
    }
    if len(step_data["grad_norms"]) > 0:
        summary["step_summary/grad_norm_mean"] = statistics.mean(step_data["grad_norms"])
        summary["step_summary/grad_norm_max"] = max(step_data["grad_norms"])
    return summary


def compute_epoch_summary(epoch_data, max_grad_norm=None):
    """Compute summary statistics for an epoch. Returns dict of metrics."""
    summary = {
        "epoch_summary/mask_ratio_mean": statistics.mean(epoch_data["mask_ratios"]),
        "epoch_summary/mask_ratio_std": statistics.stdev(epoch_data["mask_ratios"]) if len(epoch_data["mask_ratios"]) > 1 else 0.0,
        "epoch_summary/mask_ratio_min": min(epoch_data["mask_ratios"]),
        "epoch_summary/mask_ratio_max": max(epoch_data["mask_ratios"]),
        "epoch_summary/num_instances_mean": statistics.mean(epoch_data["num_instances"]),
        "epoch_summary/num_instances_std": statistics.stdev(epoch_data["num_instances"]) if len(epoch_data["num_instances"]) > 1 else 0.0,
        "epoch_summary/masked_loss_mean": statistics.mean(epoch_data["masked_losses"]),
        "epoch_summary/masked_loss_std": statistics.stdev(epoch_data["masked_losses"]) if len(epoch_data["masked_losses"]) > 1 else 0.0,
        "epoch_summary/background_loss_mean": statistics.mean(epoch_data["background_losses"]),
        "epoch_summary/background_loss_std": statistics.stdev(epoch_data["background_losses"]) if len(epoch_data["background_losses"]) > 1 else 0.0,
    }
    if len(epoch_data["grad_norms"]) > 0 and max_grad_norm is not None:
        summary["epoch_summary/grad_norm_mean"] = statistics.mean(epoch_data["grad_norms"])
        summary["epoch_summary/grad_norm_std"] = statistics.stdev(epoch_data["grad_norms"]) if len(epoch_data["grad_norms"]) > 1 else 0.0
        summary["epoch_summary/grad_norm_max"] = max(epoch_data["grad_norms"])
        summary["epoch_summary/grad_clipping_ratio"] = sum(1 for gn in epoch_data["grad_norms"] if gn > max_grad_norm) / len(epoch_data["grad_norms"])
    return summary


class TrainingStatsAccumulator:
    """Accumulator for training statistics to enable periodic summaries."""
    
    def __init__(self):
        self.reset_step_data()
        self.reset_epoch_data()
    
    def reset_step_data(self):
        self.step_data = {
            "mask_ratios": [],
            "num_instances": [],
            "grad_norms": [],
            "masked_losses": [],
            "background_losses": [],
        }
    
    def reset_epoch_data(self):
        self.epoch_data = {
            "mask_ratios": [],
            "num_instances": [],
            "grad_norms": [],
            "masked_losses": [],
            "background_losses": [],
        }
    
    def add_sample(self, mask_ratio, num_instances, masked_loss, background_loss, grad_norm=None):
        """Add a sample's statistics."""
        for data in [self.step_data, self.epoch_data]:
            data["mask_ratios"].append(mask_ratio)
            data["num_instances"].append(num_instances)
            data["masked_losses"].append(masked_loss)
            data["background_losses"].append(background_loss)
        if grad_norm is not None:
            self.step_data["grad_norms"].append(grad_norm)
            self.epoch_data["grad_norms"].append(grad_norm)
    
    def get_step_summary(self):
        """Get step summary and reset step data."""
        if len(self.step_data["masked_losses"]) == 0:
            return None
        summary = compute_step_summary(self.step_data)
        self.reset_step_data()
        return summary
    
    def get_epoch_summary(self, max_grad_norm=None):
        """Get epoch summary and reset epoch data."""
        if len(self.epoch_data["mask_ratios"]) == 0:
            return None
        summary = compute_epoch_summary(self.epoch_data, max_grad_norm)
        self.reset_epoch_data()
        return summary