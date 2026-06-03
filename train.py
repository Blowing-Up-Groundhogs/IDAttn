#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=0.31.0",
#     "transformers>=4.41.2",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "peft>=0.11.1",
#     "sentencepiece",
#     "torchvision",
#     "datasets",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from datetime import timedelta
from contextlib import nullcontext
from pathlib import Path
import json

# Suppress FutureWarning from apex about deprecated torch.cuda.amp.autocast
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*", category=FutureWarning)

if os.environ.get("DEBUGPY", "0") == "1":
    import debugpy
    port = 5679 + int(os.environ.get("LOCAL_RANK", 0))
    debugpy.listen(("0.0.0.0", port))
    print(f"[rank={os.environ.get('RANK')}] debugpy listening on port {port}", flush=True)
    # Optional: only block on rank 0
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        debugpy.wait_for_client()

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedType, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
    gather_object,
)
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
import torchvision
import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, is_wandb_available, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module
if is_wandb_available():
    import wandb

from hf_utils import save_model_card
from kontext.transformer_flux import FluxTransformer2DModel
from kontext.pipeline_utils import create_position_mask_list, find_inner_sentence_token_span_t5, create_larger_latent_bboxes
from kontext.pipeline_flux_kontext import FluxKontextPipeline
from local_datasets.dataset_crello_from_json import BucketBatchSampler, CrelloDatasetFromJson, crello_from_json_collate_fn
from train_utils import (
    import_model_class_from_model_name_or_path, load_text_encoders, sequentially_load_text_encoders, 
    sequentially_load_vae, sequentially_load_transformer, encode_prompt,
    build_wandb_tags, build_run_name, create_optimizer, build_joint_attention_kwargs, TrainingStatsAccumulator
)
from validation import log_validation
from kontext.attention import get_attention_processor



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")
logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--bring_area_to_1024_squared", action="store_true", default=False, help="Bring the area of the images to 1024 squared")
    parser.add_argument("--base-model-path", type=str, required=True, help="Path to the FLUX.1-Kontext-dev base model directory.")
    parser.add_argument("--crello-json-dir", type=str, required=True, help="Path to the Crello processed-json directory.")
    parser.add_argument("--crello-images-dir", type=str, required=True, help="Path to the Crello images directory.")

    parser.add_argument("--hard_image_attribute_binding_list", type=str, default="0,57", help="List of image attribute binding steps.")
    parser.add_argument("--use_typo_boxes", type=str, default="False", help="Whether to use typo boxes in the training data.")

    parser.add_argument("--prompt_settings", type=str, default='outer_local_prompts', choices=['base', 'inner_local_prompts', 'outer_local_prompts', 'outer_local_prompts_smart'], help="Prompt settings to use.")
    parser.add_argument("--attention_setting", type=str, default='APITA', choices=['full', 'APITA'], help="Attention setting to use.")
    parser.add_argument("--num_instances_cap", type=int, default=None, help="Number of instances to cap the samples at.")
    parser.add_argument("--en_src_only", type=str, default="False", help="Whether to only use the English source images.")

    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--vae_encode_mode", type=str, default="mode", choices=["sample", "mode"], help="VAE encoding mode.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--dataset_name", type=str, default="crello", choices=["crello"], help=("The name of the Dataset"))

    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length to use with with the T5 text encoder")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images that should be generated during validation.")
    parser.add_argument("--validation_steps", type=int, default=None, help="Run validation every X steps.")
    parser.add_argument(
        "--validation_on_all_ranks",
        action="store_true",
        default=False,
        help=(
            "If set, run validation on every rank and aggregate results via all_gather_object. "
            "By default, validation runs only on the main process to reduce stragglers and avoid "
            "distributed collective timeouts during long inference."
        ),
    )
    parser.add_argument("--rank", type=int, default=4, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=int, default=4, help="LoRA alpha to be used for additional scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument("--output_dir", type=str, default="flux-kontext-lora", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="the FLUX.1 dev variant is a guidance distilled model")
    
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"], help='We default to the "none" weighting scheme for uniform sampling and uniform loss')
    parser.add_argument("--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.")
    parser.add_argument("--optimizer", type=str, default="AdamW", help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')

    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--prodigy_beta3", type=float, default=None, help="coefficients for computing the Prodigy stepsize using running averages. If set to None, uses the value of square root of beta2. Ignored if optimizer is adamW")
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder")
    
    parser.add_argument("--lora_layers", type=str, default="", help='The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only')
    
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True, help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW")
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True, help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. Ignored if optimizer is adamW")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_log_interval", type=int, default=10, help="Log trainable weight stats every N optimizer steps (0 to disable).")
    parser.add_argument("--logging_dir", type=str, default="logs", help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--offload_folder", type=str, default=None, help="Directory to use for CPU/GPU offloading when loading large checkpoints.")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--report_to", type=str, default="wandb", help='The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--wandb_project_name", type=str, default=None, help="Optional Weights & Biases project name to use for logging.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    parser.add_argument("--upcast_before_saving", action="store_true", default=False, help="Whether to upcast the trained transformer layers to float32 before saving (at the end of training). Defaults to precision dtype used for training to save memory")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training to first N samples (for debugging/overfitting tests). If set, validation uses same samples.")
    
    parser.add_argument("--mask_loss_weight", type=float, default=1.0, help="Loss weight multiplier inside masked regions.")
    parser.add_argument("--background_loss_weight", type=float, default=1.0, help="Loss weight multiplier outside masked regions.")
    parser.add_argument("--use_bridge", type=bool, default=False, help="Use bridge tokens in attention masking during training.")
    parser.add_argument("--use_context_bridge_tokens", type=bool, default=False, help="Use context bridge tokens during training.")
    parser.add_argument("--offload_vae_and_encoders", type=str, default="False", help="Offload VAE and text encoders to CPU during transformer forward/backward to save VRAM.")
    parser.add_argument("--cache_clear_interval", type=int, default=0, help="Clear CUDA cache every N steps (0 to disable).")

    # Dynamic memory management for OOM prevention
    parser.add_argument("--max_memory_score", type=int, default=2000, help="Max memory score (num_instances * avg_chars). Samples exceeding this are dynamically truncated. Set to 0 to disable.")
    parser.add_argument("--ruin_text_areas", type=str, default="False", help="If True, ruin the text areas by putting a black rectangle over them.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.en_src_only = args.en_src_only.lower() == "true"
    args.use_typo_boxes = args.use_typo_boxes.lower() == "true"
    args.offload_vae_and_encoders = args.offload_vae_and_encoders.lower() == "true"
    args.ruin_text_areas = args.ruin_text_areas.lower() == "true"

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.pretrained_model_name_or_path = args.base_model_path

    def str2list(string):
        return [int(item) for item in string.split(',')]
    args.hard_image_attribute_binding_list = str2list(args.hard_image_attribute_binding_list)
    if len(args.hard_image_attribute_binding_list) > 0:
        args.hard_image_attribute_binding_list = list(range(args.hard_image_attribute_binding_list[0], args.hard_image_attribute_binding_list[1]))
    else:
        args.hard_image_attribute_binding_list = list(range(0, 57))

    return args


def main(args):
    # Avoid tokenizers fork/parallelism warnings (and potential deadlocks) when using DataLoader workers.
    # Keep this very early so it applies before tokenizers are instantiated.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    # IMPORTANT: PyTorch ProcessGroupNCCL has its own watchdog timeout (default often 10 minutes).
    # Long validation / inference phases can make non-main ranks wait in collectives and trip this.
    # We set a higher default here to match typical HPC validation duration, overridable via env var.
    ddp_timeout_sec = int(os.environ.get("KONTEXT_DDP_TIMEOUT_SEC", "3600"))
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=ddp_timeout_sec))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, init_pg_kwargs],
    )

    args.num_gpus = accelerator.num_processes

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    if args.wandb_project_name and args.report_to not in ("wandb", "all"):
        logger.warning("wandb_project_name was provided but report_to is set to '%s'; wandb project name will be ignored.", args.report_to)
    if args.wandb_project_name and args.report_to in ("wandb", "all"):
        os.environ["WANDB_PROJECT"] = args.wandb_project_name

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(
        "[rank=%s] DDP timeout configured as %ss (env KONTEXT_DDP_TIMEOUT_SEC=%s)",
        os.environ.get("RANK", "unknown"),
        ddp_timeout_sec,
        os.environ.get("KONTEXT_DDP_TIMEOUT_SEC", "<unset>"),
    )
    # Explicitly log the global and local rank for troubleshooting multi-node runs
    logger.info(
        "[rank=%s | local_rank=%s | machine_rank=%s] Accelerator state initialised",
        os.environ.get("RANK", "unknown"),
        os.environ.get("LOCAL_RANK", "unknown"),
        os.environ.get("SLURM_NODEID", "unknown"),
    )
    # Log the hard image attribute binding list
    logger.info(f"Hard image attribute binding list: {args.hard_image_attribute_binding_list}")
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Build the full output directory path based on training parameters
    if args.output_dir is not None:
        run_name = build_run_name(args)
        args.output_dir = os.path.join(args.output_dir, run_name)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")

    # Determine dtype to use when loading large checkpoints
    load_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32

    if args.offload_folder is not None and accelerator.is_main_process:
        os.makedirs(args.offload_folder, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Load models sequentially in distributed training to reduce peak CPU memory usage
    # Each process loads the large models one at a time instead of all at once
    if accelerator.is_main_process:
        logger.info(f"Loading models sequentially across {accelerator.num_processes} processes...")

    text_encoder_one, text_encoder_two = sequentially_load_text_encoders(accelerator, logger, [text_encoder_cls_one, text_encoder_cls_two], args.pretrained_model_name_or_path, args.revision, args.variant, load_dtype)
    vae = sequentially_load_vae(accelerator, logger, AutoencoderKL, args.pretrained_model_name_or_path, args.revision, args.variant, load_dtype)
    transformer = sequentially_load_transformer(accelerator, logger, FluxTransformer2DModel, args.pretrained_model_name_or_path, args.revision, args.variant, args.offload_folder, load_dtype)
    
    if accelerator.is_main_process:
        logger.info("All models loaded successfully across all processes")

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # Set non-trainable models to eval mode to disable dropout/batch norm
    vae.eval()
    text_encoder_one.eval()
    text_encoder_two.eval()

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")

    # Move models to device - transformer is already dispatched by load_checkpoint_and_dispatch
    # so we only move VAE and text encoders (which were loaded with correct dtype already)
    vae.to(accelerator.device)
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    # Note: transformer is already on accelerator.device with correct dtype from load_checkpoint_and_dispatch

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers != "":
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # target_modules = [
        #     "attn.to_k",
        #     "attn.to_q",
        #     "attn.to_v",
        #     "attn.to_out.0",
        #     "attn.add_k_proj",
        #     "attn.add_q_proj",
        #     "attn.add_v_proj",
        #     "attn.to_add_out",
        #     "ff.net.0.proj",
        #     "ff.net.2",
        #     "ff_context.net.0.proj",
        #     "ff_context.net.2",
        #     "proj_mlp",
        # ]
        target_modules = "(.*x_embedder|.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1\\.linear|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2|.*single_transformer_blocks\\.[0-9]+\\.norm\\.linear|.*single_transformer_blocks\\.[0-9]+\\.proj_mlp|.*single_transformer_blocks\\.[0-9]+\\.proj_out|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)"


    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config) 

    def unwrap_model(model):
        # Handle torch.compile wrapper first
        if is_compiled_module(model):
            model = model._orig_mod
        # Then handle accelerator wrapper
        model = accelerator.unwrap_model(model)
        # Check again for compiled module after unwrapping
        if is_compiled_module(model):
            model = model._orig_mod
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_one))):
                    model = unwrap_model(model)
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["text_encoder"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            FluxKontextPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_one))):
                    text_encoder_one_ = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

        else:
            # Load for checkpoint resumption - use same dtype as training
            checkpoint_load_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (
                torch.float16 if args.mixed_precision == "fp16" else torch.float32
            )
            transformer_ = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, 
                subfolder="transformer",
                low_cpu_mem_usage=True,
                device_map=None,
                torch_dtype=checkpoint_load_dtype
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Enable optimized attention backends (Flash Attention, Memory Efficient Attention)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    logger.info("✓ Enabled Flash Attention and Memory Efficient Attention backends")

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Count and log LoRA parameters
    lora_trainable_params = sum(p.numel() for p in transformer_lora_parameters)
    total_transformer_params = sum(p.numel() for p in transformer.parameters())
    lora_percentage = 100 * lora_trainable_params / total_transformer_params
    logger.info(f"Transformer LoRA trainable params: {lora_trainable_params:,} ({lora_percentage:.2f}% of {total_transformer_params:,} total)")

    # Optimization parameters
    params_to_optimize = []
    if len(transformer_lora_parameters) > 0:
        transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
        params_to_optimize.append(transformer_parameters_with_lr)

    # Sanity check: must have something to train
    assert len(params_to_optimize) > 0, "No parameters to optimize!"

    # Log total number of parameters being optimized
    total_optimized_params = sum(p.numel() for group in params_to_optimize for p in group.get("params", []))
    logger.info(f"Total optimized parameters: {total_optimized_params:,}")
        
    # Optimizer creation
    optimizer = create_optimizer(args, params_to_optimize, logger)

    if args.dataset_name == "crello":
        train_dataset = CrelloDatasetFromJson(
            json_directory=Path(args.crello_json_dir),
            split='train',
            images_directory=Path(args.crello_images_dir),
            json_path=Path(args.crello_json_dir) / "train_dataset_filtered.json",
            en_src_only=args.en_src_only,
            num_instances_cap=args.num_instances_cap,
            enable_bucketing=not args.bring_area_to_1024_squared,
            bring_area_to_1024_squared=args.bring_area_to_1024_squared,
            return_type='tensor',
            use_typo_boxes=args.use_typo_boxes,
            load_only_valid_boxes=True,
            ruin_text_areas=args.ruin_text_areas,
        )
        # Create validation dataset - use same samples if max_train_samples is set
        if args.max_train_samples is not None:
            # Limit to N samples
            subset_indices = list(range(min(args.max_train_samples, len(train_dataset))))
            train_dataset_subset = torch.utils.data.Subset(train_dataset, subset_indices)

            valid_dataset = CrelloDatasetFromJson(
                json_directory=Path(args.crello_json_dir),
                split='train',
                images_directory=Path(args.crello_images_dir),
                en_src_only=args.en_src_only,
                num_instances_cap=args.num_instances_cap,
                enable_bucketing=not args.bring_area_to_1024_squared,
                bring_area_to_1024_squared=args.bring_area_to_1024_squared,
                return_type='pil',
                use_typo_boxes=args.use_typo_boxes,
                load_only_valid_boxes=True,
                ruin_text_areas=args.ruin_text_areas,
            )
            
            # Use the SAME samples for validation to test overfitting
            valid_dataset = torch.utils.data.Subset(valid_dataset, subset_indices)
            
            train_dataset = train_dataset_subset
            logger.info(f"Using {len(subset_indices)} samples for both training and validation (overfitting test mode)")
        else:
            valid_dataset = CrelloDatasetFromJson(
                json_directory=Path(args.crello_json_dir),
                split='validation',
                images_directory=Path(args.crello_images_dir),
                en_src_only=args.en_src_only,
                num_instances_cap=args.num_instances_cap,
                enable_bucketing=not args.bring_area_to_1024_squared,
                bring_area_to_1024_squared=args.bring_area_to_1024_squared,
                return_type='pil',
                use_typo_boxes=args.use_typo_boxes,
                load_only_valid_boxes=True,
                ruin_text_areas=args.ruin_text_areas,
            )

    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    # Dataloader / sampler setup (distributed-aware)
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    train_sampler = None
    train_batch_sampler = None

    # Use simple DataLoader for overfitting test (BucketBatchSampler uses pre-computed indices)
    if args.max_train_samples is not None:
        # train_sampler = DistributedSampler(
        #     train_dataset,
        #     num_replicas=world_size,
        #     rank=rank,
        #     shuffle=False,
        #     drop_last=True,
        # )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            # sampler=train_sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda examples: crello_from_json_collate_fn(examples), 
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if args.dataloader_num_workers > 0 else False,
        )
    else:
        train_batch_sampler = BucketBatchSampler(
            train_dataset,
            batch_size=args.train_batch_size,
            json_directory=Path(args.crello_json_dir),
            drop_last=True,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            shuffle=True,
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_sampler=train_batch_sampler, 
            collate_fn=lambda examples: crello_from_json_collate_fn(examples),  
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if args.dataloader_num_workers > 0 else False,
        )

    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        collate_fn=lambda examples: crello_from_json_collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, prompt2, text_encoders, tokenizers, prompt_settings):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids, instance_text_index_lst, seq_len = encode_prompt(text_encoders, tokenizers, prompt, prompt2, args.max_sequence_length, prompt_settings)  # TODO FABIO LOOK INTO MAX SEQ LENGHT FOR T5 PROMPTS
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids, instance_text_index_lst, seq_len

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    initial_train_dataloader_len = len(train_dataloader)
    num_warmup_steps_for_scheduler = args.lr_warmup_steps
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(initial_train_dataloader_len / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch
    else:
        num_training_steps_for_scheduler = args.max_train_steps

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(transformer, optimizer, lr_scheduler, train_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the length used when the learning rate scheduler was created ({initial_train_dataloader_len}). "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    wandb_tags = build_wandb_tags(args)

    if accelerator.is_main_process:
        tracker_run_name = "text-editing-flux-kontext-lora-correct-masks"
        tracker_project_name = tracker_run_name
        tracker_init_kwargs = {}
        if args.wandb_project_name and args.report_to in ("wandb", "all"):
            tracker_project_name = args.wandb_project_name
            tracker_init_kwargs = {"wandb": {"name": tracker_run_name, "tags": wandb_tags}}
        accelerator.init_trackers(tracker_project_name, config=vars(args), init_kwargs=tracker_init_kwargs)
        if args.report_to == "wandb" and is_wandb_available():
            wandb.config.update(vars(args), allow_val_change=True)
            # Rename output directory to include wandb run id for easy correlation
            if wandb.run is not None:
                wandb_run_id = wandb.run.id
                old_output_dir = args.output_dir
                new_run_name = build_run_name(args, include_wandb_id=wandb_run_id)
                args.output_dir = os.path.join(os.path.dirname(old_output_dir), new_run_name)
                if os.path.exists(old_output_dir) and old_output_dir != args.output_dir:
                    os.rename(old_output_dir, args.output_dir)
                    logger.info(f"Renamed output directory to include wandb run id: {args.output_dir}")

    # Ensure all ranks share the same (potentially renamed) output directory.
    #
    # NOTE: We intentionally avoid `dist.broadcast_object_list` here. On some
    # PyTorch builds this can fail with:
    #   "SymIntArrayRef expected to contain only concrete integers"
    # when allocating internal CUDA buffers. Broadcasting a fixed-size byte tensor
    # is robust under NCCL.
    accelerator.wait_for_everyone()
    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        MAX_OUTPUT_DIR_BYTES = 4096
        device = accelerator.device

        if dist.get_rank() == 0:
            encoded = args.output_dir.encode("utf-8")
            if len(encoded) > MAX_OUTPUT_DIR_BYTES:
                raise ValueError(
                    f"args.output_dir is too long to broadcast ({len(encoded)} bytes) "
                    f"(max {MAX_OUTPUT_DIR_BYTES})."
                )
            length_tensor = torch.tensor([len(encoded)], dtype=torch.int32, device=device)
            payload = torch.zeros((MAX_OUTPUT_DIR_BYTES,), dtype=torch.uint8, device=device)
            if len(encoded) > 0:
                payload[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
        else:
            length_tensor = torch.zeros((1,), dtype=torch.int32, device=device)
            payload = torch.zeros((MAX_OUTPUT_DIR_BYTES,), dtype=torch.uint8, device=device)

        dist.broadcast(length_tensor, src=0)
        dist.broadcast(payload, src=0)

        out_len = int(length_tensor.item())
        out_bytes = bytes(payload[:out_len].tolist()) if out_len > 0 else b""
        args.output_dir = out_bytes.decode("utf-8")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        resume_dir = None
        if args.resume_from_checkpoint != "latest":
            if os.path.isdir(args.resume_from_checkpoint):
                resume_dir = args.resume_from_checkpoint
            else:
                path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if resume_dir is None:
            if path is None:
                raise FileNotFoundError(f"Checkpoint '{args.resume_from_checkpoint}' not found in {args.output_dir}. Refusing to start a new run because resume was explicitly requested.")
            resume_dir = os.path.join(args.output_dir, path)

        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"Checkpoint diredctory '{resume_dir}' does not exist. Refusing to start a new run because resume was explicitly requested.")

        accelerator.print(f"Resuming from checkpoint {resume_dir}")
        accelerator.load_state(resume_dir)
        global_step = int(os.path.basename(resume_dir).split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps
        accelerator.print(f"Resume details: global_step={global_step}, first_epoch={first_epoch}, resume_step={resume_step}")
        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            raise ValueError(f"Checkpoint step ({global_step}) is >= max_train_steps ({args.max_train_steps}). Increase --max_train_steps to resume further.")

    else:
        initial_global_step = 0

    assert args.train_batch_size == 1, "Batch size must be 1 for now"   # TODO FOR NOW THIS ONLY WORKS WITH BATCH SIZE 1

    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process)
    attention_processor = get_attention_processor(args.attention_setting)
    unwrap_model(transformer).set_attn_processor(attention_processor)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    has_guidance = unwrap_model(transformer).config.guidance_embeds  
    
    validation_interval_steps = args.validation_steps
    logger.info(f"Validation will run every {validation_interval_steps} steps")
    
    # Helper to log gathered validation payloads to wandb
    def _log_wandb_validation_results(all_payloads):
        if not is_wandb_available():
            return
        import wandb

        flat = [item for sub in all_payloads for item in sub]
        if len(flat) == 0:
            return
        log_dict = {}
        for item in flat:
            phase = item["phase"]
            step_value = item["step"]
            log_dict.setdefault(phase, [])
            log_dict[phase].append(wandb.Image(item["image"], caption=item["caption"]))
        for phase, images in log_dict.items():
            accelerator.log({phase: images}, step=flat[0]["step"])

    def _gather_and_log(payload):
        # Robust check for distributed environment
        is_distributed = (
            accelerator.num_processes > 1 
            and dist.is_available() 
            and dist.is_initialized()
        )
        
        if is_distributed:
            try:
                all_payloads = [None for _ in range(accelerator.num_processes)]
                dist.all_gather_object(all_payloads, payload)
            except Exception as e:
                logger.warning(f"Distributed gather failed despite checks: {e}. Falling back to local logging.")
                all_payloads = [payload]
        else:
            all_payloads = [payload]
        
        if accelerator.is_main_process:
            _log_wandb_validation_results(all_payloads)
        
        if is_distributed:
            accelerator.wait_for_everyone()

    def _run_validation_and_log(
        pipeline,
        dataloader,
        dataset_name,
        num_images,
        epoch,
        global_step,
        is_final_validation=False,
    ):
        """Run validation and (optionally) aggregate results across ranks."""
        if args.validation_on_all_ranks:
            payload = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                valid_dataloader=dataloader,
                epoch=epoch,
                is_final_validation=is_final_validation,
                torch_dtype=weight_dtype,
                prompt_settings=args.prompt_settings,
                attention_setting=args.attention_setting,
                num_validation_images=num_images,
                hard_image_attribute_binding_list=args.hard_image_attribute_binding_list,
                global_step=global_step,
                dataset_name=dataset_name,
                return_payload_only=True,
            )
            _gather_and_log(payload)
            return

        # Default: only main process runs validation. Others simply wait.
        if accelerator.is_main_process:
            payload = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                valid_dataloader=dataloader,
                epoch=epoch,
                is_final_validation=is_final_validation,
                torch_dtype=weight_dtype,
                prompt_settings=args.prompt_settings,
                attention_setting=args.attention_setting,
                num_validation_images=num_images,
                hard_image_attribute_binding_list=args.hard_image_attribute_binding_list,
                global_step=global_step,
                dataset_name=dataset_name,
                return_payload_only=True,
            )
            _log_wandb_validation_results([payload])
        accelerator.wait_for_everyone()

    # Perform validation before training starts (all ranks run, aggregate on rank 0)
    if initial_global_step == 0:
        logger.info("Running validation before training starts...")

        # Load separate text encoders for validation (requires extra GPU memory).
        # To reduce stragglers/collective timeouts, default is main-process-only validation.
        if args.validation_on_all_ranks or accelerator.is_main_process:
            val_text_encoder_one, val_text_encoder_two = load_text_encoders(
                args.pretrained_model_name_or_path,
                args.revision,
                args.variant,
                text_encoder_cls_one,
                text_encoder_cls_two,
                torch_dtype=load_dtype,
            )
            val_text_encoder_one.to(accelerator.device)
            val_text_encoder_two.to(accelerator.device)

            pipeline = FluxKontextPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                text_encoder=unwrap_model(val_text_encoder_one),
                text_encoder_2=unwrap_model(val_text_encoder_two),
                transformer=unwrap_model(transformer),
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            pipeline = None

        _run_validation_and_log(
            pipeline=pipeline,
            dataloader=valid_dataloader,
            dataset_name=args.dataset_name,
            num_images=args.num_validation_images,
            epoch=0,
            global_step=initial_global_step,
            is_final_validation=False,
        )

        del pipeline, val_text_encoder_one, val_text_encoder_two
        free_memory()

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        # Ensure distributed samplers advance shuffling each epoch
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if valid_sampler is not None and hasattr(valid_sampler, "set_epoch"):
            valid_sampler.set_epoch(epoch)

        # Optimization: If resuming in this epoch, tell the sampler to skip batches
        # This prevents loading images for 10k+ steps which is very slow
        sampler_skipped_batches = False
        if args.resume_from_checkpoint and epoch == first_epoch and resume_step > 0:
            if train_batch_sampler is not None and hasattr(train_batch_sampler, "set_skip_batches"):
                logger.info(f"Resuming at epoch {epoch}: Skipping first {resume_step} batches efficiently via Sampler.")
                train_batch_sampler.set_skip_batches(resume_step)
                sampler_skipped_batches = True
            # If we couldn't set it on the sampler (e.g. standard implementation), we fall back to loop skipping below
            # but for BucketBatchSampler this will be much faster.

        # Track statistics for periodic summaries
        summary_interval_steps = 100  # Log summary every 100 steps
        stats_accumulator = TrainingStatsAccumulator()
        accumulated_loss = 0.0
        accumulated_loss_counter = 0

        for step, batch in enumerate(train_dataloader):
            # Fallback for samplers that don't support set_skip_batches or if using standard DistributedSampler
            # This logic is necessary because we might be using a different sampler or the sampler property wasn't set
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and not sampler_skipped_batches:
                # Fallback skip when sampler cannot skip batches.
                continue

            if hasattr(attention_processor, 'clear_cached_masks'):
                attention_processor.clear_cached_masks()
            
            # Bring VAE and text encoders back to device if they were offloaded
            if args.offload_vae_and_encoders:
                vae.to(accelerator.device)
                text_encoder_one.to(accelerator.device)
                text_encoder_two.to(accelerator.device)
            
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):

                instance_bboxes_xyxy_normalized = batch["bboxes_xyxy_normalized"][0]  # TODO THIS WILL NEED TO BE CHANGED FOR BATCHING LOGIC
                image_source = batch["image_source"]
                image_target = batch["image_target"]
                global_prompts = batch["global_prompt"]  
                text_sources = batch["text_source"] 
                text_targets = batch["text_target"] 
                # bucket_idx = batch["bucket_idx"]
                height, width = image_source.shape[2], image_source.shape[3]
                
                # ============================================================================
                # Dynamic Memory Management: Truncate instances if memory score too high
                # This is deterministic across all GPUs (same data → same decision → no deadlock)
                # ============================================================================
                if args.max_memory_score > 0:
                    num_instances = len(instance_bboxes_xyxy_normalized)
                    if num_instances > 0:
                        # Calculate memory score: num_instances * average_target_text_length
                        # Longer texts = more tokens = more memory
                        avg_text_len = sum(len(t) for t in text_targets[0]) / num_instances
                        memory_score = num_instances * avg_text_len
                        
                        # If exceeds threshold, find how many instances we can keep
                        if memory_score > args.max_memory_score:
                            # Binary search for max instances that fit
                            keep_n = num_instances
                            while keep_n > 1:
                                test_score = keep_n * avg_text_len
                                if test_score <= args.max_memory_score:
                                    break
                                keep_n -= 1
                            
                            # Truncate all relevant data structures consistently
                            if keep_n < num_instances:
                                if step % 100 == 0:  # Log occasionally to avoid spam
                                    logger.info(f"Step {step}: Truncating {num_instances} → {keep_n} instances (score {memory_score:.0f} > {args.max_memory_score})")
                                
                                instance_bboxes_xyxy_normalized = instance_bboxes_xyxy_normalized[:keep_n]
                                text_sources = [ts[:keep_n] for ts in text_sources]
                                text_targets = [tt[:keep_n] for tt in text_targets]
                                
                                # Update batch dict so downstream consumers see consistent data
                                batch["text_source"] = text_sources
                                batch["text_target"] = text_targets
                                batch["bboxes_xyxy_normalized"] = [instance_bboxes_xyxy_normalized]

                cond_pixel_values = image_source.to(device=vae.device, dtype=vae.dtype)   # [batch_size, 3, height, width]
                pixel_values = image_target.to(device=vae.device, dtype=vae.dtype)   # [batch_size, 3, height, width]

                # Use empty global prompt to focus on text editing only, not image regeneration
                # This prevents the model from changing background based on prompt description
                prompts = [""] * len(global_prompts)  # Empty for pure editing
                prompts2 = []   # len(prompts2) = batch_size
                for global_prompt, text_source, text_target in zip(global_prompts, text_sources, text_targets):
                    local_prompts = [f'Change "{source}" to "{target}."' for source, target in zip(text_source, text_target)]
                    # Use empty global prompt in T5 encoding too
                    prompt2_global_prompt = ""  # Empty = "don't regenerate, just edit"
                    prompt2 = '$BREAKFLAG$'.join([prompt2_global_prompt] + local_prompts)
                    prompts2.append(prompt2)

                # if step == 0:
                #     Image.fromarray((pixel_values*255).cpu().to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]).save('pixel_values_first_train.png')
                #     Image.fromarray((cond_pixel_values*255).cpu().to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]).save('cond_pixel_values_first_train.png')
                #     print(prompts2)

                # encode batch prompts when custom prompts are provided for each image -
                prompt_embeds, pooled_prompt_embeds, text_ids, instance_text_index_lst, seq_len = compute_text_embeddings(prompts, prompts2, text_encoders, tokenizers, args.prompt_settings)

                with torch.no_grad():
                    if args.vae_encode_mode == "sample":
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                        cond_model_input = vae.encode(cond_pixel_values).latent_dist.sample()
                    else:
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()
                
                # Delete pixel values after encoding to free memory
                del pixel_values, cond_pixel_values
                
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
                cond_model_input = (cond_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                cond_model_input = cond_model_input.to(dtype=weight_dtype)
                
                # Offload VAE and text encoders to CPU to save VRAM during transformer forward/backward
                if args.offload_vae_and_encoders:
                    vae.to("cpu")
                    text_encoder_one.to("cpu")
                    text_encoder_two.to("cpu")
                    torch.cuda.empty_cache()

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                cond_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    cond_model_input.shape[0],
                    cond_model_input.shape[2] // 2,
                    cond_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                cond_latents_ids[..., 0] = 1
                latent_image_ids = torch.cat([latent_image_ids, cond_latents_ids], dim=0)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                orig_inp_shape = packed_noisy_model_input.shape
                packed_cond_input = FluxKontextPipeline._pack_latents(
                    cond_model_input,
                    batch_size=cond_model_input.shape[0],
                    num_channels_latents=cond_model_input.shape[1],
                    height=cond_model_input.shape[2],
                    width=cond_model_input.shape[3],
                )
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_input], dim=1)

                # Kontext always has guidance
                guidance = None
                if has_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])

                instance_position_mask_list = create_position_mask_list(instance_bboxes_xyxy_normalized, height, width, vae_scale_factor)

                image_w_instance_token_H_list = [height]
                image_w_instance_token_W_list = [width]
                image_w_instance_token_index_list = [list(range(orig_inp_shape[1]))]
                context_image_w_instance_token_index_list = [list(range(orig_inp_shape[1]))]

                joint_attention_kwargs = build_joint_attention_kwargs(
                    attention_setting=args.attention_setting,
                    instance_position_mask_list=instance_position_mask_list,
                    instance_text_index_lst=instance_text_index_lst,
                    seq_len=seq_len,
                    instance_bboxes_xyxy_normalized=instance_bboxes_xyxy_normalized,
                    image_w_instance_token_index_list=image_w_instance_token_index_list,
                    image_w_instance_token_H_list=image_w_instance_token_H_list,
                    image_w_instance_token_W_list=image_w_instance_token_W_list,
                    context_image_w_instance_token_index_list=context_image_w_instance_token_index_list,
                    hard_image_attribute_binding_list=args.hard_image_attribute_binding_list,
                    is_training=True,
                )

                # TODO THINK ABT DOING SMTH LIKE CREATIDESIGN: To avoid positional embedding
                # conflicts, such as between the noise image and image condition, or between the prompt and layout
                # condition, we adopt positional encoding shifts to the image and layout condition tokens [ 95, 117 ]
                # to ensure clear separation in the token space.

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : orig_inp_shape[1]]
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # Build a per-pixel loss weight map emphasizing masked regions
                if len(instance_position_mask_list) > 0:
                    # instance masks are on token grid (H_lat/2, W_lat/2); union them
                    mask_tok = torch.stack(instance_position_mask_list, dim=0).to(model_pred.device)
                    mask_tok = (mask_tok.sum(dim=0) > 0).float()
                else:
                    # no instances -> all background
                    mask_tok = torch.zeros((model_input.shape[2] // 2, model_input.shape[3] // 2), device=model_pred.device)

                # Upsample token-grid mask to latent grid (H_lat, W_lat)
                mask_lat = F.interpolate(mask_tok.unsqueeze(0).unsqueeze(0), size=(model_input.shape[2], model_input.shape[3]), mode="nearest").squeeze(0).squeeze(0)
                # Convert to element-wise weight map
                element_weight_map = args.background_loss_weight + (args.mask_loss_weight - args.background_loss_weight) * mask_lat

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute region-weighted loss.
                per_elem_se = (model_pred.float() - target.float()) ** 2
                
                # === DEBUGGING: Compute per-region losses separately ===
                mask_lat_bool = mask_lat > 0.5
                background_lat_bool = ~mask_lat_bool
                
                # Masked region loss (without weighting)
                if mask_lat_bool.any():
                    masked_se = per_elem_se[:, :, mask_lat_bool]
                    masked_loss_unweighted = masked_se.mean()
                else:
                    masked_loss_unweighted = torch.tensor(0.0, device=per_elem_se.device)
                
                # Background region loss (without weighting)
                if background_lat_bool.any():
                    background_se = per_elem_se[:, :, background_lat_bool]
                    background_loss_unweighted = background_se.mean()
                else:
                    background_loss_unweighted = torch.tensor(0.0, device=per_elem_se.device)
                
                # Mask statistics
                mask_pixel_ratio = mask_lat_bool.float().mean()
                num_instances = len(instance_position_mask_list)
                
                # Apply regional weighting
                per_elem_se = per_elem_se * element_weight_map.unsqueeze(0).unsqueeze(0)
                loss = torch.mean((weighting.float() * per_elem_se).reshape(target.shape[0], -1), 1)
                loss = loss.mean()
                accumulated_loss += loss.detach()
                accumulated_loss_counter += 1
                accelerator.backward(loss)
                
                # Clean up intermediate tensors to free memory
                del model_pred, target, noise, per_elem_se, model_input, cond_model_input
                del packed_noisy_model_input, noisy_model_input, packed_cond_input
                del prompt_embeds, pooled_prompt_embeds, text_ids
                
                # === DEBUGGING: Compute gradient norms BEFORE clipping ===
                if accelerator.sync_gradients:
                    avg_accumulated_loss = accumulated_loss / accumulated_loss_counter
                    avg_accumulated_loss = accelerator.gather(avg_accumulated_loss).mean()
                    accumulated_loss, accumulated_loss_counter = 0.0, 0

                    params_to_clip = transformer.parameters()
                    # Calculate gradient norm before clipping
                    grad_norm_before_clip = 0.0
                    num_params_with_grad = 0
                    for p in params_to_clip:
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            grad_norm_before_clip += param_norm.item() ** 2
                            num_params_with_grad += 1
                    grad_norm_before_clip = grad_norm_before_clip ** 0.5
                    
                    # Clip gradients
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient than zeroing
                
                # Periodic CUDA cache clearing to prevent memory fragmentation
                if args.cache_clear_interval > 0 and global_step % args.cache_clear_interval == 0:
                    torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Only global rank 0 saves checkpoints
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            # Collect statistics for aggregation
            grad_norm_to_log = grad_norm_before_clip if accelerator.sync_gradients and 'grad_norm_before_clip' in locals() else None
            stats_accumulator.add_sample(
                mask_ratio=mask_pixel_ratio.item(),
                num_instances=num_instances,
                masked_loss=masked_loss_unweighted.item(),
                background_loss=background_loss_unweighted.item(),
                grad_norm=grad_norm_to_log
            )
            
            # === DEBUGGING: Enhanced logging with per-region metrics ===
            logs = {
                "loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0],
                # Per-region losses
                "loss/masked_region": masked_loss_unweighted.item(),
                "loss/background_region": background_loss_unweighted.item(),
                "loss/ratio_mask_to_bg": (masked_loss_unweighted / (background_loss_unweighted + 1e-8)).item(),
                # Mask statistics
                "mask/pixel_ratio": mask_pixel_ratio.item(),
                "mask/num_instances": num_instances,
                # Timestep info
                "timestep/value": timesteps[0].item(),
                "timestep/sigma": sigmas[0].item(),
            }
            
            # Periodic lightweight weight stats for trainable LoRA params
            if (
                args.weight_log_interval > 0
                and accelerator.sync_gradients
                and global_step > 0
                and global_step % args.weight_log_interval == 0
            ):
                if len(transformer_lora_parameters) > 0:
                    with torch.no_grad():
                        total_sq = torch.zeros((), device=transformer_lora_parameters[0].device)
                        total_abs = torch.zeros((), device=transformer_lora_parameters[0].device)
                        total_numel = 0
                        for p in transformer_lora_parameters:
                            if p.requires_grad:
                                data = p.detach()
                                total_sq += data.float().pow(2).sum()
                                total_abs += data.float().abs().sum()
                                total_numel += data.numel()
                        if total_numel > 0:
                            logs["weights/lora_l2_norm"] = torch.sqrt(total_sq).item()
                            logs["weights/lora_mean_abs"] = (total_abs / total_numel).item()

            # Add gradient norm if available (only on sync steps)
            if accelerator.sync_gradients and 'grad_norm_before_clip' in locals():
                logs["loss_all_gpus_mean"] = avg_accumulated_loss.item()
                logs["grad/norm_before_clip"] = grad_norm_before_clip
                logs["grad/clipping_active"] = float(grad_norm_before_clip > args.max_grad_norm)
            
            progress_bar.set_postfix(**{k: v for k, v in logs.items() if not k.startswith("loss/") and not k.startswith("mask/") and not k.startswith("timestep/") and not k.startswith("grad/")})
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)
            
            # Log step-based summaries every N steps
            if accelerator.sync_gradients and global_step > 0 and global_step % summary_interval_steps == 0:
                if accelerator.is_main_process:
                    step_summary = stats_accumulator.get_step_summary()
                    if step_summary:
                        if accelerator.is_main_process:
                            accelerator.log(step_summary, step=global_step)
                        logger.info(f"Step {global_step} summary (last {summary_interval_steps} steps): masked_loss={step_summary['step_summary/masked_loss_mean']:.4f}")

            # Run validation at regular intervals (all ranks run, aggregate on rank 0)
            validation_triggered = False
            if accelerator.sync_gradients:
                steps_in_current_epoch = global_step - (epoch * num_update_steps_per_epoch)
                validation_triggered = (
                    validation_interval_steps > 0
                    and global_step % validation_interval_steps == 0
                )
                # validation_triggered = (
                #     validation_interval_steps > 0
                #     and steps_in_current_epoch % validation_interval_steps == 0
                #     and steps_in_current_epoch > 0
                # )

            if validation_triggered:
                logger.info(f"Running validation at epoch {epoch}, step {steps_in_current_epoch}/{num_update_steps_per_epoch} (global step {global_step})")

                # Load separate text encoders for validation (requires extra GPU memory)
                val_text_encoder_one, val_text_encoder_two = load_text_encoders(args.pretrained_model_name_or_path, args.revision, args.variant, text_encoder_cls_one, text_encoder_cls_two, torch_dtype=load_dtype)
                val_text_encoder_one.to(accelerator.device)
                val_text_encoder_two.to(accelerator.device)
                
                pipeline = FluxKontextPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=unwrap_model(val_text_encoder_one),
                    text_encoder_2=unwrap_model(val_text_encoder_two),
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    low_cpu_mem_usage=True,
                )

                _run_validation_and_log(
                    pipeline=pipeline,
                    dataloader=valid_dataloader,
                    dataset_name=args.dataset_name,
                    num_images=args.num_validation_images,
                    epoch=epoch,
                    global_step=global_step,
                    is_final_validation=False,
                )

                del pipeline, val_text_encoder_one, val_text_encoder_two
                free_memory()

                # Set models back to training mode
                transformer.train()

            if validation_triggered:
                accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break
        
        # Log epoch summary statistics
        if accelerator.is_main_process:
            epoch_summary = stats_accumulator.get_epoch_summary(max_grad_norm=args.max_grad_norm)
            if epoch_summary and accelerator.is_main_process:
                accelerator.log(epoch_summary, step=global_step)
                logger.info(f"Epoch {epoch} summary: {epoch_summary}")

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        text_encoder_lora_layers = None

        FluxKontextPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )
        
        # Final inference
        # Load previous pipeline (with memory-efficient loading)
        final_load_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (
            torch.float16 if args.mixed_precision == "fp16" else torch.float32
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="transformer", 
            revision=args.revision, 
            variant=args.variant,
            low_cpu_mem_usage=True,
            device_map=None,
            torch_dtype=final_load_dtype
        )
        pipeline = FluxKontextPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True,
        )
        # load attention processors
        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")

        # run inference (all ranks run, aggregate results on rank 0)
        if args.num_validation_images > 0:
            logger.info(f"Running final validation... \n Generating {args.num_validation_images} images.")
            final_eval_step = global_step
            for dataloader, is_final_validation in [(valid_dataloader, True)]:
                final_eval_step += 1
                _run_validation_and_log(
                    pipeline=pipeline,
                    dataloader=dataloader,
                    dataset_name=args.dataset_name,
                    num_images=args.num_validation_images,
                    epoch=epoch,
                    global_step=final_eval_step,
                    is_final_validation=is_final_validation,
                )
            del pipeline
            free_memory()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)