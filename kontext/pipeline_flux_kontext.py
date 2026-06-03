# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import math

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from torchvision.transforms.functional import pil_to_tensor

from kontext.pipeline_utils import create_larger_latent_bboxes, create_position_mask_list, find_inner_sentence_token_span_t5
from kontext.transformer_flux import FluxTransformer2DModel, FluxAttnProcessor

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxKontextPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = FluxKontextPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
        ... ).convert("RGB")
        >>> prompt = "Make Pikachu hold a sign that says 'Black Forest Labs is awesome', yarn art style, detailed, vibrant colors"
        >>> image = pipe(
        ...     image=image,
        ...     prompt=prompt,
        ...     guidance_scale=2.5,
        ...     generator=torch.Generator().manual_seed(42),
        ... ).images[0]
        >>> image.save("output.png")
        ```
"""

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
# FABIO moved out from the pipeline so we can call it from training code
def _get_t5_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt_settings: str,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    num_instance_text_tokens = 200,
    num_global_text_tokens = 200,
    use_global_prompt = True,
):

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    prompt = [prompt_.split('$BREAKFLAG$') for prompt_ in prompt]  
    if prompt_settings in ['inner_local_prompts', 'base']:
        prompt_to_encode = [prompt_[0] for prompt_ in prompt]
    elif prompt_settings in ['outer_local_prompts', 'outer_local_prompts_smart']:
        prompt_to_encode = prompt[0]
        assert batch_size == 1, "Batch size must be 1 for outer local prompts"
    else:
        raise ValueError(f"Invalid prompt settings: {prompt_settings}")

    if not use_global_prompt:
        raise ValueError("use_global_prompt must be True for now")
        prompt_to_encode = [prompt_[1:] for prompt_ in prompt]

    text_inputs = tokenizer(
            prompt_to_encode,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt_to_encode, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]

    if hasattr(text_encoder, "module"): 
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    batched_instance_text_index_lst = []
    if prompt_settings in ['inner_local_prompts', 'base']:
        batched_inner_sentences = [prompt_[1:] for prompt_ in prompt]
        for batch_idx, inner_sentences in enumerate(batched_inner_sentences):
            instance_text_index_lst = []
            for inner_sentence in inner_sentences:
                inner_mask = torch.zeros_like(text_inputs.attention_mask[batch_idx])
                start, end = find_inner_sentence_token_span_t5(tokenizer, prompt_to_encode[batch_idx], inner_sentence, max_sequence_length=max_sequence_length)
                if start is None:
                    logger.warning(f"Inner sentence not found in tokenized prompt: {inner_sentence}")
                else:
                    inner_mask[start:end] = 1
                instance_text_index_lst.append(inner_mask)
        batched_instance_text_index_lst.append(instance_text_index_lst)
    elif prompt_settings in ['outer_local_prompts', 'outer_local_prompts_smart']:
        begin_idx = 0
        prompt_embeds_list = []
        instance_text_index_lst = []
        for i, _ in enumerate(prompt_to_encode):
            if i == 0:
                _instance_token_num = num_global_text_tokens
                prompt_embeds_list.append(prompt_embeds[[i], :_instance_token_num, :].contiguous())
                instance_text_index = torch.arange(begin_idx, begin_idx+_instance_token_num).int()
                instance_text_index_lst.append(instance_text_index)
                begin_idx += _instance_token_num
            else:
                if prompt_settings == 'outer_local_prompts_smart':
                    num_instance_text_tokens = torch.sum(text_input_ids[i] != 0) 
                prompt_embeds_list.append(prompt_embeds[[i], :num_instance_text_tokens, :].contiguous())
                instance_text_index = torch.arange(begin_idx, begin_idx+num_instance_text_tokens).int()
                instance_text_index_lst.append(instance_text_index)
                begin_idx += num_instance_text_tokens

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        batched_instance_text_index_lst.append(instance_text_index_lst)
    else:
        raise ValueError(f"Invalid prompt settings: {prompt_settings}")

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, batched_instance_text_index_lst, seq_len


class FluxKontextPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    r"""
    The Flux Kontext pipeline for image-to-image and text-to-image generation.

    Reference: https://bfl.ai/announcements/flux-1-kontext-dev

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        if hasattr(self.transformer, "set_attn_processor"):
            self.transformer.set_attn_processor(FluxAttnProcessor())
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128


    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt_settings: str,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
        use_global_prompt: bool = True,
        num_instance_text_tokens = 200,
        num_global_text_tokens = 200,
        text_ids: Optional[torch.LongTensor] = None,
        instance_text_index_lst: Optional[List[List[torch.LongTensor]]] = None,
        seq_len: Optional[int] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            use_global_prompt (`bool`, *optional*):
                Whether to use the global prompt.
            num_instance_text_tokens (`int`, *optional*):
                The number of instance text tokens.
            num_global_text_tokens (`int`, *optional*):
                The number of global text tokens.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

            
            prompt = [prompt] if isinstance(prompt, str) else prompt
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

            prompt_embeds, instance_text_index_lst, seq_len = _get_t5_prompt_embeds(
                text_encoder=self.text_encoder_2,
                tokenizer=self.tokenizer_2,
                prompt_settings=prompt_settings,
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                num_instance_text_tokens=num_instance_text_tokens,
                num_global_text_tokens=num_global_text_tokens,
                use_global_prompt=use_global_prompt,
            )
            

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids, instance_text_index_lst, seq_len

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds


    def create_instance_latents_and_context_image(self, latent_bboxes, instance_bboxes_xyxy_normalized, batch_size, latents, ori_image_latents, height, width, vae_scale_factor, use_context_bridge_tokens, pad_to_square, generator, ori_image):
        instance_context_image_list = []
        instance_latents_list = []
        instance_latent_image_ids_list = []
        instance_context_image_id_list = []
        image_token_H_list = [height]
        image_token_W_list = [width]
        context_image_token_H_list = [height]
        context_image_token_W_list = [width]

        for latent_box_i, instance_box_i in zip(latent_bboxes, instance_bboxes_xyxy_normalized):

            def change_box_to_attn_mask(box, H, W):
                x1, y1, x2, y2 = box
                x1 = math.floor(x1 * W)
                y1 = math.floor(y1 * H)
                x2 = math.ceil(x2 * W)
                y2 = math.ceil(y2 * H)
                _atten_mask = torch.zeros(H, W)
                _atten_mask[y1: y2, x1: x2] = 1
                _atten_mask = _atten_mask.reshape(H * W)
                indices_of_ones = _atten_mask.nonzero(as_tuple=True)[0]
                return indices_of_ones, y2 - y1, x2 - x1

            def pad_to_square_with_white(img: Image.Image) -> Image.Image:
                width, height = img.size
                multiple_of = self.vae_scale_factor * 2
                target = math.ceil((max(height, width) + 5) / multiple_of) * multiple_of

                padded = Image.new(img.mode if img.mode != "P" else "RGB", (target, target), color="white")
                offset_x = (target - width) // 2
                offset_y = (target - height) // 2
                padded.paste(img, (offset_x, offset_y))
                return padded, offset_x, offset_y

            def pad_to_square_with_same(img: Image.Image) -> Image.Image:
                width, height = img.size
                multiple_of = self.vae_scale_factor * 2
                target = math.ceil((max(height, width) + 5) / multiple_of) * multiple_of

                base_img = img.convert("RGB") if img.mode == "P" else img
                offset_x = (target - width) // 2
                offset_y = (target - height) // 2

                if offset_x == 0 and offset_y == 0:
                    return base_img, 0, 0

                arr = np.array(base_img)
                if arr.ndim == 2:
                    padded_arr = np.pad(
                        arr,
                        ((offset_y, target - height - offset_y), (offset_x, target - width - offset_x)),
                        mode="edge",
                    )
                else:
                    padded_arr = np.pad(
                        arr,
                        (
                            (offset_y, target - height - offset_y),
                            (offset_x, target - width - offset_x),
                            (0, 0),
                        ),
                        mode="edge",
                    )
                padded = Image.fromarray(padded_arr)
                return padded, offset_x, offset_y

            instance_idx, instance_h, instance_w = change_box_to_attn_mask(instance_box_i, height // vae_scale_factor // 2, width // vae_scale_factor // 2)
            if latent_bboxes != instance_bboxes_xyxy_normalized:
                latent_instance_idx, latent_instance_h, latent_instance_w = change_box_to_attn_mask(latent_box_i, height // vae_scale_factor // 2, width // vae_scale_factor // 2)
                instance_latents_latents = latents[:, latent_instance_idx, :].clone()
                instance_image_latents = ori_image_latents[:, instance_idx, :].clone()
                instance_latent_image_ids = self._prepare_latent_image_ids(batch_size, latent_instance_h, latent_instance_w, latents.device, latents.dtype)
                instance_context_image_id = self._prepare_latent_image_ids(batch_size, instance_h, instance_w, latents.device, latents.dtype)


                instance_token_h = latent_instance_h * vae_scale_factor * 2
                instance_token_w = latent_instance_w * vae_scale_factor * 2

                context_token_h = instance_token_h
                context_token_w = instance_token_w
            if pad_to_square:
                now_latents = latents[:, instance_idx, :]
                multiple_of = self.vae_scale_factor * 2
                now_instance_token_h = instance_h * multiple_of
                now_instance_token_w = instance_w * multiple_of

                now_latents_unpacked = self._unpack_latents(now_latents, now_instance_token_h, now_instance_token_w, self.vae_scale_factor)
                now_latents_decoded = self.vae.decode(now_latents_unpacked, return_dict=False)[0]
                now_latents_decoded = self.image_processor.postprocess(now_latents_decoded, output_type='pil')[0]
                now_latents_decoded_padded, offset_x, offset_y = pad_to_square_with_white(now_latents_decoded)
                now_latents_decoded_padded = self.image_processor.preprocess(now_latents_decoded_padded, now_latents_decoded_padded.height, now_latents_decoded_padded.width)
                now_latents_decoded_padded = now_latents_decoded_padded.to(latents.device, latents.dtype)
                now_latents_padded = self._encode_vae_image(now_latents_decoded_padded, generator)
                instance_latents_latents = self._pack_latents(now_latents_padded, batch_size, now_latents_padded.shape[1], now_latents_padded.shape[2], now_latents_padded.shape[3])


                context_image_decoded = ori_image.crop((instance_box_i[0] * width, instance_box_i[1] * height, instance_box_i[2] * width, instance_box_i[3] * height))
                context_image_padded, offset_x, offset_y = pad_to_square_with_white(context_image_decoded)
                context_image_padded = self.image_processor.preprocess(context_image_padded, context_image_padded.height, context_image_padded.width)
                context_image_padded = context_image_padded.to(latents.device, latents.dtype)
                context_image_padded = self._encode_vae_image(context_image_padded, generator)
                instance_image_latents = self._pack_latents(context_image_padded, batch_size, context_image_padded.shape[1], context_image_padded.shape[2], context_image_padded.shape[3])

                # padded_now_latents_unpacked_target = math.ceil((max(now_latents_unpacked.shape[2], now_latents_unpacked.shape[3]) + 2) / multiple_of) * multiple_of
                # padded_now_latents_unpacked_offset_x = (padded_now_latents_unpacked_target - now_latents_unpacked.shape[2]) // 2
                # padded_now_latents_unpacked_offset_y = (padded_now_latents_unpacked_target - now_latents_unpacked.shape[3]) // 2

                # white = Image.new("RGB", (padded_now_latents_unpacked_target*8, padded_now_latents_unpacked_target*8), color="white")
                # white = self.image_processor.preprocess(white, white.height, white.width).to(latents.device, latents.dtype)
                # padded_now_latents_unpacked = self._encode_vae_image(white, generator)
                # # padded_now_latents_unpacked = torch.zeros((batch_size, now_latents_unpacked.shape[1], padded_now_latents_unpacked_target, padded_now_latents_unpacked_target)).to(latents.device, latents.dtype)
                # padded_now_latents_unpacked[:, :,padded_now_latents_unpacked_offset_x:padded_now_latents_unpacked_offset_x + now_latents_unpacked.shape[2], padded_now_latents_unpacked_offset_y:padded_now_latents_unpacked_offset_y + now_latents_unpacked.shape[3]] = now_latents_unpacked
                # padded_h = padded_now_latents_unpacked.shape[2]
                # padded_w = padded_now_latents_unpacked.shape[3]
                # instance_latents_latents = self._pack_latents(padded_now_latents_unpacked, batch_size, padded_now_latents_unpacked.shape[1], padded_now_latents_unpacked.shape[2], padded_now_latents_unpacked.shape[3])

                # now_latents = self._unpack_latents(now_latents, instance_h * vae_scale_factor * 2, instance_w * vae_scale_factor * 2, self.vae_scale_factor)
                # latent_decoded = self.vae.decode(now_latents, return_dict=False)[0]
                # latent_decoded = self.image_processor.postprocess(latent_decoded, output_type='pil')[0]
                # latent_decoded_padded, offset_x, offset_y = pad_to_square_with_margin(latent_decoded)
                # latent_decoded_padded = self.image_processor.preprocess(latent_decoded_padded, latent_decoded_padded.height, latent_decoded_padded.width)
                # latent_decoded_padded = latent_decoded_padded.to(latents.device, latents.dtype)

                # image_cut = ori_image_latents[:, instance_idx, :]
                # image_cut_unpacked = self._unpack_latents(image_cut, instance_h * vae_scale_factor * 2, instance_w * vae_scale_factor * 2, self.vae_scale_factor)
                # padded_image_cut_unpacked = self._encode_vae_image(white, generator)
                # padded_image_cut_unpacked = torch.zeros((batch_size, now_latents_unpacked.shape[1], padded_now_latents_unpacked_target, padded_now_latents_unpacked_target)).to(latents.device, latents.dtype)
                # padded_image_cut_unpacked[:, :,padded_now_latents_unpacked_offset_x:padded_now_latents_unpacked_offset_x + now_latents_unpacked.shape[2], padded_now_latents_unpacked_offset_y:padded_now_latents_unpacked_offset_y + now_latents_unpacked.shape[3]] = image_cut_unpacked
                # instance_image_latents = self._pack_latents(padded_image_cut_unpacked, batch_size, padded_image_cut_unpacked.shape[1], padded_image_cut_unpacked.shape[2], padded_image_cut_unpacked.shape[3])

                # image_cut = self._unpack_latents(image_cut, instance_h * vae_scale_factor * 2, instance_w * vae_scale_factor * 2, self.vae_scale_factor)
                # image_cut = self.vae.decode(image_cut, return_dict=False)[0]
                # image_cut = self.image_processor.postprocess(image_cut, output_type='pil')[0]
                # image_cut_padded, _, _ = pad_to_square_with_margin(image_cut)   
                # image_cut_padded = self.image_processor.preprocess(image_cut_padded, image_cut_padded.height, image_cut_padded.width)
                # image_cut_padded = image_cut_padded.to(ori_image_latents.device, ori_image_latents.dtype)

                # instance_latents_latents = self._encode_vae_image(image=latent_decoded_padded, generator=generator)
                # padded_h = instance_latents_latents.shape[2] // 2
                # padded_w = instance_latents_latents.shape[3] // 2
                # instance_latents_latents = self._pack_latents(instance_latents_latents, batch_size, instance_latents_latents.shape[1], instance_latents_latents.shape[2], instance_latents_latents.shape[3])
                
                # instance_image_latents = self._encode_vae_image(image=image_cut_padded, generator=generator)
                # instance_image_latents = self._pack_latents(instance_image_latents, batch_size, instance_image_latents.shape[1], instance_image_latents.shape[2], instance_image_latents.shape[3])
                # offset_x = offset_x // self.vae_scale_factor // 2
                # offset_y = offset_y // self.vae_scale_factor // 2

                offset_x = offset_x // 2
                offset_y = offset_y // 2

                padded_h_packed = now_latents_padded.shape[2] // 2
                padded_w_packed = now_latents_padded.shape[3] // 2
                
                instance_latent_image_ids = self._prepare_padded_latent_image_ids(batch_size, now_latents_padded.shape[2] // 2, now_latents_padded.shape[3] // 2, offset_y, offset_x, latents.device, latents.dtype)
                instance_context_image_id = self._prepare_padded_latent_image_ids(batch_size, context_image_padded.shape[2] // 2, context_image_padded.shape[3] // 2, offset_y, offset_x, latents.device, latents.dtype)
                instance_token_h = padded_h_packed * vae_scale_factor * 2
                instance_token_w = padded_w_packed * vae_scale_factor * 2
                context_token_h = context_image_padded.shape[2] * vae_scale_factor
                context_token_w = context_image_padded.shape[3] * vae_scale_factor
            else:
                instance_latents_latents = latents[:, instance_idx, :].clone()
                instance_image_latents = ori_image_latents[:, instance_idx, :].clone()
                instance_latent_image_ids = self._prepare_latent_image_ids(batch_size, instance_h, instance_w, latents.device, latents.dtype)
                instance_context_image_id = self._prepare_latent_image_ids(batch_size, instance_h, instance_w, latents.device, latents.dtype)
                instance_token_h = instance_h * vae_scale_factor * 2
                instance_token_w = instance_w * vae_scale_factor * 2

                context_token_h = instance_token_h
                context_token_w = instance_token_w

            assert instance_idx.shape[0] == instance_h * instance_w

            instance_latents_list.append(instance_latents_latents)
            instance_latent_image_ids_list.append(instance_latent_image_ids)

            if use_context_bridge_tokens:
                instance_context_image_list.append(instance_image_latents)
                instance_context_image_id[..., 0] = 1
                instance_context_image_id_list.append(instance_context_image_id)
                
            image_token_H_list.append(instance_token_h)
            image_token_W_list.append(instance_token_w)
            context_image_token_H_list.append(context_token_h)
            context_image_token_W_list.append(context_token_w)

        return instance_latents_list, instance_context_image_list, instance_latent_image_ids_list, instance_context_image_id_list, image_token_H_list, image_token_W_list, context_image_token_H_list, context_image_token_W_list

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt
    ):
        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(single_ip_adapter_image, device, 1)
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]

            if len(ip_adapter_image_embeds) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as the number of IP Adapters. Got {len(ip_adapter_image_embeds)} image embeds and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype) 

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_padded_latent_image_ids(batch_size, height, width, offset_y, offset_x, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None] - offset_y
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :] - offset_x

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)


    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        depr_message = f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        depr_message = f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            image_ids = self._prepare_latent_image_ids(
                batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
            )
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            image_ids[..., 0] = 1

        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents, latent_ids, image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        text_ids: Optional[torch.LongTensor] = None,
        batched_instance_text_index_lst: Optional[List[List[torch.LongTensor]]] = None, 
        batched_neg_instance_text_index_lst: Optional[List[List[torch.LongTensor]]] = None,
        seq_len: Optional[int] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_text_ids: Optional[torch.LongTensor] = None,
        negative_batched_instance_text_index_lst: Optional[List[List[torch.LongTensor]]] = None,
        negative_seq_len: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        hard_image_attribute_binding_list = [],
        num_hard_control_steps = 0,
        instance_bboxes_xyxy_normalized = [],
        _auto_resize: bool = True,
        use_bridge: bool = True,
        use_global_prompt: bool = True,
        use_pos_embeds_from_orig: bool = False,
        use_context_bridge_tokens:bool = True,
        pad_to_square: bool = False,
        attention_setting: str = 'full',
        prompt_settings: str = 'inner_local_prompts',
        larger_latent_boxes: bool = False,
        characters_source = [],
        characters_target = [],
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guidance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with prompt at the expense of lower image quality.

                Guidance-distilled models approximates true classifier-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512):
                Maximum sequence length to use with the `prompt`.
            max_area (`int`, defaults to `1024 ** 2`):
                The maximum area of the generated image in pixels. The height and width will be adjusted to fit this
                area while maintaining the aspect ratio.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        if larger_latent_boxes and characters_source and characters_target:
            latent_bboxes = create_larger_latent_bboxes(instance_bboxes_xyxy_normalized, characters_source, characters_target)
        else:
            latent_bboxes = instance_bboxes_xyxy_normalized

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        # width = round((max_area * aspect_ratio) ** 0.5)   # FABIO WE HAD TO COMMENT THIS
        # height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        # width = width // multiple_of * multiple_of
        # height = height // multiple_of * multiple_of

        if height != original_height or width != original_width:
            logger.warning(
                f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        instance_num = len(instance_bboxes_xyxy_normalized)
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            instance_text_index_lst,
            seq_len,
        ) = self.encode_prompt(
            prompt_settings=prompt_settings,
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            instance_text_index_lst=batched_instance_text_index_lst,
            seq_len=seq_len,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
                neg_instance_text_index_lst,
                neg_seq_len,
            ) = self.encode_prompt(
                prompt_settings=prompt_settings,
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                text_ids=negative_text_ids,
                instance_text_index_lst=batched_neg_instance_text_index_lst,
                seq_len=negative_seq_len,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        ori_image = image

        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            if _auto_resize:
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)
        
        # no need to do instance image list because we are nopt passing them to the text encoder
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        ori_latents = latents
        ori_image_latents = image_latents

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        assert batch_size == 1, "Batch size must be 1 for now"

        #Create masks from boxes
        instance_text_index_lst = instance_text_index_lst[0]
        txt_seq_lens_list = [instance_text_index_lst[text_idx].shape[0] for text_idx in range(len(instance_text_index_lst))]
        if do_true_cfg:
            neg_instance_text_index_lst = neg_instance_text_index_lst[0]
            negative_txt_seq_lens_list = [neg_instance_text_index_lst[text_idx].shape[0] for text_idx in range(len(neg_instance_text_index_lst))]


        instance_position_mask_list = create_position_mask_list(latent_bboxes, height, width, self.vae_scale_factor)
        context_image_position_mask_list = create_position_mask_list(instance_bboxes_xyxy_normalized, height, width, self.vae_scale_factor)

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                now_steps = i
                if self.interrupt:
                    continue

                self._current_timestep = t
                latents = latents[:, :ori_latents.shape[1], :]
                latent_ids = latent_ids[:ori_latents.shape[1], :]
                image_latents = image_latents[:, :ori_image_latents.shape[1], :]
                image_ids = image_ids[:ori_image_latents.shape[1], :]
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                if prompt_settings in ['inner_local_prompts', 'base']:
                    pass

                else:
                    pass

                image_w_instance_token_index_list = [list(range(ori_latents.shape[1]))]
                context_image_w_instance_token_index_list = [list(range(ori_image_latents.shape[1]))]
                if use_bridge:
                    instance_latents_list, instance_context_image_list, instance_latent_image_ids_list, instance_context_image_ids_list, image_w_instance_token_H_list, image_w_instance_token_W_list, context_image_w_instance_token_H_list, context_image_w_instance_token_W_list = self.create_instance_latents_and_context_image(
                        latent_bboxes, instance_bboxes_xyxy_normalized, batch_size, latents, ori_image_latents, height, width, self.vae_scale_factor, use_context_bridge_tokens, pad_to_square, generator, ori_image)

                    if use_pos_embeds_from_orig:
                        for index_pos_embeds in range(instance_num):
                            if pad_to_square:
                                instance_box_xyxy_normalized = instance_bboxes_xyxy_normalized[index_pos_embeds]
                                x1 = instance_box_xyxy_normalized[0] * image_width // self.vae_scale_factor // 2
                                y1 = instance_box_xyxy_normalized[1] * image_height // self.vae_scale_factor // 2
                                instance_latent_image_ids_from_origin = instance_latent_image_ids_list[index_pos_embeds]
                                instance_latent_image_ids_from_origin[..., 1] += y1
                                instance_latent_image_ids_from_origin[..., 2] += x1
                                instance_latent_image_ids_list[index_pos_embeds] = instance_latent_image_ids_from_origin

                                if use_context_bridge_tokens:
                                    instance_context_image_ids_from_origin = instance_context_image_ids_list[index_pos_embeds]
                                    instance_context_image_ids_from_origin[..., 1] += y1
                                    instance_context_image_ids_from_origin[..., 2] += x1
                                    instance_context_image_ids_list[index_pos_embeds] = instance_context_image_ids_from_origin

                            else:
                                instance_img_in_patch_idxs = instance_position_mask_list[index_pos_embeds].reshape(ori_latents.shape[1]).nonzero(as_tuple=True)[0].to(ori_latents.device)
                                instance_context_image_in_patch_idxs = context_image_position_mask_list[index_pos_embeds].reshape(ori_image_latents.shape[1]).nonzero(as_tuple=True)[0].to(ori_image_latents.device)
                                instance_latent_image_ids_list[index_pos_embeds] = latent_ids[instance_img_in_patch_idxs, :]
                                instance_context_image_ids_list[index_pos_embeds] = image_ids[instance_context_image_in_patch_idxs, :]
                                if use_context_bridge_tokens:
                                    instance_context_image_ids_list[index_pos_embeds] = image_ids[instance_img_in_patch_idxs, :]
                                    instance_context_image_ids_list[index_pos_embeds] = image_ids[instance_context_image_in_patch_idxs, :]
                    
                    if instance_num > 0:
                        instance_latents = torch.cat(instance_latents_list, dim=1)   # (batch_size * num_images_per_prompt, instance_token_num, C)
                        latents = torch.cat([latents, instance_latents], dim=1)

                        instance_latent_image_ids = torch.cat(instance_latent_image_ids_list, dim=0)  # (instance_token_num, C)
                        latent_ids = torch.cat([latent_ids, instance_latent_image_ids], dim=0)

                        if use_context_bridge_tokens:
                            instance_context_image_ids = torch.cat(instance_context_image_ids_list, dim=0)  # (instance_token_num, C)
                            image_ids = torch.cat([image_ids, instance_context_image_ids], dim=0)
                            instance_context_image = torch.cat(instance_context_image_list, dim=1)   # (batch_size * num_images_per_prompt, instance_token_num, C)
                            image_latents = torch.cat([ori_image_latents, instance_context_image], dim=1) # image latents are the latents of the entire context image (i.e. the one we are editing)
                        
                    # after concat, calculate the index for each original instance token in new latents
                    begin_context_index = begin_index = ori_latents.shape[1]
                    
                    for _ in range(instance_num):
                        instance_latents_len = instance_latents_list[_].shape[1]
                        instance_context_image_len = instance_context_image_list[_].shape[1]

                        context_image_w_instance_token_index_list.append(list(range(begin_context_index, begin_context_index + instance_context_image_len)))
                        begin_context_index += instance_context_image_len

                        image_w_instance_token_index_list.append(list(range(begin_index, begin_index + instance_latents_len)))
                        begin_index += instance_latents_len
                else:
                    instance_latents_list = []
                    instance_context_image_list = []
                    image_w_instance_token_H_list = [height]
                    image_w_instance_token_W_list = [width]
                    image_w_instance_token_index_list = [list(range(ori_latents.shape[1]))]
                    context_image_w_instance_token_index_list = [list(range(ori_image_latents.shape[1]))]
                    context_image_w_instance_token_H_list = [height]
                    context_image_w_instance_token_W_list = [width]
                    
                if attention_setting == 'full':
                    pass
                elif attention_setting == 'APITA':
                    self._joint_attention_kwargs['pos_instance_text_index_lst'] = instance_text_index_lst
                    if do_true_cfg:
                        self._joint_attention_kwargs['neg_instance_text_index_lst'] = neg_instance_text_index_lst
                    self._joint_attention_kwargs['instance_position_mask_list'] = instance_position_mask_list
                    self._joint_attention_kwargs['pos_seq_len'] = seq_len
                    if do_true_cfg:
                        self._joint_attention_kwargs['neg_seq_len'] = neg_seq_len
                    self._joint_attention_kwargs['instance_bboxes_xyxy_normalized'] = instance_bboxes_xyxy_normalized
                    self._joint_attention_kwargs['hard_image_attribute_binding_list'] = hard_image_attribute_binding_list
                    self._joint_attention_kwargs['num_inference_steps'] = num_inference_steps
                    self._joint_attention_kwargs['use_bridge'] = use_bridge
                    self._joint_attention_kwargs['use_global_prompt'] = use_global_prompt
                    self._joint_attention_kwargs['use_context_bridge_tokens'] = use_context_bridge_tokens
                    self._joint_attention_kwargs['image_w_instance_token_index_list'] = image_w_instance_token_index_list
                    self._joint_attention_kwargs['image_w_instance_token_H_list'] = image_w_instance_token_H_list
                    self._joint_attention_kwargs['image_w_instance_token_W_list'] = image_w_instance_token_W_list
                    self._joint_attention_kwargs['is_conditional'] = True
                    self._joint_attention_kwargs['context_image_w_instance_token_index_list'] = context_image_w_instance_token_index_list


                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                if image_ids is not None:
                    latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    self._joint_attention_kwargs['is_conditional'] = False
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            # latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            # image = self.vae.decode(latents, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)

            all_results = []

            image_list = []
            for i, image_token_index in enumerate(image_w_instance_token_index_list):
                now_latents = latents[:, image_token_index, :]
                now_latents = self._unpack_latents(now_latents, image_w_instance_token_H_list[i], image_w_instance_token_W_list[i], self.vae_scale_factor)
                now_latents = (now_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(now_latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
                image_list.append(image)
            all_results.extend(image_list)
            context_image_list = []
            for i, image_token_index in enumerate(context_image_w_instance_token_index_list):
                now_latents = image_latents[:, image_token_index, :]
                now_latents = self._unpack_latents(now_latents, context_image_w_instance_token_H_list[i], context_image_w_instance_token_W_list[i], self.vae_scale_factor)
                now_latents = (now_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(now_latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
                context_image_list.append(image)

            all_results.extend(context_image_list)
            

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=all_results)
