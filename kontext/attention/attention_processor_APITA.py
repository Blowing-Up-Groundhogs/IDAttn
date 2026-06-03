import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from enum import Enum

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
if TYPE_CHECKING:
    from kontext.transformer_flux import FluxAttention



def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class MaskType(Enum):
    HARD = 'hard'
    HARD2 = 'hard2'
    SOFT = 'soft'


def fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, use_bridge, use_global_prompt, instance_position_mask_list, image_token_H, image_token_W, use_context_bridge_tokens=True, context_image_w_instance_token_index_list=None):
    for i in range(instance_num + 1):

        if not use_global_prompt and i == 0:
            continue

        if use_global_prompt:
            instance_text_idxs = instance_text_index_lst[i]
        else:
            instance_text_idxs = instance_text_index_lst[i-1]
        
        # Activate text-to-text attention (always needed)
        atten_mask[instance_text_idxs[:, None], instance_text_idxs] = 1
        
        # Activate text-to-image attention
        # For global prompt (i==0) or when using bridge, activate attention to the corresponding image tokens
        # When not using bridge and not using global prompt, still need to activate local prompts to global image
        if (use_bridge or i == 0) and (i != 0 or use_global_prompt):
            image_token_index = torch.tensor(image_w_instance_token_index_list[i])
            context_image_token_index = torch.tensor(context_image_w_instance_token_index_list[i])
            
            # Activate attentions prompt to image if i==0 else local-prompt to bridge-image/global-image
            atten_mask[instance_text_idxs[:, None], seq_len + image_token_index] = 1

            if use_context_bridge_tokens or i == 0:
                # Activate attentions prompt to context if i==0 else local-prompt to context-bridge-image/context-global-image
                atten_mask[instance_text_idxs[:, None], seq_len + HW + context_image_token_index] = 1
        
        if i > 0:
            if use_bridge:
                # Activate attentions bridge-image to local-prompt
                atten_mask[(seq_len + image_token_index)[:, None], instance_text_idxs] = 1
                if use_context_bridge_tokens:
                    # Activate attentions context-bridge-image to local-prompt
                    atten_mask[(seq_len + HW + context_image_token_index)[:, None], instance_text_idxs] = 1

                # Activate attentions bridge-image to bridge-image 
                atten_mask[(seq_len + image_token_index)[:, None], seq_len + image_token_index] = 1
                if use_context_bridge_tokens:
                    # Activate attentions context-bridge-image to bridge-image 
                    atten_mask[(seq_len + HW + context_image_token_index)[:, None], seq_len + image_token_index] = 1

                    # Activate attentions bridge-image to context-bridge-image 
                    atten_mask[(seq_len + image_token_index)[:, None], seq_len + HW + context_image_token_index] = 1
                    
                    # Activate attentions context-bridge-image to context-bridge-image 
                    atten_mask[(seq_len + HW + context_image_token_index)[:, None], seq_len + HW + context_image_token_index] = 1
            else:
                instance_img_in_patch_idxs = instance_position_mask_list[i-1].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(atten_mask.device)
                # Activate attentions local-prompt to instance-image
                atten_mask[instance_text_idxs[:, None], seq_len + instance_img_in_patch_idxs] = 1
                # Activate attentions local-prompt to context-instance-image
                atten_mask[instance_text_idxs[:, None], seq_len + HW + instance_img_in_patch_idxs] = 1

    return atten_mask


def fill_image_bind_mask(atten_mask, mask_type,instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, use_global_prompt, context_image_w_instance_token_index_list=None):
    global_image_token_index = torch.tensor(image_w_instance_token_index_list[0])
    global_context_image_token_index = torch.tensor(context_image_w_instance_token_index_list[0])

    # Activate global image to global-prompt  (will be deactivated later for instances)
    atten_mask[(seq_len + global_image_token_index)[:, None], : global_seq_len] = 1

    # Activate global image to global-image  (will be deactivated later for instances)
    atten_mask[(seq_len + global_image_token_index)[:, None], seq_len + global_image_token_index] = 1 

    # Activate global image to context-global-image (will be deactivated later for instances)
    atten_mask[(seq_len + global_image_token_index)[:, None], seq_len + HW + global_context_image_token_index] = 1

    # activate global context image to global-prompt (will be deactivated later for instances)
    atten_mask[(seq_len + HW + global_context_image_token_index)[:, None], : global_seq_len] = 1

    # Activate global context image to global-image  (will be deactivated later for instances)
    atten_mask[(seq_len + HW + global_context_image_token_index)[:, None], seq_len + global_image_token_index] = 1

    # Activate global context image to context-global-image (will be deactivated later for instances)
    atten_mask[(seq_len + HW + global_context_image_token_index)[:, None], seq_len + HW + global_context_image_token_index] = 1

    for i in range(1, instance_num+1):
        if use_global_prompt:
            instance_text_idxs = instance_text_index_lst[i]
        else:
            instance_text_idxs = instance_text_index_lst[i-1]
        instance_img_in_patch_idxs = instance_position_mask_list[i-1].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(query.device)
        
        # Activate attention instance-image to local-prompt
        atten_mask[seq_len + instance_img_in_patch_idxs[:, None], instance_text_idxs] = 1
        # Activate attention context-instance-image to local-prompt
        atten_mask[seq_len + HW + instance_img_in_patch_idxs[:, None], instance_text_idxs] = 1

        if mask_type in [MaskType.HARD2, MaskType.HARD]:
            # Deactivate attention instance-image to global-prompt
            atten_mask[seq_len + instance_img_in_patch_idxs[:, None], : global_seq_len] = 0
            # Deactivate attention context-instance-image to global-prompt
            atten_mask[seq_len + HW + instance_img_in_patch_idxs[:, None], : global_seq_len] = 0

        if mask_type == MaskType.HARD:
            # Deactivate attention instance-image to all image tokens (so that we can then activate just instance-image to instance-image later)
            atten_mask[seq_len + instance_img_in_patch_idxs[:, None], seq_len:] = 0
            # Deactivate attention context-instance-image to all image tokens
            atten_mask[seq_len + HW + instance_img_in_patch_idxs[:, None], seq_len:] = 0
        
            # Activate attention instance-image to instance-image
            atten_mask[seq_len + instance_img_in_patch_idxs[:, None], seq_len + instance_img_in_patch_idxs] = 1
            # Activate attention instance-image to context-instance-image
            atten_mask[seq_len + instance_img_in_patch_idxs[:, None], seq_len + HW + instance_img_in_patch_idxs] = 1

            # Activate attention context-instance-image to instance-image
            atten_mask[seq_len + HW + instance_img_in_patch_idxs[:, None], seq_len + instance_img_in_patch_idxs] = 1
            # Activate attention context-instance-image to context-instance-image
            atten_mask[seq_len + HW + instance_img_in_patch_idxs[:, None], seq_len + HW + instance_img_in_patch_idxs] = 1

    atten_mask = atten_mask.bool()
    return atten_mask



class FluxAPITAAttnProcessor:
    _attention_backend = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    cond_hard_bind_mask2 = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    uncond_hard_bind_mask2 = None
    cfg_inference_steps_multiplier = 1

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")
    
    @classmethod
    def clear_cached_masks(cls):
        """Clear all cached attention masks. Should be called after each training sample."""
        # print('clear attn mask from class method', flush=True)
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.cond_hard_bind_mask2 = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.uncond_hard_bind_mask2 = None
        cls.counter = 0
        # print('cleared all masks', flush=True)

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_instance_text_index_lst: Optional[List[List[int]]] = None,
        neg_instance_text_index_lst: Optional[List[List[int]]] = None,
        instance_position_mask_list: Optional[List[List[int]]] = None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_bboxes_xyxy_normalized: Optional[List[List[int]]] = None,
        hard_image_attribute_binding_list: Optional[List[List[int]]] = None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list: Optional[List[List[int]]] = None,
        image_w_instance_token_H_list: Optional[List[int]] = None,
        image_w_instance_token_W_list: Optional[List[int]] = None,
        context_image_w_instance_token_index_list: Optional[List[List[int]]] = None,
        is_conditional: Optional[bool] = None,
        use_bridge: bool = False,
        use_global_prompt: bool = True,
        use_context_bridge_tokens: bool = True,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst

        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        if use_global_prompt:
            global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        else:
            global_seq_len = 0
        instance_num = len(instance_bboxes_xyxy_normalized)

        if not is_conditional:
            FluxAPITAAttnProcessor.cfg_inference_steps_multiplier = 2

        if instance_num == 0:
            instance_num = -1

        if (FluxAPITAAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (FluxAPITAAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            # Build the hard text mask
            atten_mask = torch.zeros(query.shape[1], query.shape[1], device=query.device, dtype=torch.bool)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, use_bridge, use_global_prompt, instance_position_mask_list, image_token_H, image_token_W, use_context_bridge_tokens=use_context_bridge_tokens, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, use_global_prompt=use_global_prompt, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            if is_conditional:
                 FluxAPITAAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                 FluxAPITAAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (FluxAPITAAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (FluxAPITAAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.zeros(query.shape[1], query.shape[1], device=query.device, dtype=torch.bool)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, use_bridge, use_global_prompt, instance_position_mask_list, image_token_H, image_token_W, use_context_bridge_tokens=use_context_bridge_tokens, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, use_global_prompt=use_global_prompt, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            if is_conditional:
                FluxAPITAAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                FluxAPITAAttnProcessor.uncond_soft_bind_mask = atten_mask

        
        if FluxAPITAAttnProcessor.counter % 57 in hard_image_attribute_binding_list:
            atten_mask = FluxAPITAAttnProcessor.cond_hard_bind_mask if is_conditional else FluxAPITAAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = FluxAPITAAttnProcessor.cond_soft_bind_mask if is_conditional else FluxAPITAAttnProcessor.uncond_soft_bind_mask 
        FluxAPITAAttnProcessor.counter += 1

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=atten_mask,
            backend=self._attention_backend,
        )

        if FluxAPITAAttnProcessor.counter % (num_inference_steps * 57 * FluxAPITAAttnProcessor.cfg_inference_steps_multiplier) == 0:
            FluxAPITAAttnProcessor.clear_cached_masks()
        
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states