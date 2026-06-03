import math
import torch


def find_inner_sentence_token_span_t5(tokenizer, full_prompt: str, inner_sentence: str, max_sequence_length: int = 512):
    """
    Returns (start_idx, end_idx) token span for `inner_sentence` inside `full_prompt` for T5.
    Indices are over the token dimension used by `_encode_prompt_with_t5` (before duplication),
    end_idx is exclusive. Returns None if no contiguous match is found.
    """
    # Tokenize full prompt as done in _encode_prompt_with_t5 (padded to max length, truncated)
    full_tok = tokenizer(
        full_prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    full_ids = full_tok.input_ids[0]
    full_mask = full_tok.attention_mask[0]
    valid_len = int(full_mask.sum().item())
    full_ids = full_ids[:valid_len].tolist()

    # Tokenize inner sentence without padding, keep special tokens then strip them explicitly
    inner_tok = tokenizer(
        inner_sentence,
        padding=False,
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    inner_ids = inner_tok.input_ids[0].tolist()

    # Remove pad/eos from the query to search only content tokens
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    inner_ids = [tid for tid in inner_ids if tid != pad_id and tid != eos_id]

    if not inner_ids:
        return None, None

    n = len(inner_ids)
    for i in range(0, len(full_ids) - n + 1):
        if full_ids[i : i + n] == inner_ids:
            return i, i + n

    return None, None


def create_position_mask_list(instance_bboxes_xyxy_normalized, height, width, vae_scale_factor):
    position_mask_list = []
    for i in range(len(instance_bboxes_xyxy_normalized)):
        instance_box_i = instance_bboxes_xyxy_normalized[i]
        position_mask_list.append(change_box_to_position_mask(instance_box_i, height // vae_scale_factor // 2, width // vae_scale_factor // 2))
    return position_mask_list


def create_larger_latent_bboxes(instance_bboxes_xyxy_normalized, characters_source, characters_target):
    larger_latent_bboxes = []
    for i in range(len(instance_bboxes_xyxy_normalized)):
        instance_box_i = instance_bboxes_xyxy_normalized[i]
        characters_source_i = characters_source[i]
        characters_target_i = characters_target[i]


        if characters_source_i > characters_target_i:
            larger_latent_bboxes.append(instance_box_i)
            continue

        x1, y1, x2, y2 = instance_box_i

        width_source = x2 - x1
        height_source = y2 - y1

        character_width_source = width_source / characters_source_i

        width_target_first = characters_target_i * character_width_source

        if width_target_first <= width_source * 1.25:
            width_target = width_target_first
            height_target = height_source
        else:
            n_lines = math.ceil(characters_target_i / characters_source_i)
            height_target = height_source * n_lines
            width_target = width_source

        # x1_target = x1
        # y1_target = y1
        # x2_target = x1_target + width_target
        # y2_target = y1_target + height_target
        w_p = width_target - width_source
        h_p = height_target - height_source
        x1_target = x1 - w_p/2
        y1_target = y1 - h_p/2
        x2_target = x2 + w_p/2
        y2_target = y2 + h_p/2

        # recover from overflow

        if y2_target > 1:
            y_shift = y2_target -1
            y2_target -= y_shift
            y1_target -= y_shift
        
        if y2_target < 0:
            y_shift = y1_target
            y1_target -= y_shift
            y2_target -= y_shift
        
        if x2_target > 1:
            x_shift = x2_target -1
            x2_target -= x_shift
            x1_target -= x_shift
        
        if x2_target < 0:
            x_shift = x1_target
            x1_target -= x_shift
            x2_target -= x_shift



        larger_latent_bboxes.append((x1_target, y1_target, x2_target, y2_target))
    return larger_latent_bboxes


def change_box_to_position_mask(box, H, W):
    x1, y1, x2, y2 = box
    x1 = math.floor(x1 * W)
    y1 = math.floor(y1 * H)
    x2 = math.ceil(x2 * W)
    y2 = math.ceil(y2 * H)
    position_mask = torch.zeros(H, W)
    position_mask[y1: y2, x1: x2] = 1
    return position_mask


def create_instance_image_list(instance_bboxes_xyxy_normalized, prompt_image):
    instance_image_list = [prompt_image]
    for i in range(len(instance_bboxes_xyxy_normalized)):
        x1, y1, x2, y2 = instance_bboxes_xyxy_normalized[i]
        x1 = math.floor(x1 * prompt_image.width)
        y1 = math.floor(y1 * prompt_image.height)
        x2 = math.ceil(x2 * prompt_image.width)
        y2 = math.ceil(y2 * prompt_image.height)
        instance_image = prompt_image.crop((x1, y1, x2, y2)) 
        instance_image_list.append(instance_image)
    return instance_image_list