from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import itertools
from pathlib import Path
import json
import math
import copy
from typing import List, Optional, Tuple, Union
from torch import Tensor, stack
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from diffusers.training_utils import find_nearest_bucket
from torch.utils.data.sampler import BatchSampler


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

DEFAULT_BUCKETS = [
        (1024, 1024),    # Square small
        (1536, 1536),    # Square medium
        (2048, 2048),    # Square large
        (1024, 1536),    # Portrait small
        (1536, 2048),    # Portrait medium
        # (2048, 3072),    # Portrait large
        (1536, 1024),    # Landscape small
        (2048, 1536),    # Landscape medium
        # (3072, 2048),    # Landscape large
    ]

ADAPTED_PREFERRED_KONTEXT_RESOLUTIONS = [
    (832, 1248),  #  
    (880, 1184),  #
    (1024, 1024),    #
    (1184, 880),  #
    (1248, 832),  #
]

mapping_bucket_idx_default_buckets_to_preferred_kontext_resolutions = {
    0: 2,
    1: 2,
    2: 2, 
    3: 0,
    4: 1,
    5: 4,
    6: 3,
}

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


def get_filename_with_prefix_sft(base_name, num_instances_min, only_english_source):
    # CAREFUL THAT THIS WILL BE DIFFERENT FOR SFT AND RL.
    """Generate filename with optional min instances prefix and language direction."""
    prefix_parts = []
    
    if only_english_source:
        prefix_parts.append("en_src")
    
    if num_instances_min is not None: 
        prefix_parts.append(f"min{num_instances_min}")
    
    if prefix_parts:
        return "_".join(prefix_parts) + "_" + base_name
    return base_name


"""
Currently the logic is the following (for instance_num bucketing):
We limit the dataset to samples that have at least num_instances_cap valid boxes.
Then, for each sampling iteration, we chose num_instances boxes from the valid boxes so that it is batchable.
By choosing num_instances_boxes < num_instances_cap, we ensure that the chosen sample will have at least num_instances_cap valid boxes.
"""
class CrelloDatasetFromJson(Dataset):
    def __init__(
        self, 
        json_directory: Path, 
        split: str, 
        images_directory: Path, 
        en_src_only: bool = False, 
        num_instances_cap: int = None, 
        num_instances_min: int = None,
        buckets: Optional[List[Tuple[int, int]]] = None, 
        enable_bucketing: bool = False, 
        bring_area_to_1024_squared: bool = False,
        return_type: str = 'tensor',
        use_typo_boxes: bool = False,
        load_tgt_image: bool = True,
        load_only_valid_boxes: bool = False,
        use_custom_filtered: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
        ruin_text_areas: bool = False,
        use_filtered_version_test: bool = False,
        ):

        self.use_custom_filtered = use_custom_filtered
        self.split = split
        self.images_directory = images_directory
        self.en_src_only = en_src_only
        self.num_instances_cap = num_instances_cap
        self.buckets = buckets if buckets is not None else DEFAULT_BUCKETS
        self.enable_bucketing = enable_bucketing
        self.bring_area_to_1024_squared = bring_area_to_1024_squared
        self.return_type = return_type
        self.use_typo_boxes = use_typo_boxes
        self.num_instances_min = num_instances_min
        self.load_tgt_image = load_tgt_image
        self.load_only_valid_boxes = load_only_valid_boxes
        self.ruin_text_areas_enabled = ruin_text_areas
        
        assert (self.return_type in ['tensor', 'pil', 'path'])
        assert (self.num_instances_cap is None or self.num_instances_cap in [1, 5, 10, 20])
        # Prevent double-resizing: bucketing and area normalization are mutually exclusive
        assert not (self.enable_bucketing and self.bring_area_to_1024_squared), "enable_bucketing and bring_area_to_1024_squared cannot both be True"

        if json_path is not None:
            json_path = Path(json_path)
        elif self.use_custom_filtered:
            json_path = Path(self.use_custom_filtered)
        elif self.use_filtered_version_validation:
            json_path = json_directory / 'crello_test_filtered.json'
        else:
            json_file_name = get_filename_with_prefix_sft(f'{split}_dataset.json', num_instances_min, en_src_only)
            json_path = json_directory / json_file_name
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.to_tensor_transform = transforms.ToTensor() if return_type == 'tensor' else None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(self.data, list):
            sample = copy.deepcopy(self.data[index])
        else:
            try:
                sample = copy.deepcopy(self.data[str(index)])
            except KeyError:
                raise IndexError(f"Index {index} out of range")

        sample_src_path = self.images_directory / Path(sample["image_source"])

        if self.return_type == 'path':
            image1 = str(sample_src_path)
            # Lazy load for dimensions needed for bucketing
            if self.enable_bucketing:
                with Image.open(sample_src_path) as img:
                    src_w, src_h = img.size
            if self.load_tgt_image:
                sample_tgt_path = self.images_directory / Path(sample["image_target"])
                image2 = str(sample_tgt_path)
        else:
            image1 = Image.open(sample_src_path).convert('RGB')
            src_w, src_h = image1.width, image1.height

            if self.load_tgt_image:
                sample_tgt_path = self.images_directory / Path(sample["image_target"])
                image2 = Image.open(sample_tgt_path).convert('RGB')
                if self.ruin_text_areas_enabled:
                    bboxes_to_ruin = sample.get('bboxes_xyxy_normalized', [])
                    if self.load_only_valid_boxes and sample.get('valid_bboxes') is not None:
                        bboxes_to_ruin = [bbox for bbox, is_valid in zip(bboxes_to_ruin, sample['valid_bboxes']) if is_valid]
                    image2 = self.ruin_text_areas(image2, bboxes_to_ruin)


        bucket_idx = None
        if self.enable_bucketing:
            # Use saved bucket_idx if available, otherwise recalculate
            saved_bucket_idx = sample.get('bucket_idx')
            if saved_bucket_idx is None:
                saved_bucket_idx = find_nearest_bucket(src_h, src_w, DEFAULT_BUCKETS)
            
            # saved_bucket_idx is an index into DEFAULT_BUCKETS (0-6)
            # If self.buckets is ADAPTED_PREFERRED_KONTEXT_RESOLUTIONS, we need to map
            if len(self.buckets) == len(ADAPTED_PREFERRED_KONTEXT_RESOLUTIONS):
                bucket_idx = mapping_bucket_idx_default_buckets_to_preferred_kontext_resolutions[saved_bucket_idx]
            else:
                # Otherwise, use directly (for DEFAULT_BUCKETS)
                bucket_idx = saved_bucket_idx
            
            if self.return_type != 'path':
                target_height, target_width = self.buckets[bucket_idx]
                image1 = transforms.Resize((target_height, target_width), interpolation=transforms.InterpolationMode.BILINEAR)(image1)
                
                if self.load_tgt_image:
                    image2 = transforms.Resize((target_height, target_width), interpolation=transforms.InterpolationMode.BILINEAR)(image2)

        if self.bring_area_to_1024_squared and self.return_type != 'path':
            aspect_ratio = image1.width / image1.height
            width = round((1024 * 1024 * aspect_ratio) ** 0.5)
            height = round((1024 * 1024 / aspect_ratio) ** 0.5)
            width = width // 16 * 16
            height = height // 16 * 16
            image1 = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR)(image1)
            
            if self.load_tgt_image:
                image2 = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR)(image2)
        
        if self.load_only_valid_boxes:
            valid_boxes = sample['valid_bboxes']
            valid_boxes_indices = [i for i in range(len(valid_boxes)) if valid_boxes[i]]
        else:
            valid_boxes_indices = list(range(len(sample['bboxes_xyxy_normalized'])))

        if self.num_instances_cap is not None:
            num_instances = min(self.num_instances_cap, len(valid_boxes_indices)) 
            chosen_boxes_indices = random.sample(valid_boxes_indices, num_instances)
        else:
            chosen_boxes_indices = valid_boxes_indices
            
        # Rebuild image_target by pasting back text from boxes we're not editing
        all_boxes = sample['bboxes_xyxy_normalized']
        left_over_boxes_indices = [i for i in range(len(all_boxes)) if i not in chosen_boxes_indices]
        left_over_boxes = [sample['bboxes_xyxy_normalized'][i] for i in left_over_boxes_indices]

        if self.return_type != 'path':
            for bbox in left_over_boxes:
                x1, y1, x2, y2 = bbox
                x1 = int(x1 * image2.width)
                y1 = int(y1 * image2.height)
                x2 = int(x2 * image2.width)
                y2 = int(y2 * image2.height)
                image_source_cropped = image1.crop((x1, y1, x2, y2))
                image2.paste(image_source_cropped, (x1, y1))

        sample['bboxes_xyxy_normalized'] = [sample['bboxes_xyxy_normalized'][i] for i in chosen_boxes_indices]
        sample['text_source'] = [sample['text_source'][i] for i in chosen_boxes_indices]
        sample['text_target'] = [sample['text_target'][i] for i in chosen_boxes_indices]

        if self.use_typo_boxes:
            sample['bboxes_xyxy_normalized'] = create_larger_latent_bboxes(sample['bboxes_xyxy_normalized'], [len(source) for source in sample['text_source']], [len(target) for target in sample['text_target']])

        if self.return_type == 'tensor':
            image1 = self.to_tensor_transform(image1)
            if self.load_tgt_image:
                image2 = self.to_tensor_transform(image2)

        result = {
            "bboxes_xyxy_normalized": sample['bboxes_xyxy_normalized'],
            "image_source": image1,
            "image_target": image2 if self.load_tgt_image else None,
            "global_prompt": sample['global_prompt'],
            "text_source": sample['text_source'],
            "text_target": sample['text_target'],
            "id": sample['id'],
            "original_sample_index": sample['original_sample_index'],
            "bucket_idx": bucket_idx,
            "src_language": sample['language1'],
            "tgt_language": sample['language2']
        }

        return result

    def ruin_text_areas(self, image, bboxes_xyxy_normalized):
        width, height = image.width, image.height
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for bbox in bboxes_xyxy_normalized:
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            # draw a black rectangle over the text area
            draw.rectangle((x1, y1, x2, y2), fill='black')
        return image


def crello_from_json_collate_fn(batch):
    """Custom collate function that handles PIL images and tensors and adds bucket information"""
    image_source = [item['image_source'] for item in batch]
    image_target = [item['image_target'] for item in batch]

    if isinstance(image_source[0], Tensor):
        image_source = stack(image_source, dim=0)
        image_target = stack(image_target, dim=0)
    
    result = {
        'bboxes_xyxy_normalized': [item['bboxes_xyxy_normalized'] for item in batch],
        'image_source': image_source,
        'image_target': image_target,
        'global_prompt': [item['global_prompt'] for item in batch],
        'text_source': [item['text_source'] for item in batch],
        'text_target': [item['text_target'] for item in batch],
        'id': [item['id'] for item in batch],
        'bucket_idx': [item['bucket_idx'] for item in batch],
        'original_sample_index': [item['original_sample_index'] for item in batch],
    }

    return result


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,   
        dataset: 'CrelloDatasetFromJson', 
        batch_size: int, 
        json_directory: Union[str, Path], 
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle: bool = True
    ):
        """
        Bucket-based batch sampler with distributed training support.
        
        Args:
            dataset: The CrelloDatasetFromJson to sample from
            batch_size: Number of samples per batch
            json_directory: Directory containing the bucket indices file
            drop_last: Whether to drop incomplete batches
            rank: Process rank for distributed training (default: 0)
            world_size: Total number of processes for distributed training (default: 1)
            seed: Random seed for reproducibility (default: 0)
            shuffle: Whether to shuffle batches and indices within buckets (default: True)
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, but got batch_size={}".format(batch_size))
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank should be a non-negative integer, but got rank={}".format(rank))
        if not isinstance(world_size, int) or world_size <= 0:
            raise ValueError("world_size should be a positive integer, but got world_size={}".format(world_size))
        if rank >= world_size:
            raise ValueError("rank should be less than world_size, but got rank={} and world_size={}".format(rank, world_size))

        json_directory = Path(json_directory)

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.skip_batches = 0

        # Load bucket indices from pre-generated file
        bucket_indices_path = json_directory / get_filename_with_prefix_sft(f'{dataset.split}_bucket_ids.json', dataset.num_instances_min, dataset.en_src_only)
        with open(bucket_indices_path, 'r') as f:
            self.bucket_indices = json.load(f)

        # Calculate total number of batches for this rank
        self._calculate_num_batches()

    def _calculate_num_batches(self):
        """Calculate the number of batches this rank will process."""
        # Count total batches across all buckets
        total_batches = 0
        for indices_in_bucket in self.bucket_indices.values():
            num_samples = len(indices_in_bucket)
            if self.drop_last:
                batches_in_bucket = num_samples // self.batch_size
            else:
                batches_in_bucket = (num_samples + self.batch_size - 1) // self.batch_size
            total_batches += batches_in_bucket
        
        # Distribute batches across ranks
        batches_per_rank = total_batches // self.world_size
        remainder = total_batches % self.world_size
        
        # Ranks with index < remainder get one extra batch
        if self.rank < remainder:
            self.sampler_len = batches_per_rank + 1
        else:
            self.sampler_len = batches_per_rank

    def __iter__(self):
        # Create a deterministic random generator for this epoch
        g = random.Random(self.seed + self.epoch)
        
        all_batches = []
        
        # Generate batches for each bucket with per-epoch shuffling
        for bucket_id, indices_in_bucket in self.bucket_indices.items():
            # Make a copy to avoid modifying the original
            indices = list(indices_in_bucket)
            
            # Shuffle indices within the bucket if shuffle is enabled
            if self.shuffle:
                g.shuffle(indices)
            
            # Create batches from shuffled indices
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # Skip partial batch if drop_last is True
                all_batches.append(batch)
        
        # Shuffle the order of batches across buckets if shuffle is enabled
        if self.shuffle:
            g.shuffle(all_batches)
        
        # Partition batches for this rank (interleaved assignment for load balancing)
        rank_batches = [batch for i, batch in enumerate(all_batches) if i % self.world_size == self.rank]
        
        # Yield batches for this rank
        for i, batch in enumerate(rank_batches):
            if i < self.skip_batches:
                continue
            yield batch
            
        # Reset skip_batches after one iteration so subsequent epochs don't skip
        self.skip_batches = 0

    def __len__(self):
        return self.sampler_len
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler. This ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def set_skip_batches(self, skip_batches: int):
        """
        Set number of batches to skip at the beginning of the next iteration.
        """
        self.skip_batches = skip_batches


