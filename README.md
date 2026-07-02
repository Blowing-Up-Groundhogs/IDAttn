# APITA: Attention-guided text editing with FLUX.1-Kontext

This repository contains a minimal implementation of **IDAttn** (Instance-Disentangled Attention) for text editing on top of
[FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

IDAttn replaces the default joint-attention processor of the FLUX Kontext
transformer with a masked processor that binds each text-editing instruction to
its target image region (defined by a bounding box). It can be run **with a
LoRA checkpoint fine-tuned for text editing** or **directly on the base model**
(no checkpoint), and it works on three benchmarks out of the box: **LoMOE**,
**Crello**, and **InfoDet**.

## Requirements

* CUDA 12.4
* Python 3.11

```bash
pip install -r requirements.txt
```

The pinned `diffusers` build (installed from source via `requirements.txt`) is
required because the custom pipeline/transformer in `kontext/` subclasses it.

## Repository layout

```
kontext/                         Core package
├── pipeline_flux_kontext.py     FLUX Kontext image-to-image pipeline (APITA-aware)
├── transformer_flux.py          Custom FLUX transformer
├── attention/                   Attention processors
│   ├── attention_processor_APITA.py   APITA masked attention
│   └── attention_processor_base.py    Standard ("full") attention
├── pipeline_utils.py            Position-mask / bbox helpers
└── modeling_parallel.py

local_datasets/
└── dataset_crello_from_json.py  Crello dataset + bucketing (used for inference & training)

infer_crello.py                  APITA inference on Crello
infer_lomoe.py                   APITA inference on LoMOE-Bench
infer_infodet.py                 APITA inference on InfoDet (paragraph level)
train.py                         LoRA fine-tuning on Crello
train_utils.py / validation.py / hf_utils.py   Training helpers
```

## Choosing the attention processor

Every inference script accepts `--attention_setting` (or `--attention-setting`
for `infer_infodet.py`):

* `APITA` — masked, region-aware attention (default).
* `full`  — vanilla FLUX Kontext joint attention.

## Running inference

All scripts run with **or without** a fine-tuned checkpoint:

* with a checkpoint: pass `--lora_path /path/to/lora` (and optionally a second,
  RL-fine-tuned adapter via `--rl_lora_path`).
* without a checkpoint: simply omit the LoRA arguments — the base
  FLUX.1-Kontext-dev weights are used with the APITA processor.

The base model defaults to the HuggingFace id `black-forest-labs/FLUX.1-Kontext-dev`;
override it with `--base-model-path` to point at a local snapshot.

### Crello

```bash
python infer_crello.py \
  --crello-json-dir /path/to/crello_processed_json \
  --crello-images-dir /path/to/crello_images \
  --output-dir ./results \
  --exp_name apita_demo \
  --attention_setting APITA \
  --lora_path /path/to/lora        # omit for base-model-only
```

### LoMOE-Bench

```bash
python infer_lomoe.py \
  --lomoe_json_path /path/to/LoMOE-Bench/LoMOE.json \
  --mask_orig_prompts_path /path/to/LoMOE-Bench/utils/mask_orig_prompts.txt \
  --lomoe_base_dir /path/to/LoMOE-Bench \
  --output-dir ./results \
  --exp_name apita_demo \
  --attention_setting APITA \
  --lora_path /path/to/lora        # omit for base-model-only
```

### InfoDet (paragraph level)

```bash
python infer_infodet.py \
  --image-dir /path/to/infodet/images \
  --json-dir /path/to/infodet/json \
  --languages French German Italian Spanish \
  --save-dir ./results \
  --exp-name apita_demo \
  --attention-setting APITA \
  --lora-path /path/to/lora        # omit for base-model-only
```

Pass `--num_samples N` (`--num-samples` for InfoDet) to process only the first
`N` samples. Sharding flags (`--job-index`/`--total-jobs`, or
`--shard-index`/`--num-shards`) are available for multi-GPU/SLURM array runs.

## Training (Crello LoRA)

`train.py` fine-tunes a LoRA adapter on Crello with the APITA attention
processor. Paths are passed explicitly:

```bash
accelerate launch train.py \
  --base-model-path /path/to/FLUX.1-Kontext-dev \
  --crello-json-dir /path/to/crello_processed_json \
  --crello-images-dir /path/to/crello_images \
  --output_dir ./sft_checkpoints \
  --dataset_name crello \
  --attention_setting APITA \
  --mixed_precision bf16 \
  --train_batch_size 1 \
  --gradient_checkpointing \
  --rank 16 \
  --learning_rate 1e-4 \
  --max_train_steps 3000
```

Run `python train.py --help` for the full set of options.
