# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is a course project with two main components:
1. **EoMT (Edge-aware Mask Transformers)** — a mask-based segmentation model built on DINOv2 ViT backbones, supporting semantic, instance, and panoptic segmentation on Cityscapes, COCO, and ADE20K.
2. **ERFNet baseline** — an efficient CNN for road scene semantic segmentation and anomaly detection.

## Rules
1. Remember to always read and double check the file `project_guide.pdf`. This guide provide you the path to follow for successfully developing and solving the proposed exercises.
2. Do not edit a file wihout ask it explicitly to the user.

## Environment Setup

```bash
# Root project (ERFNet / eval scripts)
py -3.11 -m venv .venv
.venv/Scripts/Activate
pip install -r requirements.txt

# EoMT (separate env recommended)
conda create -n eomt python==3.13.2
conda activate eomt
pip install -r eomt/requirements.txt
```

## Common Commands

### Train EoMT (from `eomt/` directory)
```bash
python main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

### Fine-tune from checkpoint
```bash
python main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

### Validate EoMT
```bash
python main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

### ERFNet anomaly evaluation (from `eval/` directory)
```bash
python evalAnomaly.py --input '/path/to/images/*.png'
python eval_cityscapes_color.py --datadir /path/to/cityscapes/ --subset val
python eval_iou.py --datadir /path/to/cityscapes/ --subset val
python eval_forwardTime.py --width 1024 --height 512
```

## Architecture

### EoMT (`eomt/`)

Entry point: `eomt/main.py` — uses `jsonargparse` + PyTorch Lightning CLI (`python main.py fit|validate|test -c config.yaml`).

**Model flow:**
```
Image (640×640 or 1024×1024)
  → ViT encoder (DINOv2 backbone via timm/HuggingFace)
  → Learnable mask queries (num_q=100) attend to patch features
  → num_blocks Scale Blocks (ConvTranspose2d upsampling + depthwise-sep conv)
  → Per-block: class logits (Linear) + mask logits (MLP + einsum with features)
  → Hungarian matching loss (focal + dice for masks, focal for class)
```

Key files:
- `eomt/models/eomt.py` — `EoMT` model: encoder, query embeddings, mask head, class head, scale blocks
- `eomt/models/vit.py` — `ViT` wrapper around timm/transformers backbone
- `eomt/models/scale_block.py` — decoder upsampling blocks with LayerNorm2d
- `eomt/training/lightning_module.py` — base LightningModule with LLRD optimizer, polynomial LR schedule, gradient clipping
- `eomt/training/mask_classification_semantic.py` — semantic segmentation training with Jaccard IoU metric
- `eomt/training/mask_classification_loss.py` — Mask2Former-style Hungarian matcher + loss terms
- `eomt/datasets/dataset.py` — reads directly from zip files (no extraction needed)
- `eomt/datasets/cityscapes_semantic.py` — Cityscapes 19-class loader with augmentations

**Attention mask annealing**: During training, query-to-patch attention is progressively masked per block (`attn_mask_annealing_enabled`). Start/end steps are specified per-block in the YAML config.

### ERFNet (`eval/`)

- `eval/erfnet.py` — Encoder (4 downsampling + bottleneck blocks) + Decoder (4 upsampling blocks)
- `eval/evalAnomaly.py` — anomaly score = `1 - max(softmax(logits))` per pixel; evaluates on Road Anomaly, Road Obstacle, Fishyscapes
- Pre-trained weights in `trained_models/`

### Configuration System

All EoMT configs live in `eomt/configs/dinov2/{dataset}/{task}/`. Every parameter can be overridden from the CLI with `--section.key value`. Key parameters:

| Parameter | Meaning |
|---|---|
| `model.network.num_q` | Number of mask queries (default 100) |
| `model.network.num_blocks` | Decoder blocks (typically 3–4) |
| `model.network.encoder.backbone_name` | timm model name (e.g. `vit_base_patch14_reg4_dinov2`) |
| `model.attn_mask_annealing_*` | Per-block annealing start/end step lists |
| `data.batch_size` | Per-GPU batch size |
| `trainer.devices` | Number of GPUs |

### Dataset Handling

Cityscapes (and other datasets) are loaded **directly from zip files** — no extraction required. Place `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` in one directory and pass it via `--data.path`. The `Dataset` class in `eomt/datasets/dataset.py` handles in-memory zip reads with multiprocessing.

### Metrics

- Semantic: Jaccard Index (IoU) via `torchmetrics`
- Instance: mAP
- Panoptic: PQ/SQ/RQ
- Anomaly: AUC, FPR@95TPR

### Experiment Tracking

WandB is configured by default in YAML configs. Run `wandb login` before training. Project name and run name are set in the `trainer.logger` section of each config.
