# Comprehensive Road Scene Understanding for Autonomous Driving

Semantic segmentation and out-of-distribution (OoD) **anomaly segmentation** for
road scenes, built around the **Encoder-only Mask Transformer (EoMT)**.
Course project, **Politecnico di Torino**.

**Authors:** Gaia Barberis · Miriam Galerati · Davide Motta · Chiara Apolito

## Overview

Reliable perception for autonomous driving must both segment the known
categories of a scene accurately and flag **unknown / out-of-distribution
objects** that were never seen during training — a safety-critical capability
that closed-set segmentation networks lack by construction.

We study this problem along a single experimental workflow centred on the
mask-classification paradigm (MaskFormer → Mask2Former → DINOv2 → EoMT). We first
characterise how the training domain shapes a segmenter by comparing two EoMT
checkpoints — one trained on **COCO** (panoptic) and one on **Cityscapes**
(semantic) — under one fair semantic protocol. We then close the resulting domain
gap by **fine-tuning** the COCO model on Cityscapes through a staged,
progressively-unfrozen schedule. Finally, we repurpose the resulting segmenters
as **post-hoc anomaly detectors**, contrasting a pixel-based baseline (ERFNet)
with the mask-based EoMT across the standard road-anomaly benchmarks and
calibrating the anomaly scores with **temperature scaling**.

The central finding is that the best segmenter is not the best anomaly detector:
a Cityscapes-trained model segments road scenes best, yet the COCO-pretrained,
fine-tuned model — retaining COCO's far broader visual vocabulary — is the
stronger anomaly detector. The full analysis is in the report
([`report/report.pdf`](report/)); the original assignment is in
[`docs/project_guide.pdf`](docs/project_guide.pdf).

## What this repository contains

The study is implemented as four Colab notebooks under [`notebook/`](notebook/),
each realising one component of the workflow:

- **[`Task4_Comparison.ipynb`](notebook/Task4_Comparison.ipynb)** — cross-domain
  comparison of the COCO- and Cityscapes-trained EoMT, evaluated as semantic
  segmenters on the Cityscapes validation set under a shared protocol
  (COCO$\rightarrow$Cityscapes label mapping; 16 reliably-mappable classes).
- **[`Task5_EoMT_Finetuning_on_Cityscapes.ipynb`](notebook/Task5_EoMT_Finetuning_on_Cityscapes.ipynb)**
  — staged fine-tuning of the COCO model on Cityscapes (class head → class+mask
  heads → full prediction head → full head plus the last DINOv2 blocks); the
  frozen-backbone variants are handled in
  [`eomt/training/lightning_module.py`](eomt/training/lightning_module.py).
- **[`Task7.ipynb`](notebook/Task7.ipynb)** — pixel-based anomaly baseline:
  ERFNet with the MSP, Max-Logit and Max-Entropy post-hoc scores.
- **[`Task8_EoMT_Anomaly.ipynb`](notebook/Task8_EoMT_Anomaly.ipynb)** — mask-based
  anomaly detection: EoMT with MSP, Max-Logit, Max-Entropy and the mask-native
  Rejected-by-All (RbA) score, plus temperature scaling, across all three EoMT
  checkpoints (COCO, Cityscapes, fine-tuned).

```
.
├── notebook/        # Colab notebooks: 00_setup, Task4, Task5, Task7, Task8
├── eomt/            # EoMT code, training configs and pretrained checkpoints
├── eval/            # ERFNet evaluation / anomaly-segmentation tools
├── trained_models/  # ERFNet / EoMT checkpoints used by the baselines
├── report/          # LaTeX report (sources + compiled report.pdf)
├── docs/            # project_guide.pdf (assignment)
└── data/            # local datasets (on Colab the data lives on Drive — see below)
```

## Reproducing the experiments (Google Colab)

The experiments are designed to run on **Google Colab**; use the **same Colab
server** throughout a session.

### 1. Prerequisites
A Google account with Google Drive; access to the GitHub repository
[`ChiaraApolito/MaskArchitectureAnomaly_CourseProject`](https://github.com/ChiaraApolito/MaskArchitectureAnomaly_CourseProject)
(branch `main`); and Google Colab — install the *Colab* extension / use
“Open in Colab” to open the notebooks.

### 2. Runtime
Set **Runtime → Change runtime type → GPU**, preferring an **L4 GPU** (best
speed/memory trade-off for these notebooks).

### 3. Get the code — automatic
Run [`notebook/00_setup.ipynb`](notebook/00_setup.ipynb) **first, in every
session**. It mounts Google Drive, clones (or force-refreshes) the repository
into `MyDrive/MaskArchitectureAnomaly_CourseProject` and installs the
dependencies (`eomt/requirements.txt`). The **code** folder is created by the
clone, so you do not need to create it by hand.

### 4. Prepare data and weights — manual
`00_setup` does **not** download the datasets or the checkpoints. The notebooks
read them from a **separate folder on Drive**,
`MyDrive/FAIML_project_and_presentation/01_Project/`, which you must create and
populate with the layout below:

```
MyDrive/
├── MaskArchitectureAnomaly_CourseProject/        # created by 00_setup (code)
└── FAIML_project_and_presentation/01_Project/    # create this manually
    ├── large_files/
    │   ├── weights/
    │   │   ├── eomt_coco.bin
    │   │   ├── eomt_cityscapes.bin
    │   │   └── finetuned/coco_to_cityscapes_stage2_unfreeze_last/stage2_weights_rerun.bin
    │   └── datasets/
    │       ├── cityscapes/      # leftImg8bit_trainvaltest.zip, gtFine_trainvaltest.zip
    │       └── anomaly/         # Anomaly_Validation_Datasets.zip
    └── results/                 # task4/ … task8/ outputs (created automatically)
```

- **`eomt_coco.bin`, `eomt_cityscapes.bin`** — the two provided EoMT checkpoints,
  from the
  [course Drive folder](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav).
- **`stage2_weights_rerun.bin`** — the fine-tuned model; it is **produced by
  `Task5`** and **consumed by `Task8`**, so run `Task5` before `Task8` (or place a
  pre-computed copy at the path above).
- **Cityscapes** — `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`
  from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) (free
  registration); used by `Task4` and `Task5`.
- **Anomaly datasets** — `Anomaly_Validation_Datasets.zip` (SMIYC RoadAnomaly21 /
  RoadObstacle21, Fishyscapes Lost&Found / Static, RoadAnomaly) from the
  [course Drive folder](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav);
  used by `Task7` and `Task8`.

If your Drive layout differs, edit the path constants at the top of each notebook
(`DRIVE_ROOT`, `LARGE_FILES`, `WEIGHTS_ROOT`, `DATA_ROOT`, `RESULTS_ROOT`).

### 5. Run the notebooks
On the **same** Colab server, open the notebooks in order —
**`00_setup` → `Task4` → `Task5` → `Task7` → `Task8`**. Each task's first cell
only mounts Drive and resolves the paths above, then runs end to end.

### Weights & Biases (fine-tuning only)
`Task5` logs to WandB. Do not hard-code the API key: set `WANDB_API_KEY` in
*Colab Secrets* (key icon in the sidebar); otherwise the notebook prompts for it
securely via `getpass`.

## Results (summary)

Fine-tuning lifts the COCO model from **67.0%** to **76.7%** mIoU (16-class
protocol), approaching the **83.7%** of the natively-trained Cityscapes model.
For anomaly detection, the mask-based EoMT decisively outperforms the pixel-based
ERFNet baseline; the fine-tuned model is the strongest detector on most
benchmarks, and temperature scaling further improves the magnitude- and
rejection-based scores. 
