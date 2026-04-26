# EoMT Inference Notebook — project task 4

## Overview
This notebook runs qualitative and quantitative evaluation of two pretrained EoMT models on the Cityscapes validation set as required in task 4:
- **EoMT-Cityscapes**: trained for semantic segmentation (19 Cityscapes classes)
- **EoMT-COCO**: trained for panoptic segmentation (133 COCO classes)

---

### Requirements
- Python 3.13, conda environment `eomt`
- PyTorch with CUDA support (`torch==2.7.0+cu126`)
- heavy files (.bin of the models and .zip of cityscapes must not be committed). Those files need to be saved in the local repo according to the following repo structure: 
        1. Cityscapes dataset (zip files) in `eomt/datasets/cityscapes/` 
        2. Model weights in `eomt/weights/`



## Notebook Structure

| Cell | Description |
|------|-------------|
| Config | EXCLUDE_CLASSES,  selection of EVAL_SINGLE = True for qualitative analysis (1 image) or False for looping (quantitative analysis) |
| Setup | Imports, model/device selection, config loading. In the setup, you need to selct either coco or cityscapes- pretrained model. This will select the correct .bin. Selct the image index of the image to visualize if you want to run qualitative analysis|
| Load dataset | Loads Cityscapes val set (always, regardless of MODEL and qualitative or quantitative analysis) |
| Load model | Builds EoMT architecture from config |
| Load weights | Loads local `.bin` checkpoint |
| Semantic inference | Qualitative visualization on `img_idx` |
| Panoptic inference | Qualitative visualization on `img_idx` |
| COCO→CS Mapping | Defines mapping from 133 COCO classes to 19 Cityscapes classes |
| Inference function | `get_semantic_logits`: produces (19,H,W) logits for either model |
| Evaluation loop | mIoU per class + mean mIoU (or panoptic visualization if EVAL_SINGLE=True) |

---

##  Quantitative Evaluation Strategy


Both models are evaluated in **Cityscapes 19-class space** using the same metric (`MulticlassJaccardIndex`, which is the IoU).

For the COCO model, COCO logits are projected to Cityscapes space by summing logits of COCO classes that share a Cityscapes equivalent (e.g., COCO `road` + `pavement` → Cityscapes `road`).

### Class Mismatch
~6 Cityscapes classes have no COCO equivalent (`pole`, `fence`, `rider`, `wall`, `terrain`, `traffic sign`). The COCO model will score IoU=0 on these. Use `EXCLUDE_CLASSES` to remove them from the mIoU computation for a fairer comparison.

### COCO → Cityscapes Mapping

| COCO class | Cityscapes class |
|------------|-----------------|
| person (0) | person (12) |
| bicycle (1) | bicycle (18) |
| car (2) | car (13) |
| motorcycle (3) | motorcycle (17) |
| bus (5) | bus (15) |
| train (6) | train (16) |
| truck (7) | truck (14) |
| traffic light (9) | traffic light (6) |
| stop sign (11) | traffic sign (7) |
| road (100), pavement (130) | road (0), sidewalk (1) |
| sky-other (106) | sky (10) |
| tree (110), grass (122) | vegetation (8) |
| wall variants (111-117, 128) | wall (3) |
| house (91), roof (101), building (126) | building (2) |
| playingfield (97), sand (102), dirt (123) | terrain (9) |

### Running
1. Set `MODEL`, `device`, `EVAL_SINGLE` in the config/set-up cells
2. Run all cells sequentially
3. Repeat with the other model to compare results


