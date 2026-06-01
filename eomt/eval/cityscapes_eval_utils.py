"""Shared Cityscapes evaluation utilities for Task 4 and Task 5.

This module centralizes the semantic evaluation logic used to compare EoMT
checkpoints on the Cityscapes validation split.

It supports two cases:

1. Cityscapes-space checkpoints
   The model already predicts 19 Cityscapes classes. This is the standard case
   for the Cityscapes pretrained checkpoint and all Task 5 fine-tuned checkpoints.

2. COCO-space checkpoints
   The model predicts 133 COCO classes. The logits are projected to the
   Cityscapes 19-class space using a COCO-to-Cityscapes mapping. If multiple
   COCO classes map to the same Cityscapes class, the maximum logit is used.

Typical usage in a notebook:

    from eval.cityscapes_eval_utils import (
        CS_NAMES,
        DEFAULT_TASK4_EXCLUDED_CLASSES,
        build_eomt_eval_model,
        build_cityscapes_val_dataset,
        evaluate_cityscapes_semantic,
        make_iou_tables,
    )

    model = build_eomt_eval_model(
        config_path=CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        img_size=(512, 512),
        num_classes=19,
        device="cuda:0",
    )

    dataset = build_cityscapes_val_dataset(
        data_cls=CityscapesSemantic,
        data_path=DATA_DIR,
        img_size=(512, 512),
    )

    per_class, miou, pixel_acc = evaluate_cityscapes_semantic(
        model=model,
        dataset=dataset,
        output_space="cityscapes",
        excluded_classes=DEFAULT_TASK4_EXCLUDED_CLASSES,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import importlib
import warnings

import yaml
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

try:
    from torch.amp.autocast_mode import autocast
except Exception:  # pragma: no cover
    autocast = None


IGNORE_INDEX = 255
NUM_CITYSCAPES_CLASSES = 19

CS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

# Classes excluded in Task 4 to make the COCO -> Cityscapes comparison fair.
# 5 = pole, 7 = traffic sign, 12 = rider
DEFAULT_TASK4_EXCLUDED_CLASSES = [5, 7, 12]

# COCO model-index -> Cityscapes train-id mapping used in the Task 4 notebook.
# Unlisted COCO classes are ignored.
COCO_TO_CITYSCAPES_RAW = {
    100: 0,   # road
    123: 1,   # pavement-merged -> sidewalk
    91: 2,    # house -> building
    129: 2,   # building-other-merged -> building
    109: 3,   # wall-brick -> wall
    110: 3,   # wall-stone -> wall
    111: 3,   # wall-tile -> wall
    112: 3,   # wall-wood -> wall
    131: 3,   # wall-other-merged -> wall
    117: 4,   # fence-merged -> fence
    9: 6,     # traffic light
    116: 8,   # tree-merged -> vegetation
    125: 9,   # grass-merged -> terrain
    119: 10,  # sky-other-merged -> sky
    0: 11,    # person
    2: 13,    # car
    7: 14,    # truck
    5: 15,    # bus
    6: 16,    # train
    3: 17,    # motorcycle
    1: 18,    # bicycle
}


def make_coco_to_cityscapes_tensor(num_coco_classes: int = 133) -> torch.Tensor:
    """Return a tensor mapping COCO model indices to Cityscapes train ids."""
    mapping = torch.full((num_coco_classes,), -1, dtype=torch.long)
    for coco_id, cityscapes_id in COCO_TO_CITYSCAPES_RAW.items():
        if coco_id < num_coco_classes:
            mapping[coco_id] = cityscapes_id
    return mapping


COCO_TO_CITYSCAPES = make_coco_to_cityscapes_tensor(133)


def resolve_class(class_path: str) -> type:
    """Resolve a fully qualified class path, e.g. 'models.eomt.EoMT'."""
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def normalize_img_size(img_size: int | Sequence[int]) -> tuple[int, int]:
    """Convert an int/list/tuple image size to a two-element tuple."""
    if isinstance(img_size, int):
        return (img_size, img_size)
    if len(img_size) != 2:
        raise ValueError(f"img_size must have two elements, got: {img_size}")
    return (int(img_size[0]), int(img_size[1]))


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_state_dict_from_file(weights_path: str | Path) -> dict[str, torch.Tensor]:
    """Load either an exported .bin state_dict or a Lightning .ckpt checkpoint."""
    weights_path = Path(weights_path)
    obj = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise TypeError(f"Unsupported checkpoint object type from {weights_path}: {type(obj)!r}")

    # Remove loss-only tensors when present. They are not needed for inference and
    # may create harmless loading mismatches across configs.
    return {
        key.replace("._orig_mod", ""): value
        for key, value in state_dict.items()
        if "criterion.empty_weight" not in key
    }


def _infer_img_size_from_config(config: Mapping[str, Any], fallback: int | Sequence[int]) -> tuple[int, int]:
    """Return the explicit evaluation image size passed by the notebook.

    The training YAML can contain a different img_size from the one used in a
    notebook through CLI overrides. For evaluation we therefore trust the explicit
    `img_size` argument passed by the notebook.
    """
    return normalize_img_size(fallback)


def build_eomt_eval_model(
    config_path: str | Path,
    weights_path: str | Path,
    img_size: int | Sequence[int] = (512, 512),
    num_classes: int | None = None,
    device: str | torch.device | None = None,
    load_strict: bool = False,
    force_masked_attn_enabled: bool | None = None,
    extra_model_kwargs: Mapping[str, Any] | None = None,
):
    """Build an EoMT LightningModule from config and load evaluation weights.

    Parameters
    ----------
    config_path:
        YAML config used to build the model.
    weights_path:
        Exported .bin state_dict or Lightning .ckpt checkpoint.
    img_size:
        Explicit evaluation image size used to build the encoder and validation
        preprocessing. This value is taken from the notebook, not from the YAML,
        because the notebooks often override img_size through CLI arguments.
    num_classes:
        Number of model classes before the no-object class. Use 19 for
        Cityscapes-space checkpoints and 133 for the original COCO checkpoint.
        If omitted, this function tries to infer it from the config and falls
        back to 19.
    device:
        Device for the returned model. Defaults to cuda:0 if available.
    load_strict:
        Passed to load_state_dict. False is usually safer for evaluation.
    force_masked_attn_enabled:
        Optional override for network.masked_attn_enabled.
    extra_model_kwargs:
        Optional kwargs injected into the LightningModule constructor.
    """
    config = load_yaml(config_path)
    eval_img_size = _infer_img_size_from_config(config, img_size)

    if num_classes is None:
        num_classes = int(config.get("model", {}).get("init_args", {}).get("num_classes", 19))

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_init = dict(config["model"].get("init_args", {}) or {})
    network_cfg = model_init.pop("network")

    encoder_cfg = dict(network_cfg["init_args"]["encoder"])
    encoder_cls = resolve_class(encoder_cfg["class_path"])
    encoder_init = dict(encoder_cfg.get("init_args", {}) or {})
    encoder = encoder_cls(img_size=eval_img_size, **encoder_init)

    network_cls = resolve_class(network_cfg["class_path"])
    network_init = dict(network_cfg.get("init_args", {}) or {})
    network_init.pop("encoder", None)

    if force_masked_attn_enabled is not None:
        network_init["masked_attn_enabled"] = force_masked_attn_enabled

    network = network_cls(
        encoder=encoder,
        num_classes=num_classes,
        **network_init,
    )

    # Some panoptic configs pass dataset-specific constructor arguments such as
    # stuff_classes through the data section rather than through model.init_args.
    data_init = config.get("data", {}).get("init_args", {}) or {}
    if "stuff_classes" in data_init and "stuff_classes" not in model_init:
        model_init["stuff_classes"] = data_init["stuff_classes"]

    # We load the requested evaluation weights manually below. Avoid also loading
    # the training-time ckpt_path from the YAML, because that would first load the
    # original initialization checkpoint.
    model_init["ckpt_path"] = None
    model_init["load_ckpt_class_head"] = True

    if extra_model_kwargs:
        model_init.update(dict(extra_model_kwargs))

    lit_cls = resolve_class(config["model"]["class_path"])
    model = lit_cls(
        img_size=eval_img_size,
        num_classes=num_classes,
        network=network,
        **model_init,
    )

    state_dict = load_state_dict_from_file(weights_path)
    incompatible = model.load_state_dict(state_dict, strict=load_strict)

    if not load_strict:
        missing = [k for k in incompatible.missing_keys if "criterion.empty_weight" not in k]
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            warnings.warn(
                f"Missing keys while loading {weights_path}: {missing[:20]}"
                + (" ..." if len(missing) > 20 else "")
            )
        if unexpected:
            warnings.warn(
                f"Unexpected keys while loading {weights_path}: {unexpected[:20]}"
                + (" ..." if len(unexpected) > 20 else "")
            )

    return model.eval().to(device)


def build_cityscapes_val_dataset(
    data_cls: type | str,
    data_path: str | Path,
    img_size: int | Sequence[int] = (512, 512),
    batch_size: int = 1,
    num_workers: int = 0,
    check_empty_targets: bool = False,
):
    """Build the Cityscapes validation dataset used by the project datamodule."""
    if isinstance(data_cls, str):
        data_cls = resolve_class(data_cls)

    data = data_cls(
        path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        check_empty_targets=check_empty_targets,
        img_size=normalize_img_size(img_size),
    )
    data.setup("fit")
    return data.val_dataloader().dataset


def semantic_logits(
    model,
    img: torch.Tensor,
    output_space: str = "cityscapes",
    device: str | torch.device | None = None,
    coco_to_cityscapes: torch.Tensor | None = None,
    autocast_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run semantic inference for one image and return Cityscapes-space logits.

    output_space:
        - 'cityscapes': model output is already 19 Cityscapes classes.
        - 'coco': model output is COCO-space and is mapped to 19 Cityscapes classes.
    """
    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:]]
        crops, origins = model.window_imgs_semantic(imgs)

        use_autocast = torch.cuda.is_available() and str(device).startswith("cuda")
        if autocast is not None and use_autocast:
            with autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
                mask_logits_per_layer, class_logits_per_layer = model(crops)
                mask_logits = F.interpolate(
                    mask_logits_per_layer[-1],
                    model.img_size,
                    mode="bilinear",
                )
                crop_logits = model.to_per_pixel_logits_semantic(
                    mask_logits,
                    class_logits_per_layer[-1],
                )
        else:
            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1],
                model.img_size,
                mode="bilinear",
            )
            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits,
                class_logits_per_layer[-1],
            )

        logits = model.revert_window_logits_semantic(
            crop_logits,
            origins,
            img_sizes,
        )[0]

    if output_space.lower() == "cityscapes":
        return logits.cpu()

    if output_space.lower() != "coco":
        raise ValueError("output_space must be either 'cityscapes' or 'coco'.")

    if coco_to_cityscapes is None:
        coco_to_cityscapes = COCO_TO_CITYSCAPES

    coco_to_cityscapes = coco_to_cityscapes.to(logits.device)
    num_coco_classes = logits.shape[0]
    height, width = logits.shape[-2:]
    cityscapes_logits = torch.full(
        (NUM_CITYSCAPES_CLASSES, height, width),
        -float("inf"),
        device=logits.device,
        dtype=logits.dtype,
    )

    for coco_id in range(min(num_coco_classes, len(coco_to_cityscapes))):
        cityscapes_id = int(coco_to_cityscapes[coco_id].item())
        if cityscapes_id >= 0:
            cityscapes_logits[cityscapes_id] = torch.maximum(
                cityscapes_logits[cityscapes_id],
                logits[coco_id],
            )

    return torch.nan_to_num(cityscapes_logits, neginf=-1e9).cpu()


def evaluate_cityscapes_semantic(
    model,
    dataset,
    output_space: str = "cityscapes",
    excluded_classes: Sequence[int] | None = None,
    ignore_index: int = IGNORE_INDEX,
    device: str | torch.device | None = None,
    model_name: str = "model",
    print_every: int = 50,
) -> tuple[torch.Tensor, float, float]:
    """Evaluate one model on Cityscapes semantic segmentation.

    Returns
    -------
    per_class_iou:
        Tensor with IoU (%) for all 19 Cityscapes classes.
    miou:
        Mean IoU (%) over non-excluded classes.
    pixel_acc:
        Micro pixel accuracy (%) over non-ignored pixels.
    """
    if excluded_classes is None:
        excluded_classes = []
    excluded_classes = list(excluded_classes)

    if device is None:
        device = next(model.parameters()).device

    iou_metric = MulticlassJaccardIndex(
        num_classes=NUM_CITYSCAPES_CLASSES,
        ignore_index=ignore_index,
        average=None,
        validate_args=False,
    )
    acc_metric = MulticlassAccuracy(
        num_classes=NUM_CITYSCAPES_CLASSES,
        ignore_index=ignore_index,
        average="micro",
        validate_args=False,
    )

    for index in range(len(dataset)):
        img, target = dataset[index]

        gt = model.to_per_pixel_targets_semantic([target], ignore_index)[0].cpu()
        for class_id in excluded_classes:
            gt[gt == class_id] = ignore_index

        logits = semantic_logits(
            model=model,
            img=img,
            output_space=output_space,
            device=device,
        )
        pred = logits.argmax(0).cpu()

        iou_metric.update(pred.unsqueeze(0), gt.unsqueeze(0))
        acc_metric.update(pred.unsqueeze(0), gt.unsqueeze(0))

        if print_every and (index + 1) % print_every == 0:
            print(f"[{model_name}] {index + 1}/{len(dataset)} images processed")

    per_class_iou = iou_metric.compute().cpu() * 100
    pixel_acc = float(acc_metric.compute().cpu()) * 100

    valid = torch.ones(NUM_CITYSCAPES_CLASSES, dtype=torch.bool)
    for class_id in excluded_classes:
        valid[class_id] = False

    miou = float(per_class_iou[valid].mean())

    return per_class_iou, miou, pixel_acc


def make_iou_tables(
    results: Mapping[str, Mapping[str, Any]],
    excluded_classes: Sequence[int] | None = None,
):
    """Create summary and per-class IoU pandas DataFrames from eval results."""
    import pandas as pd

    if excluded_classes is None:
        excluded_classes = []

    df_per_class = pd.DataFrame({
        "class_id": list(range(NUM_CITYSCAPES_CLASSES)),
        "class_name": CS_NAMES,
        "excluded": [i in excluded_classes for i in range(NUM_CITYSCAPES_CLASSES)],
    })

    for model_name, result in results.items():
        df_per_class[f"IoU {model_name} (%)"] = result["per_class_iou"]

    df_summary = pd.DataFrame([
        {
            "model": model_name,
            "description": result.get("description", ""),
            "mIoU (%)": result["miou"],
            "pixel_acc (%)": result["pixel_acc"],
        }
        for model_name, result in results.items()
    ])

    return df_summary, df_per_class
