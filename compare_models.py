"""
Point 4: Compare EoMT-Cityscapes (semantic) vs EoMT-COCO (panoptic) on Cityscapes val.

Qualitative : saves side-by-side PNGs to --out_dir
Quantitative: add --gt_dir <path/to/gtFine/val> to compute per-class mIoU

Usage
-----
python compare_models.py --n_samples 5
python compare_models.py --n_samples 5 --gt_dir data/gtFine/val
"""

import sys, math, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.datasets import Cityscapes as CS
from torchmetrics.classification import MulticlassJaccardIndex

sys.path.insert(0, str(Path(__file__).parent / "eomt"))
from models.eomt import EoMT
from models.vit import ViT


# ── Cityscapes metadata ──────────────────────────────────────────────────────
CS_CLASSES = [c for c in CS.classes if not c.ignore_in_eval]
CS_NAMES   = [c.name for c in CS_CLASSES]          # 19 names
CS_COLORS  = [c.color for c in CS_CLASSES]          # 19 RGB tuples
ID_TO_TRAIN = {c.id: c.train_id for c in CS.classes}


# ── COCO continuous ID (0-132) → Cityscapes train_id (-1 = no match) ─────────
# COCO panoptic has 80 things (IDs 0-79) + 53 stuff (IDs 80-132)
_COCO_CS_RAW = {
    # Things
    0: 12,  1: 18,  2: 13,  3: 17,             # person, bicycle, car, motorcycle
    5: 15,  6: 16,  7: 14,                      # bus, train, truck
    9: 6,   11: 7,                               # traffic light, stop sign → traffic sign
    # Stuff  (continuous IDs from coco_panoptic.py CLASS_MAPPING)
    91: 2,  101: 2, 126: 2,                      # house, roof, building-other → building
    100: 0, 130: 1,                              # road, pavement → road, sidewalk
    97: 9,  102: 9, 123: 9,                      # playingfield, sand, dirt → terrain
    106: 10,                                     # sky-other → sky
    110: 8, 122: 8, 132: 8,                      # tree, grass → vegetation
    111: 3, 112: 3, 113: 3, 114: 3,             # wall variants → wall
    115: 3, 116: 3, 117: 3, 128: 3,
}
COCO_TO_CS = torch.full((133,), -1, dtype=torch.long)
for _k, _v in _COCO_CS_RAW.items():
    COCO_TO_CS[_k] = _v

# COCO class names for qualitative visualization (just the mapped ones)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 6: "train", 7: "truck", 9: "traffic light", 11: "stop sign",
    91: "house", 100: "road", 101: "roof", 102: "sand",
    106: "sky", 110: "tree", 111: "wall-brick", 112: "wall-concrete",
    113: "wall-other", 122: "grass", 123: "dirt", 126: "building",
    128: "wall", 130: "pavement", 132: "grass",
}


# ── Model helpers ────────────────────────────────────────────────────────────
def load_model(ckpt_path, num_classes, num_q, img_size):
    """Build EoMT and load weights from a LightningModule checkpoint."""
    enc = ViT(img_size=img_size, backbone_name="vit_base_patch14_reg4_dinov2",
              ckpt_path="skip_timm_download")   # any non-None → pretrained=False
    model = EoMT(enc, num_classes=num_classes, num_q=num_q,
                 num_blocks=3, masked_attn_enabled=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    # LightningModule wraps the network under "network.*"
    ckpt = {k[len("network."):]: v for k, v in ckpt.items() if k.startswith("network.")}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:3]}...")
    return model.eval()


# ── Inference helpers ────────────────────────────────────────────────────────
def _window(img_t, img_size):
    """Resize + window a (C,H,W) uint8 tensor. Returns crops list and metadata."""
    H, W = img_t.shape[-2:]
    factor = max(img_size[0] / H, img_size[1] / W)
    nH, nW = round(H * factor), round(W * factor)
    arr = np.array(
        Image.fromarray(img_t.permute(1, 2, 0).numpy()).resize((nW, nH), Image.BILINEAR)
    )
    resized = torch.from_numpy(arr).permute(2, 0, 1)

    size = min(img_size)
    long = max(nH, nW)
    n = math.ceil(long / size)
    overlap = (n * size - long) / (n - 1) if n > 1 else 0

    crops, slices = [], []
    for j in range(n):
        s = int(j * (size - overlap))
        e = s + size
        crop = resized[:, s:e, :] if nH > nW else resized[:, :, s:e]
        crops.append(crop)
        slices.append((s, e, nH > nW))  # (start, end, tall_flag)
    return crops, slices, nH, nW


@torch.no_grad()
def predict_cs(model, img_t, img_size, device):
    """Cityscapes model → (19, H, W) per-pixel logits via windowed inference."""
    H, W = img_t.shape[-2:]
    crops, slices, nH, nW = _window(img_t, img_size)
    acc = torch.zeros(19, nH, nW, device=device)
    cnt = torch.zeros(1,  nH, nW, device=device)

    for crop, (s, e, tall) in zip(crops, slices):
        x = crop.float().to(device) / 255.0
        ml, cl = model(x[None])
        ml = F.interpolate(ml[-1], img_size, mode="bilinear")
        pp = torch.einsum("bqhw,bqc->bchw",
                          ml.sigmoid(), cl[-1].softmax(-1)[..., :-1])[0]
        if tall:
            acc[:, s:e, :] += pp
            cnt[:, s:e, :] += 1
        else:
            acc[:, :, s:e] += pp
            cnt[:, :, s:e] += 1

    return F.interpolate((acc / cnt)[None], (H, W), mode="bilinear")[0]  # (19,H,W)


@torch.no_grad()
def predict_coco(model, img_t, img_size, device):
    """COCO model → (133,H,W) logits remapped to (19,H,W) Cityscapes space."""
    H, W = img_t.shape[-2:]
    factor = min(img_size[0] / H, img_size[1] / W)
    nH, nW = round(H * factor), round(W * factor)
    arr = np.array(
        Image.fromarray(img_t.permute(1, 2, 0).numpy()).resize((nW, nH), Image.BILINEAR)
    )
    padded = np.pad(arr, ((0, img_size[0] - nH), (0, img_size[1] - nW), (0, 0)))
    x = torch.from_numpy(padded).permute(2, 0, 1).float().to(device) / 255.0

    ml, cl = model(x[None])
    ml = F.interpolate(ml[-1], img_size, mode="bilinear")[:, :, :nH, :nW]
    ml = F.interpolate(ml, (H, W), mode="bilinear")
    coco = torch.einsum("bqhw,bqc->bchw",
                        ml.sigmoid(), cl[-1].softmax(-1)[..., :-1])[0]  # (133,H,W)

    mapping = COCO_TO_CS.to(device)
    cs = torch.zeros(19, H, W, device=device)
    for i in range(133):
        c = mapping[i].item()
        if c >= 0:
            cs[c] += coco[i]
    return cs  # (19,H,W)


# ── Visualisation ─────────────────────────────────────────────────────────────
def colorize(pred_hw, colors):
    out = np.zeros((*pred_hw.shape, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        out[pred_hw == i] = color
    return out


def save_comparison(img_np, pred_cs, pred_coco, path):
    cs_vis   = colorize(pred_cs.argmax(0).cpu().numpy(),   CS_COLORS)
    coco_vis = colorize(pred_coco.argmax(0).cpu().numpy(), CS_COLORS)

    patches = [mpatches.Patch(color=np.array(c) / 255, label=n)
               for n, c in zip(CS_NAMES, CS_COLORS)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title in zip(
        axes,
        [img_np, cs_vis, coco_vis],
        ["Input image",
         "EoMT-Cityscapes  (semantic, 19 cls)",
         "EoMT-COCO  (panoptic → CS mapping)"],
    ):
        ax.imshow(data)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=7,
               bbox_to_anchor=(0.5, -0.14))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── GT loading ────────────────────────────────────────────────────────────────
def load_gt(gt_path):
    """gtFine_labelIds.png → train_id tensor (255 = ignore)."""
    raw = np.array(Image.open(gt_path))
    out = np.full_like(raw, 255)
    for label_id, train_id in ID_TO_TRAIN.items():
        if 0 <= train_id < 19:
            out[raw == label_id] = train_id
    return torch.from_numpy(out).long()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir",   default="data/4_val")
    p.add_argument("--cs_ckpt",   default="trained_models/eomt_cityscapes.bin")
    p.add_argument("--coco_ckpt", default="trained_models/eomt_coco.bin")
    p.add_argument("--gt_dir",    default=None,
                   help="Path to gtFine val folder for quantitative eval "
                        "(e.g. data/gtFine/val)")
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--out_dir",   default="comparison_outputs")
    args = p.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Device: {device}")
    print("Loading Cityscapes model (semantic, 19 cls, img_size=1024)...")
    cs_model = load_model(args.cs_ckpt,   num_classes=19,  num_q=100, img_size=(1024, 1024)).to(device)
    print("Loading COCO model (panoptic, 133 cls, img_size=640)...")
    co_model = load_model(args.coco_ckpt, num_classes=133, num_q=200, img_size=(640, 640)).to(device)

    img_paths = sorted(Path(args.val_dir).rglob("*_leftImg8bit.png"))

    # ── Qualitative ───────────────────────────────────────────────────────────
    print(f"\n=== Qualitative: {args.n_samples} samples ===")
    for img_path in img_paths[: args.n_samples]:
        print(f"  {img_path.name}")
        img_np = np.array(Image.open(img_path).convert("RGB"))
        img_t  = torch.from_numpy(img_np).permute(2, 0, 1)

        p_cs   = predict_cs(cs_model,   img_t, (1024, 1024), device)
        p_coco = predict_coco(co_model, img_t, (640, 640),   device)

        save_comparison(img_np, p_cs, p_coco,
                        out_dir / (img_path.stem + "_comparison.png"))

    # ── Quantitative ──────────────────────────────────────────────────────────
    if not args.gt_dir:
        print("\nSkipping quantitative eval (no --gt_dir provided).")
        print("Run with: --gt_dir /path/to/gtFine/val")
        return

    print(f"\n=== Quantitative: {len(img_paths)} images ===")
    gt_dir   = Path(args.gt_dir)
    metric_cs   = MulticlassJaccardIndex(19, ignore_index=255, average=None)
    metric_coco = MulticlassJaccardIndex(19, ignore_index=255, average=None)
    count = 0

    for img_path in img_paths:
        stem = img_path.stem.replace("_leftImg8bit", "")
        city = stem.split("_")[0]
        gt_path = gt_dir / city / f"{stem}_gtFine_labelIds.png"
        if not gt_path.exists():
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))
        img_t  = torch.from_numpy(img_np).permute(2, 0, 1)
        gt     = load_gt(gt_path)

        p_cs   = predict_cs(cs_model,   img_t, (1024, 1024), device).argmax(0).cpu()
        p_coco = predict_coco(co_model, img_t, (640, 640),   device).argmax(0).cpu()

        metric_cs.update(p_cs[None], gt[None])
        metric_coco.update(p_coco[None], gt[None])
        count += 1
        if count % 50 == 0:
            print(f"  processed {count}/{len(img_paths)}")

    if count == 0:
        print("No GT files found. Check --gt_dir path.")
        return

    iou_cs   = metric_cs.compute()
    iou_coco = metric_coco.compute()

    header = f"\n{'Class':<22} {'CS-model (%)':>14} {'COCO-model (%)':>15}"
    sep    = "-" * 53
    rows   = [f"{n:<22} {iou_cs[i].item()*100:>13.1f}  {iou_coco[i].item()*100:>13.1f}"
              for i, n in enumerate(CS_NAMES)]
    footer = f"{'mIoU':<22} {iou_cs.mean().item()*100:>13.1f}  {iou_coco.mean().item()*100:>13.1f}"

    print(header)
    print(sep)
    for r in rows:
        print(r)
    print(sep)
    print(footer)

    txt_path = out_dir / "miou_comparison.txt"
    with open(txt_path, "w") as f:
        f.write(f"Evaluated on {count} images\n")
        f.write(header + "\n" + sep + "\n")
        f.writelines(r + "\n" for r in rows)
        f.write(sep + "\n" + footer + "\n")
    print(f"\nResults saved to {txt_path}")


if __name__ == "__main__":
    main()
