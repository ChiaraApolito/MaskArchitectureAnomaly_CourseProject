# ---------------------------------------------------------------
# Versione di evalAnomaly.py adattata a EoMT.
#
# Gestisce:
#   1) checkpoint COCO / Cityscapes / fine-tuned
#   2) MSP / MaxLogit / Entropy / RbA
#   3) temperature scaling
#   4) salvataggio CSV
# ---------------------------------------------------------------

import os
import sys
import csv
import glob
import random
from pathlib import Path
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------
# Permette di importare models.vit e models.eomt anche se il file è in eval/
# ---------------------------------------------------------------------
# REPO_ROOT = Path(__file__).resolve().parents[1]
# if str(REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPO_ROOT))

# from models.vit import ViT
# from models.eomt import EoMT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EOMT_ROOT = PROJECT_ROOT / "eomt"

for p in [PROJECT_ROOT, EOMT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.vit import ViT
from models.eomt import EoMT

# ---------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

IMG_HEIGHT = 512
IMG_WIDTH = 1024
input_transform = Compose(
    [
        Resize((IMG_HEIGHT, IMG_WIDTH), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST),
    ]
)


# ---------------------------------------------------------------------
# Preset checkpoint
# ---------------------------------------------------------------------
def apply_preset(args):
    """
    Preset pratici per evitare di scrivere ogni volta num_classes e num_q.

    Cityscapes semantic:
        num_classes = 19
        num_q = 100

    Fine-tuned su Cityscapes:
        stessa architettura del Cityscapes semantic
        num_classes = 19
        num_q = 200

    COCO panoptic:
        num_classes = 133
        num_q = 200
    """

    if args.preset == "cityscapes":
        args.num_classes = 19
        args.num_q = 100

    elif args.preset == "finetuned":
        args.num_classes = 19
        args.num_q = 200

    elif args.preset == "coco":
        args.num_classes = 133
        args.num_q = 200

    else:
        raise ValueError(f"Preset non riconosciuto: {args.preset}")

    return args


# ---------------------------------------------------------------------
# Costruzione modello EoMT
# ---------------------------------------------------------------------
def build_eomt(args):
    """
    Costruisce EoMT in modo coerente con i config:

    models.eomt.EoMT
      encoder: models.vit.ViT
        backbone_name: vit_base_patch14_reg4_dinov2

    Nota:
    - ViT contiene pixel_mean/pixel_std.
    - Quindi le immagini devono entrare come tensori [0, 1].
    - Non bisogna applicare Normalize() nel transform.
    """

    encoder = ViT(
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        patch_size=args.patch_size,
        backbone_name=args.backbone_name,
        ckpt_path="disable_timm_pretrained",
    )

    if args.preset == "finetuned":
        from models.eomt_finetuning import EoMT as EoMTClass
        masked_attn_enabled = False
    else:
        from models.eomt import EoMT as EoMTClass
        masked_attn_enabled = True

    model = EoMTClass(
        encoder=encoder,
        num_classes=args.num_classes,
        num_q=args.num_q,
        num_blocks=args.num_blocks,
        masked_attn_enabled=masked_attn_enabled,
    )

    return model


# ---------------------------------------------------------------------
# Caricamento pesi
# ---------------------------------------------------------------------
def load_eomt_weights(model, weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    new_ckpt = {}

    for k, v in ckpt.items():
        if "criterion.empty_weight" in k:
            continue

        k = k.replace("._orig_mod", "")

        if k.startswith("network."):
            k = k.replace("network.", "", 1)

        if k.startswith("module."):
            k = k.replace("module.", "", 1)

        new_ckpt[k] = v

    pos_key = "encoder.backbone.pos_embed"

    if pos_key in new_ckpt and pos_key in model.state_dict():
        ckpt_pos = new_ckpt[pos_key]
        model_pos = model.state_dict()[pos_key]

        if ckpt_pos.shape != model_pos.shape:
            print(
                f"Interpolating pos_embed from {tuple(ckpt_pos.shape)} "
                f"to {tuple(model_pos.shape)}"
            )

            _, n_ckpt, dim = ckpt_pos.shape
            _, n_model, _ = model_pos.shape

            h_ckpt = w_ckpt = int(n_ckpt ** 0.5)

            h_model, w_model = model.encoder.backbone.patch_embed.grid_size

            pos_2d = ckpt_pos.reshape(1, h_ckpt, w_ckpt, dim).permute(0, 3, 1, 2)

            pos_2d = F.interpolate(
                pos_2d,
                size=(h_model, w_model),
                mode="bicubic",
                align_corners=False,
            )

            new_ckpt[pos_key] = pos_2d.permute(0, 2, 3, 1).reshape(1, h_model * w_model, dim)
    incompatible = model.load_state_dict(new_ckpt, strict=False)

    print("Checkpoint caricato:", weights_path)
    print("Numero chiavi checkpoint:", len(new_ckpt))

    if incompatible.missing_keys:
        print("\nMissing keys:")
        for k in incompatible.missing_keys[:30]:
            print("  -", k)
        if len(incompatible.missing_keys) > 30:
            print("  ...")

    if incompatible.unexpected_keys:
        print("\nUnexpected keys:")
        for k in incompatible.unexpected_keys[:30]:
            print("  -", k)
        if len(incompatible.unexpected_keys) > 30:
            print("  ...")

    return model


# ---------------------------------------------------------------------
# Conversione output EoMT -> pixel logits
# ---------------------------------------------------------------------
def eomt_to_pixel_logits(mask_logits, class_logits, target_size):
    """
    Stessa logica di training/lightning_module.py:

    torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )

    Output:
        pixel_logits: [B, C, H, W]
    """

    if mask_logits.shape[-2:] != target_size:
        mask_logits = F.interpolate(
            mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    pixel_logits = torch.einsum(
        "bqhw,bqc->bchw",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )

    return pixel_logits


# ---------------------------------------------------------------------
# Score anomaly MSP / MaxLogit / Entropy
# ---------------------------------------------------------------------
def compute_anomaly_score(pixel_logits, method, temperature=1.0):
    """
    logits: torch.Tensor [B, C, H, W]
    returns: np.ndarray [H, W]
    """

    logits = pixel_logits / temperature

    if method == "maxlogit":
        score = -torch.max(logits, dim=1)[0]

    elif method == "msp":
        probs = torch.softmax(logits, dim=1)
        score = 1.0 - torch.max(probs, dim=1)[0]

    elif method == "entropy":
        probs = torch.softmax(logits, dim=1)
        score = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

    else:
        raise ValueError(f"Metodo non riconosciuto: {method}")

    return score


# ---------------------------------------------------------------------
# Score RbA semplice
# ---------------------------------------------------------------------
def compute_rba_score(mask_logits, class_logits, target_size):
    """
    RbA semplificato per mask architecture.

    Idea:
        un pixel è normale se almeno una query/mask lo accetta con alta confidenza.
        un pixel è anomalo se è rifiutato da tutte le query.

    Output:
        anomaly_score: [B, H, W]
    """

    if mask_logits.shape[-2:] != target_size:
        mask_logits = F.interpolate(
            mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    class_probs = class_logits.softmax(dim=-1)[..., :-1]  # rimuove no-object
    mask_probs = mask_logits.sigmoid()

    query_conf = class_probs.max(dim=-1)[0]               # [B, Q]
    acceptance = query_conf[:, :, None, None] * mask_probs # [B, Q, H, W]

    normal_score = acceptance.max(dim=1)[0]               # [B, H, W]
    anomaly_score = 1.0 - normal_score

    return anomaly_score


# ---------------------------------------------------------------------
# Path e conversione GT anomaly
# ---------------------------------------------------------------------
def get_gt_path(path):
    pathGT = path.replace("images", "labels_masks")

    if "RoadObsticle21" in pathGT or "RoadObstacle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")

    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    return pathGT


def convert_gt(ood_gts, pathGT):
    """
    Mantiene la stessa logica del vecchio evalAnomaly.py.
    Valori finali:
        0   = normale
        1   = anomalo
        255 = ignore
    """

    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)

    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

    return ood_gts


# ---------------------------------------------------------------------
# Salvataggio CSV
# ---------------------------------------------------------------------
def save_csv(args, auprc, fpr95, num_images, num_pixels, num_anomaly_pixels):
    file_exists = os.path.exists(args.results_csv)

    row = {
        "checkpoint_name": args.checkpoint_name,
        "preset": args.preset,
        "weights": args.weights,
        "input": args.input[0],
        "method": args.method,
        "temperature": args.temperature,
        "num_classes": args.num_classes,
        "num_q": args.num_q,
        "num_blocks": args.num_blocks,
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "AUPRC": auprc * 100.0,
        "FPR95": fpr95 * 100.0,
        "num_images": num_images,
        "num_pixels": num_pixels,
        "num_anomaly_pixels": num_anomaly_pixels,
    }

    with open(args.results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print("Risultati salvati in:", args.results_csv)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        default="../datasets/anomaly/Anomaly_Validation_Datasets/RoadAnomaly/images/*.jpg",
        nargs="+",
        help="Glob pattern immagini, come nel vecchio evalAnomaly.py",
    )

    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--checkpoint-name",
        default="eomt",
        help="Nome libero da salvare nel CSV",
    )

    parser.add_argument(
        "--preset",
        default="cityscapes",
        choices=["cityscapes", "coco", "finetuned"],
        help="Sceglie automaticamente num_classes e num_q",
    )

    parser.add_argument("--num-blocks", type=int, default=3)

    parser.add_argument(
        "--method",
        default="maxlogit",
        choices=["msp", "maxlogit", "entropy", "rba"],
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling. Usato per MSP/MaxLogit/Entropy. Ignorato per RbA.",
    )

    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--backbone-name", default="vit_base_patch14_reg4_dinov2")

    parser.add_argument("--results-csv", default="results_task8_eomt.csv")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    args = apply_preset(args)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)
    print("Preset:", args.preset)
    print("num_classes:", args.num_classes)
    print("num_q:", args.num_q)
    print("Metodo:", args.method)
    print("Temperature:", args.temperature)

    model = build_eomt(args)
    model = load_eomt_weights(model, args.weights)
    model = model.to(device)
    model.eval()

    anomaly_score_list = []
    ood_gts_list = []

    paths = sorted(glob.glob(os.path.expanduser(str(args.input[0]))))
    print("Numero immagini trovate:", len(paths))

    if len(paths) == 0:
        raise RuntimeError("Nessuna immagine trovata. Controlla --input.")

    for i, path in enumerate(paths):
        print(f"[{i+1}/{len(paths)}] {path}")

        image = Image.open(path).convert("RGB")
        images = input_transform(image).unsqueeze(0).float().to(device)

        with torch.no_grad():
            mask_logits_per_layer, class_logits_per_layer = model(images)

        mask_logits = mask_logits_per_layer[-1]
        class_logits = class_logits_per_layer[-1]

        target_size = (IMG_HEIGHT, IMG_WIDTH)

        if args.method == "rba":
            score = compute_rba_score(mask_logits, class_logits, target_size)
        else:
            pixel_logits = eomt_to_pixel_logits(mask_logits, class_logits, target_size)
            score = compute_anomaly_score(
                pixel_logits,
                method=args.method,
                temperature=args.temperature,
            )

        anomaly_result = score.squeeze(0).detach().cpu().numpy()

        pathGT = get_gt_path(path)

        if not os.path.exists(pathGT):
            print("GT non trovata, salto:", pathGT)
            continue

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)
        ood_gts = convert_gt(ood_gts, pathGT)

        if 1 not in np.unique(ood_gts):
            continue

        valid_mask = (ood_gts == 0) | (ood_gts == 1)

        ood_gts_list.append(ood_gts[valid_mask])
        anomaly_score_list.append(anomaly_result[valid_mask])

        del images, score, anomaly_result, mask_logits, class_logits
        if device.type == "cuda":torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        raise RuntimeError("Nessuna ground truth valida trovata.")

    val_label = np.concatenate(ood_gts_list)
    val_out = np.concatenate(anomaly_score_list)

    auprc = average_precision_score(val_label, val_out)
    fpr95 = fpr_at_95_tpr(val_out, val_label)

    print("\n-------------------------")
    print(f"AUPRC score: {auprc * 100.0}")
    print(f"FPR@TPR95:   {fpr95 * 100.0}")
    print("-------------------------\n")

    save_csv(
        args=args,
        auprc=auprc,
        fpr95=fpr95,
        num_images=len(ood_gts_list),
        num_pixels=len(val_label),
        num_anomaly_pixels=int(val_label.sum()),
    )


if __name__ == "__main__":
    main()