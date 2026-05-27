from pathlib import Path
import shutil
import torch
from torch.utils.data import Dataset


def precompute_head_cache(
    model,
    train_dataloader,
    cache_dir,
    device=None,
    max_batches=None,
    overwrite=False,
):
    """
    Crea una cache LEGGERA per head-only fine-tuning.

    Salva per ogni immagine:
        - q_features: input della class_head, shape [Q, C]
        - query_class_targets: target di classe per query, shape [Q]

    NON salva:
        - mask_logits
        - masks complete
        - target completi

    Nota:
    il matching viene fatto una sola volta durante il precompute.
    Durante il training si usa cross-entropy diretta sulla class_head.
    """

    cache_dir = Path(cache_dir)

    if overwrite and cache_dir.exists():
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(cache_dir.glob("sample_*.pt"))
    if existing_files and not overwrite:
        print(
            f"[light-cache] Found {len(existing_files)} cached samples in {cache_dir}. "
            f"Skipping precompute. Set overwrite=True to recreate."
        )
        return len(existing_files)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_training_state = model.training
    old_network_training_state = model.network.training
    old_requires_grad = {
        name: p.requires_grad
        for name, p in model.named_parameters()
    }

    model = model.to(device)
    model.eval()
    model.network.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    sample_idx = 0
    no_object_class = model.num_classes

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                imgs, targets = batch
                imgs = imgs.to(device, non_blocking=True).float()

                # LightningModule.forward normalmente fa imgs / 255.0.
                # Qui chiamiamo direttamente model.network, quindi dividiamo qui.
                x = imgs / 255.0

                # forward frozen: serve solo durante il precompute
                q_features, mask_logits = model.network.forward_frozen_features(x)

                # class logits iniziali della head corrente
                class_logits = model.network.class_head(q_features)

                mask_labels = [
                    target["masks"].to(device).to(mask_logits.dtype)
                    for target in targets
                ]
                class_labels = [
                    target["labels"].to(device).long()
                    for target in targets
                ]

                # Matching fatto una sola volta
                indices = model.criterion.matcher(
                    masks_queries_logits=mask_logits,
                    mask_labels=mask_labels,
                    class_queries_logits=class_logits,
                    class_labels=class_labels,
                )

                for b in range(q_features.shape[0]):
                    src_idx, tgt_idx = indices[b]

                    query_class_targets = torch.full(
                        (q_features.shape[1],),
                        fill_value=no_object_class,
                        dtype=torch.long,
                    )

                    if len(src_idx) > 0:
                        labels_b = targets[b]["labels"].long()
                        matched_labels = labels_b[tgt_idx.cpu()]
                        query_class_targets[src_idx.cpu()] = matched_labels

                    item = {
                        "q_features": q_features[b].detach().cpu().half(),
                        "query_class_targets": query_class_targets.cpu(),
                    }

                    torch.save(item, cache_dir / f"sample_{sample_idx:06d}.pt")
                    sample_idx += 1

                if batch_idx % 20 == 0:
                    print(
                        f"[light-cache] batch {batch_idx} processed "
                        f"| saved samples: {sample_idx}"
                    )

    finally:
        model.train(old_training_state)
        model.network.train(old_network_training_state)

        for name, p in model.named_parameters():
            if name in old_requires_grad:
                p.requires_grad_(old_requires_grad[name])

    print(f"[light-cache] completed. Saved {sample_idx} samples in: {cache_dir}")
    return sample_idx


# DA ELIMINARE SE IL CACHE- FINETUNING HEAD+ULTIMI LATERS NON FUNZIONA
def precompute_backbone_token_cache(
    model,
    train_dataloader,
    cache_dir,
    device=None,
    overwrite=False,
):
    cache_dir = Path(cache_dir)

    if overwrite and cache_dir.exists():
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(cache_dir.glob("batch_*.pt"))
    if existing_files and not overwrite:
        print(
            f"[backbone-cache] Found {len(existing_files)} cached batches in {cache_dir}. "
            f"Skipping precompute. Set overwrite=True to recreate."
        )
        return len(existing_files)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_training_state = model.training
    old_network_training_state = model.network.training
    old_requires_grad = {
        name: p.requires_grad
        for name, p in model.named_parameters()
    }

    model = model.to(device)
    model.eval()
    model.network.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    stop_i = len(model.network.encoder.backbone.blocks) - model.network.num_blocks

    sample_idx = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_dataloader):
                imgs, targets = batch
                imgs = imgs.to(device).float()
                x = imgs / 255.0

                x = (x - model.network.encoder.pixel_mean) / model.network.encoder.pixel_std
                x = model.network.encoder.backbone.patch_embed(x)

                if hasattr(model.network.encoder.backbone, "_pos_embed"):
                    x = model.network.encoder.backbone._pos_embed(x)

                for i, block in enumerate(model.network.encoder.backbone.blocks[:stop_i]):
                    if hasattr(block, "attn"):
                        attn = block.attn
                    else:
                        attn = block.attention

                    attn_out = model.network._attn(
                        attn,
                        block.norm1(x),
                        mask=None,
                        rope=None,
                    )

                    if hasattr(block, "ls1"):
                        x = x + block.ls1(attn_out)
                    elif hasattr(block, "layer_scale1"):
                        x = x + block.layer_scale1(attn_out)

                    mlp_out = block.mlp(block.norm2(x))

                    if hasattr(block, "ls2"):
                        x = x + block.ls2(mlp_out)
                    elif hasattr(block, "layer_scale2"):
                        x = x + block.layer_scale2(mlp_out)

                item = {
                    "tokens": x.detach().cpu().half(),
                    # "targets": [
                    #     {
                    #         "labels": t["labels"].cpu().long(),
                    #         "masks": t["masks"].to(torch.bool).cpu(),
                    #     }
                    #     for t in targets
                    # ],
                }

                torch.save(item, cache_dir / f"batch_{batch_idx:06d}.pt")
                sample_idx += x.shape[0]

                if batch_idx % 20 == 0:
                    print(
                        f"[backbone-cache] batch {batch_idx} processed "
                        f"| saved samples: {sample_idx}"
                    )

    finally:
        model.train(old_training_state)
        model.network.train(old_network_training_state)

        for name, p in model.named_parameters():
            if name in old_requires_grad:
                p.requires_grad_(old_requires_grad[name])

    print(f"[backbone-cache] completed. Saved {sample_idx} samples in: {cache_dir}")
    return sample_idx

class CachedHeadDataset(Dataset):
    """
    Dataset per la cache leggera.

    Ogni item restituisce:
        q_features: [Q, C]
        query_class_targets: [Q]
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("sample_*.pt"))

        if len(self.files) == 0:
            raise RuntimeError(f"No cached files found in {self.cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], map_location="cpu")

        q_features = item["q_features"]
        query_class_targets = item["query_class_targets"]

        return q_features, query_class_targets


def cached_head_collate_fn(batch):
    """
    Collate function per cache leggera.

    Output:
        q_features: [B, Q, C]
        query_class_targets: [B, Q]
    """

    q_features, query_class_targets = zip(*batch)

    q_features = torch.stack(q_features, dim=0)
    query_class_targets = torch.stack(query_class_targets, dim=0)

    return q_features, query_class_targets


# DA ELIMINARE SE IL CACHE- FINETUNING HEAD+ULTIMI LATERS NON FUNZIONA
# class CachedBackboneTokenDataset(Dataset):
#     def __init__(self, cache_dir):
#         self.cache_dir = Path(cache_dir)
#         self.files = sorted(self.cache_dir.glob("batch_*.pt"))

#         if len(self.files) == 0:
#             raise RuntimeError(f"No cached backbone-token files found in {self.cache_dir}")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         item = torch.load(self.files[idx], map_location="cpu")
#         return item["tokens"], item["targets"]

# NUOVA PERCHE' NON SALVO LE MASK NELLA CACHE
class CachedBackboneTokenWithTargetsDataset(Dataset):
    def __init__(self, cache_dir, original_dataset, original_batch_size):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("batch_*.pt"))
        self.original_dataset = original_dataset
        self.original_batch_size = original_batch_size

        if len(self.files) == 0:
            raise RuntimeError(f"No cached backbone-token files found in {self.cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], map_location="cpu")
        tokens = item["tokens"]

        start = idx * self.original_batch_size
        end = min(start + tokens.shape[0], len(self.original_dataset))

        targets = []
        for original_idx in range(start, end):
            _, target = self.original_dataset[original_idx]
            targets.append(
                {
                    "labels": target["labels"].cpu().long(),
                    "masks": target["masks"].to(torch.bool).cpu(),
                }
            )

        return tokens, targets

# DA ELIMINARE SE IL CACHE- FINETUNING HEAD+ULTIMI LATERS NON FUNZIONA
# def cached_backbone_token_collate_fn(batch):
#     tokens, targets = zip(*batch)

#     # Ogni file contiene già un batch: tokens ha shape [B, N, C].
#     tokens = torch.cat(tokens, dim=0)

#     # targets è una lista di liste: la appiattiamo.
#     flat_targets = []
#     for target_list in targets:
#         flat_targets.extend(target_list)

#     return tokens, flat_targets
def cached_backbone_token_with_targets_collate_fn(batch):
    tokens, targets = zip(*batch)

    tokens = torch.cat(tokens, dim=0)

    flat_targets = []
    for target_list in targets:
        flat_targets.extend(target_list)

    return tokens, flat_targets


