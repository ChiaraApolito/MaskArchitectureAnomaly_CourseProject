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
    Creation of a light cache for class head-only fine-tuning.

    Save for each image:
     - q_features: class_head input, shape [Q, C]
     - query_class_targets: class targets per query, shape [Q]
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

                x = imgs / 255.0

                # forward frozen
                q_features, mask_logits = model.network.forward_frozen_features(x)

                # initial class logits (before matching) for target assignment
                class_logits = model.network.class_head(q_features)

                mask_labels = [
                    target["masks"].to(device).to(mask_logits.dtype)
                    for target in targets
                ]
                class_labels = [
                    target["labels"].to(device).long()
                    for target in targets
                ]

                # Hungarian Matching between queries and GT objects using the current head outputs
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


class CachedHeadDataset(Dataset):
    """
    Dataset of pre-extracted features for avoiding to re-run the entire model during training.

    For each sample returns:
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
    Collate function for light-cache.

    Output:
        q_features: [B, Q, C]
        query_class_targets: [B, Q]
    """

    q_features, query_class_targets = zip(*batch)

    q_features = torch.stack(q_features, dim=0)
    query_class_targets = torch.stack(query_class_targets, dim=0)

    return q_features, query_class_targets



