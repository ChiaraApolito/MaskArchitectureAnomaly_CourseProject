from pathlib import Path
import shutil
import torch
from torch.utils.data import Dataset


def move_target_to_cpu(target):
    return {
        k: v.cpu() if torch.is_tensor(v) else v
        for k, v in target.items()
    }


def precompute_head_cache(
    model,
    train_dataloader,
    cache_dir,
    device=None,
    max_batches=None,
    overwrite=False,
):
    """
    Precompute della cache per head-only fine-tuning.

    Salva per ogni immagine:
        - q_features: input della class_head
        - mask_logits: maschere frozen usate dal matcher/loss
        - target: annotazioni
    """

    cache_dir = Path(cache_dir)

    if overwrite and cache_dir.exists():
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(cache_dir.glob("sample_*.pt"))
    if existing_files and not overwrite:
        print(
            f"[cache] Found {len(existing_files)} cached samples in {cache_dir}. "
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

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                imgs, targets = batch
                imgs = imgs.to(device, non_blocking=True).float()

                # LightningModule.forward fa imgs / 255.0.
                # Qui chiamiamo direttamente model.network, quindi dividiamo qui.
                x = imgs / 255.0

                q_features, mask_logits = model.network.forward_frozen_features(x)

                for b in range(q_features.shape[0]):
                    item = {
                        "q_features": q_features[b].cpu(),
                        "mask_logits": mask_logits[b].cpu(),
                        "target": move_target_to_cpu(targets[b]),
                    }

                    torch.save(item, cache_dir / f"sample_{sample_idx:06d}.pt")
                    sample_idx += 1

                if batch_idx % 20 == 0:
                    print(
                        f"[cache] batch {batch_idx} processed "
                        f"| saved samples: {sample_idx}"
                    )

    finally:
        # Ripristina training/eval state
        model.train(old_training_state)
        model.network.train(old_network_training_state)

        # Ripristina requires_grad, così la class_head resta allenabile dopo la cache
        for name, p in model.named_parameters():
            if name in old_requires_grad:
                p.requires_grad_(old_requires_grad[name])

    print(f"[cache] completed. Saved {sample_idx} samples in: {cache_dir}")
    return sample_idx


class CachedHeadDataset(Dataset):
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
        mask_logits = item["mask_logits"]
        target = item["target"]

        return q_features, mask_logits, target


def cached_head_collate_fn(batch):
    q_features, mask_logits, targets = zip(*batch)

    q_features = torch.stack(q_features, dim=0)
    mask_logits = torch.stack(mask_logits, dim=0)
    targets = list(targets)

    return q_features, mask_logits, targets