"""
Extract L2-normalized embeddings from trained CNN backbones.

CLI example:
python scripts/extract_embeddings.py --backbone resnet50 --split val --batch-size 128

Outputs:
- Per-image embeddings under embeddings/{backbone}/{split}/{category_id}/
- Metadata CSV/JSONL mapping image paths, categories, bbox (if available), and embedding files.

Embedding dims: resnet18=512, resnet50=2048, efficientnet_b0=1280.
These files feed directly into FAISS by loading the metadata CSV, stacking embedding tensors,
and building an index per split/backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.dataset import create_embedding_dataloader
from models.embedder import Embedder, EmbedderConfig, available_backbones, embedding_dim
from utils.common import get_device, load_yaml, set_seed


DEFAULT_DATA_ROOT = Path("data/processed/deepfashion2_shop_clean")
DEFAULT_SAVE_ROOT = Path("embeddings")
DEFAULT_CHECKPOINTS = {
    "resnet50": Path("models/resnet50_deepfashion2_shop_clean.pth"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract embeddings for retrieval.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Processed dataset root.")
    parser.add_argument("--backbone", type=str, choices=available_backbones(), default="resnet50")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path.")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop", type=str, default="center", choices=["center", "random_resized"])
    parser.add_argument("--save-dir", type=Path, default=None, help="Where to store embeddings.")
    parser.add_argument("--save-format", type=str, default="pt", choices=["pt", "npy"])
    parser.add_argument("--metadata-only", action="store_true", help="Skip embedding export and only write metadata.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config to override defaults.")
    return parser


def apply_config_overrides(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    cfg = load_yaml(args.config)
    if not cfg:
        return args

    for key, value in cfg.items():
        if not hasattr(args, key):
            continue
        default = parser.get_default(key)
        current = getattr(args, key)
        if current == default or current is None:
            setattr(args, key, value)
    return args


def resolve_checkpoint(backbone: str, checkpoint: Optional[Path]) -> Optional[Path]:
    if checkpoint:
        return Path(checkpoint)
    candidate = DEFAULT_CHECKPOINTS.get(backbone.lower())
    if candidate and candidate.exists():
        return candidate
    return None


def write_metadata(records: List[Dict[str, Any]], dest: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # JSONL sidecar for flexibility
    jsonl_path = dest.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = apply_config_overrides(args, parser)

    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)

    save_dir = args.save_dir or DEFAULT_SAVE_ROOT / args.backbone / args.split
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint(args.backbone, args.checkpoint)
    if ckpt_path and not ckpt_path.exists():
        print(f"[warn] Checkpoint not found at {ckpt_path}, proceeding with random/pretrained weights.")
        ckpt_path = None

    embedder = Embedder(
        EmbedderConfig(
            backbone=args.backbone,
            checkpoint=ckpt_path,
            device=device,
            normalize=True,
            pretrained=ckpt_path is None,  # use ImageNet weights when no checkpoint is given
        )
    )

    loader, dataset = create_embedding_dataloader(
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        crop=args.crop,
    )

    records: List[Dict[str, Any]] = []
    ext = ".pt" if args.save_format == "pt" else ".npy"

    if args.metadata_only:
        print("Skipping embedding export; writing metadata only.")

    with torch.no_grad():
        for images, metas in tqdm(loader, desc="Extracting embeddings", total=len(loader)):
            if args.metadata_only:
                embeddings = None
            else:
                embeddings = embedder.embed_batch(images)

            for idx, meta in enumerate(metas):
                img_path = Path(meta["image_path"])
                cat_id = int(meta["category_id"])
                emb_dir = save_dir / str(cat_id)
                emb_dir.mkdir(parents=True, exist_ok=True)

                if args.metadata_only:
                    emb_path = None
                else:
                    emb_path = emb_dir / f"{img_path.stem}{ext}"
                    if args.save_format == "pt":
                        torch.save(embeddings[idx].cpu(), emb_path)
                    else:
                        np.save(emb_path, embeddings[idx].cpu().numpy().astype(np.float32))

                bbox = meta.get("bbox") or {}
                records.append(
                    {
                        "split": args.split,
                        "image_path": str(img_path),
                        "category_id": cat_id,
                        "category_name": meta.get("category_name"),
                        "embedding_path": str(emb_path) if emb_path else "",
                        "bbox_x1": bbox.get("x1"),
                        "bbox_y1": bbox.get("y1"),
                        "bbox_x2": bbox.get("x2"),
                        "bbox_y2": bbox.get("y2"),
                        "backbone": args.backbone,
                        "embedding_dim": embedding_dim(args.backbone),
                    }
                )

    meta_path = save_dir / f"{args.split}_metadata.csv"
    write_metadata(records, meta_path)
    print(f"Saved {len(records)} records.")
    print(f"Metadata: {meta_path}")
    print(f"Embeddings dir: {save_dir}")


if __name__ == "__main__":
    main()
