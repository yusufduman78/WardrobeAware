"""Utility to report per-class accuracy for a saved checkpoint."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

# Resolve project root (scripts/ -> repo root)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_loader import DeepFashionDataset, get_data_transforms
from train import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-class accuracy for a checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the checkpoint (.pth). Defaults to curated/best paths if omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (use 0 on Windows)",
    )
    return parser.parse_args()


def resolve_checkpoint(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit
    candidates = [
        project_root / "models" / "resnet50_deepfashion2_shop_clean.pth",
        project_root / "models" / "resnet50_deepfashion2_balanced.pth",
        project_root / "experiments" / "training_balance_off" / "best_model.pth",
        project_root / "experiments" / "training" / "best_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No checkpoint found in default locations. Use --checkpoint.")


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")

    dataset_root = project_root / "data" / "processed" / "deepfashion2_shop_clean"
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_root}. Run preparing.py first."
        )

    _, val_tf = get_data_transforms()
    val_dataset = DeepFashionDataset(
        str(dataset_root), split="val", transform=val_tf, load_metadata=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = len(val_dataset.classes)
    model = create_model("resnet50", num_classes, pretrained=False)
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1)
            for label, pred in zip(labels, preds):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    print("\nPer-class accuracy:")
    for idx, cls in enumerate(val_dataset.classes):
        total = int(class_total[idx].item())
        correct = int(class_correct[idx].item())
        acc = (correct / total) if total else 0.0
        print(f"  Class {cls}: {correct}/{total} ({acc*100:.2f}%)")

    overall_acc = class_correct.sum().item() / class_total.sum().item()
    macro_acc = (class_correct / class_total.clamp(min=1)).mean().item()
    print(f"\nOverall accuracy: {overall_acc*100:.2f}%")
    print(f"Macro (per-class) accuracy: {macro_acc*100:.2f}%")


if __name__ == "__main__":
    main()
