"""Embedding utilities for pretrained CNN backbones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet, EfficientNet


BACKBONE_OUTPUT_DIM = {
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
}


def _strip_classifier(model: nn.Module, backbone: str) -> nn.Module:
    """Remove the classification head and return a feature extractor."""
    if backbone in {"resnet18", "resnet50"}:
        assert isinstance(model, ResNet)
        model.fc = nn.Identity()
    elif backbone == "efficientnet_b0":
        assert isinstance(model, EfficientNet)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def _load_backbone(backbone: str, pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return _strip_classifier(model, backbone)


@dataclass
class EmbedderConfig:
    backbone: str = "resnet50"
    checkpoint: Optional[Path] = None
    device: Optional[torch.device] = None
    normalize: bool = True
    pretrained: bool = True


class Embedder(nn.Module):
    """Wraps a CNN backbone to output L2-normalized embeddings."""

    def __init__(self, config: EmbedderConfig):
        super().__init__()
        self.backbone_name = config.backbone.lower()
        self.normalize = config.normalize
        self.output_dim = BACKBONE_OUTPUT_DIM[self.backbone_name]
        self.backbone = _load_backbone(self.backbone_name, pretrained=config.pretrained)

        if config.checkpoint:
            state = torch.load(config.checkpoint, map_location="cpu")
            state_dict = state.get("model_state_dict", state)
            target_state = self.backbone.state_dict()
            matched = [
                k for k, v in state_dict.items()
                if k in target_state and target_state[k].shape == v.shape
            ]
            matched_ratio = len(matched) / max(len(target_state), 1)
            if matched_ratio < 0.5:
                raise ValueError(
                    f"Checkpoint '{config.checkpoint}' appears to be a different architecture "
                    f"({len(matched)} of {len(target_state)} keys match). "
                    f"Check backbone name or checkpoint path."
                )

            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[Embedder] Loaded with missing keys: {missing}, unexpected keys: {unexpected}")

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x.to(self.device))
        if self.normalize:
            feats = nn.functional.normalize(feats, p=2, dim=1)
        return feats

    def embed_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for forward for readability."""
        return self.forward(images)


def embedding_dim(backbone: str) -> int:
    return BACKBONE_OUTPUT_DIM[backbone.lower()]


def available_backbones() -> Tuple[str, ...]:
    return tuple(BACKBONE_OUTPUT_DIM.keys())
