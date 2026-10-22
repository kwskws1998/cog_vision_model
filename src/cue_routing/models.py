"""Shared cue readout for CNN maps and ViT tokens, plus the PACS ResNet-18."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class CueReadout(nn.Module):
    """Keep area-weighted cue integration identical to a global mean linear head."""

    def __init__(self, feature_dim, num_classes, grid_size=2):
        super().__init__()
        if (
            any(type(value) is not int for value in (feature_dim, num_classes, grid_size))
            or feature_dim <= 0
            or num_classes < 2
            or grid_size <= 0
        ):
            raise ValueError("Invalid readout dimensions")
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.grid_size = grid_size

    def spatial_map(self, features, token_grid=None):
        """Accept BCHW or BND patch tokens; CLS tokens must be removed explicitly."""
        if features.ndim == 4:
            if token_grid is not None:
                raise ValueError("token_grid is only valid for BND inputs")
            return features
        if features.ndim != 3 or token_grid is None:
            raise ValueError("ViT tokens require an explicit (height, width) token_grid")
        if (
            not isinstance(token_grid, (tuple, list))
            or len(token_grid) != 2
            or any(type(value) is not int or value <= 0 for value in token_grid)
        ):
            raise ValueError("token_grid must contain two positive integers")
        height, width = token_grid
        if height * width != features.shape[1]:
            raise ValueError("Token count does not match token_grid; exclude CLS tokens")
        return features.transpose(1, 2).reshape(features.shape[0], features.shape[2], height, width)

    def cue_features(self, features, token_grid=None):
        """Use disjoint integer regions, including unequal regions on odd-size maps."""
        features = self.spatial_map(features, token_grid)
        batch, channels, height, width = features.shape
        if (
            batch == 0
            or channels != self.classifier.in_features
            or min(height, width) < self.grid_size
        ):
            raise ValueError("Feature dimensions cannot form the requested nonempty cues")
        features = features.to(dtype=self.classifier.weight.dtype)
        vectors, areas = [], []
        for row in range(self.grid_size):
            y0, y1 = row * height // self.grid_size, (row + 1) * height // self.grid_size
            for column in range(self.grid_size):
                x0, x1 = column * width // self.grid_size, (column + 1) * width // self.grid_size
                vectors.append(features[:, :, y0:y1, x0:x1].mean(dim=(2, 3)))
                areas.append((y1 - y0) * (x1 - x0) / (height * width))
        return torch.stack(vectors, dim=1), features.new_tensor(areas)

    def cue_logits(self, features, token_grid=None):
        """Compute FP32 readout statistics even when the encoder uses CUDA AMP."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            vectors, weights = self.cue_features(features, token_grid)
            return self.classifier(vectors), weights

    def forward(self, features, token_grid=None):
        """Collapse inference to one global-mean linear readout."""
        features = self.spatial_map(features, token_grid)
        if (
            features.shape[0] == 0
            or features.shape[1] != self.classifier.in_features
            or min(features.shape[2:]) < 1
        ):
            raise ValueError("Invalid feature dimensions for global pooling")
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.classifier(features.to(self.classifier.weight.dtype).mean(dim=(2, 3)))


class ResNet18Classifier(nn.Module):
    """Expose the final feature map while sharing one classifier across all methods."""

    def __init__(
        self,
        num_classes=7,
        grid_size=2,
        pretrained=True,
        pretrained_path="",
        cache_dir=None,
        freeze_bn=True,
    ):
        super().__init__()
        if cache_dir is not None:
            torch.hub.set_dir(str(Path(cache_dir)))
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained and not pretrained_path else None
        network = resnet18(weights=weights)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            network.load_state_dict(state, strict=True)
        self.encoder = nn.Sequential(*list(network.children())[:-2])
        self.readout = CueReadout(512, num_classes, grid_size)
        self.freeze_bn = freeze_bn

    @property
    def classifier(self):
        """Expose the shared head for the reference RSC computation."""
        return self.readout.classifier

    def train(self, mode=True):
        """Freeze BatchNorm running statistics equally in all benchmark arms."""
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward_features(self, images):
        """Produce the final BCHW map before global average pooling."""
        return self.encoder(images)

    def forward(self, images):
        """Use the same unmasked prediction rule during all validation and testing."""
        return self.readout(self.forward_features(images))
