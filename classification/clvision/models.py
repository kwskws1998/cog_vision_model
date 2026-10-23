"""Selectable small-image ResNet-18 backbones and the two-stage spatial Zipf readout."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from cue_routing.io import state_hash
from cue_routing.models import CueReadout

from .architecture import ACTIVATIONS


class ZipfPreActBlock(nn.Module):
    """Match pinned Zipf PreActBlock, including its preactivated identity shortcut."""

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False)
            )

    def forward(self, values):
        """Apply BN/ReLU before convolutions and add the preactivated shortcut without ReLU."""
        activated = self.relu(self.bn1(values))
        shortcut = self.shortcut(activated)
        values = self.conv1(activated)
        values = self.conv2(self.relu(self.bn2(values)))
        return values + shortcut


class _ZipfPreActResNet18(nn.Module):
    """Match the official CIFAR stem, stage order, and default parameter initialization."""

    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        self.layer1 = nn.Sequential(ZipfPreActBlock(64, 64), ZipfPreActBlock(64, 64))
        self.layer2 = nn.Sequential(ZipfPreActBlock(64, 128, 2), ZipfPreActBlock(128, 128))
        self.layer3 = nn.Sequential(ZipfPreActBlock(128, 256, 2), ZipfPreActBlock(256, 256))
        self.layer4 = nn.Sequential(ZipfPreActBlock(256, 512, 2), ZipfPreActBlock(512, 512))
        self.fc = nn.Linear(512, classes)


class SmallResNet18(nn.Module):
    """Share the backbone and inference head; only Zipf adds a train-time auxiliary head."""

    def __init__(
        self, classes, auxiliary=False, seed=0, activation="postact", initialization="reference"
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError("activation must be 'postact' or 'preact'")
        self.activation = activation
        if activation == "postact":
            network = resnet18(weights=None, num_classes=classes)
            network.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(network.conv1.weight, mode="fan_out", nonlinearity="relu")
            network.maxpool = nn.Identity()
        else:
            network = _ZipfPreActResNet18(classes)
        if initialization == "msra":
            for layer in network.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        elif initialization != "reference":
            raise ValueError("Unknown initialization")
        self.backbone = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        self.readout = CueReadout(512, classes, grid_size=2)
        self.readout.classifier = network.fc
        self.auxiliary = None
        if auxiliary:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator().manual_seed(seed + 32017).get_state())
                self.auxiliary = nn.Linear(256, classes)

    def feature_maps(self, images):
        """Expose the penultimate and final spatial maps without extra encoder passes."""
        values = images
        penultimate = None
        for index, layer in enumerate(self.backbone):
            values = layer(values)
            if index == 6:
                penultimate = values
        return penultimate, values

    def shared_state_hash(self):
        """Compare exactly the tensors shared by every method, excluding auxiliary parameters."""
        return state_hash(
            {
                key: value
                for key, value in self.state_dict().items()
                if not key.startswith("auxiliary.")
            }
        )

    def forward(self, images):
        """Use identical global-average final-head predictions for all inference."""
        _, final = self.feature_maps(images)
        return self.readout(final)

    def zipf_predictions(self, penultimate, final):
        """Retain dense2 votes and auxiliary CE; bilinear upsampling matches the reference."""
        if self.auxiliary is None:
            raise ValueError("Zipf dense2 requires its auxiliary classifier")
        with torch.autocast(device_type=final.device.type, enabled=False):
            global_logits = self.readout.classifier(final.float().mean((2, 3)))
            auxiliary_logits = self.auxiliary(penultimate.float().mean((2, 3)))
            votes = []
            with torch.no_grad():
                for features, classifier in (
                    (final, self.readout.classifier),
                    (penultimate, self.auxiliary),
                ):
                    enlarged = F.interpolate(
                        features.float(), scale_factor=2.0, mode="bilinear", align_corners=False
                    )
                    local_logits = classifier(enlarged.permute(0, 2, 3, 1))
                    votes.append(local_logits.argmax(-1).flatten(1))
            return global_logits, auxiliary_logits, torch.cat(votes, dim=1)
