"""The published TorNet detector as PyTorch.

`TorNetEncoder` is the backbone: four blocks of CoordConv layers (a 3x3 convolution over the
features concatenated with the two coordinate channels), each block followed by 2x2 max pooling
of features and coordinates and a dropout; it returns the bottleneck map, 384 x 8 x 15 for a
standard chip. `TorNetHead` is the authors' max-pool head: 1x1 convolutions to a per-cell logit
map, max over cells. `TorNetDetector` is both, the published model.

The weights are a torch state dict on the Hub (`jakeryderv/tornet-detector-torch`), converted
once from the authors' Keras checkpoint and verified against its logits; see that model card.
"""

from __future__ import annotations

import os
from typing import cast

import torch
from torch import nn

from tornado_onset.data import N_INPUT_CHANNELS

WEIGHTS_REPO = "jakeryderv/tornet-detector-torch"
WEIGHTS_FILE = "tornet_detector_v1.pt"
LOCAL_WEIGHTS = f"weights/{WEIGHTS_FILE}"  # a local copy, if one exists, wins
BLOCK_CONVS = (2, 2, 3, 3)
COORD_CHANNELS = 2


class CoordConv(nn.Module):
    """3x3 convolution over [features, coordinates], ReLU; returns features only."""

    def __init__(self, in_channels: int, out_channels: int, coord_channels: int = COORD_CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + coord_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(torch.cat([x, coords], dim=1)))


class TorNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = N_INPUT_CHANNELS,
        start_filters: int = 48,
        block_convs: tuple[int, ...] = BLOCK_CONVS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = in_channels
        for b, n_convs in enumerate(block_convs):
            filters = start_filters * 2**b
            layers = nn.ModuleList()
            for _ in range(n_convs):
                layers.append(CoordConv(channels, filters))
                channels = filters
            self.blocks.append(layers)
        self.out_channels = channels
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Keras padding="same"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(B, 14, A, R), (B, 2, A, R) -> the bottleneck map (B, C, A/16, R/16)."""
        c = coords
        for layers in self.blocks:
            for layer in cast(nn.ModuleList, layers):
                x = layer(x, c)
            x = self.dropout(self.pool(x))
            c = self.pool(c)
        return x


class TorNetHead(nn.Module):
    """Two 1x1 convolutions with ReLU, a 1x1 map (one per output), max over cells. The map
    before the max is the paper's spatial likelihood map."""

    def __init__(
        self, in_channels: int, hidden: tuple[int, ...] = (512, 256), out_channels: int = 1
    ):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for units in hidden:
            layers += [nn.Conv2d(channels, units, 1), nn.ReLU()]
            channels = units
        self.hidden = nn.Sequential(*layers)
        self.heatmap = nn.Conv2d(channels, out_channels, 1)

    def heat(self, features: torch.Tensor) -> torch.Tensor:
        """Per-cell logits, (batch, out_channels, H, W)."""
        return self.heatmap(self.hidden(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.heat(features).amax(dim=(2, 3))  # (batch, out_channels)


class TorNetDetector(nn.Module):
    """Backbone plus head: the published tornado detector."""

    def __init__(self, start_filters: int = 48, in_channels: int = N_INPUT_CHANNELS):
        super().__init__()
        self.encoder = TorNetEncoder(in_channels=in_channels, start_filters=start_filters)
        self.head = TorNetHead(self.encoder.out_channels)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, coords))

    def heat(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.head.heat(self.encoder(x, coords))


def resolve_weights(path: str | None = None) -> str:
    """A local state-dict path if it exists, else the copy on the Hub (downloaded once)."""
    path = path or LOCAL_WEIGHTS
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE)


def load_state_dict(path: str | None = None) -> dict[str, torch.Tensor]:
    return torch.load(resolve_weights(path), map_location="cpu")


def load_detector(path: str | None = None, device: str | torch.device = "cpu") -> TorNetDetector:
    """The published detector, frozen."""
    sd = load_state_dict(path)
    model = TorNetDetector(start_filters=int(sd["encoder.blocks.0.0.conv.weight"].shape[0]))
    model.load_state_dict(sd)
    return model.eval().requires_grad_(False).to(device)
