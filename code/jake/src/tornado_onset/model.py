"""The onset model: the published TorNet encoder on the observed scans, the authors' head on top.

    the two observed scans (14 channels each, polar)
            |
    TorNet encoder, run per scan -> one 384 x 8 x 15 feature map per scan
            |
    maps concatenated along channels (768)
            |
    the authors' head: 1x1 convolutions to a per-cell logit map, max over cells
            -> P(tornado at +5), P(tornado at +10)

One loss: weighted cross-entropy on the per-horizon logits against the labels of the future
scans. The encoder starts from the published weights and is frozen or fine-tuned per run; the
head is always new. The frozen published detector (the judge) is not part of training; it sets
the floor (judge on the newest observed scan) and the ceiling (judge on the real future scan).
"""

from __future__ import annotations

import torch
from torch import nn

from tornado_onset.config import Config
from tornado_onset.tornet import TorNetEncoder, TorNetHead, load_state_dict


class OnsetModel(nn.Module):
    def __init__(self, n_in: int = 2, horizons: int = 2, start_filters: int = 48):
        super().__init__()
        self.encoder = TorNetEncoder(start_filters=start_filters)
        self.head = TorNetHead(self.encoder.out_channels * n_in, out_channels=horizons)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(B, T, 14, A, R), (B, 2, A, R) -> per-horizon logits (B, horizons)."""
        b, n = x.shape[:2]
        f = self.encoder(x.flatten(0, 1), coords.repeat_interleave(n, dim=0))  # (B*T, C, h, w)
        return self.head(f.reshape(b, n * f.shape[1], *f.shape[2:]))


def build_model(cfg: Config) -> OnsetModel:
    """The model with its encoder initialised from the published detector's weights."""
    sd = load_state_dict(cfg.weights)
    start_filters = int(sd["encoder.blocks.0.0.conv.weight"].shape[0])
    model = OnsetModel(cfg.n_in, cfg.horizons, start_filters)
    enc = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc, strict=True)
    if not cfg.encoder_trainable:
        model.encoder.requires_grad_(False)
    return model


def onset_loss(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Sample-weighted binary cross-entropy, mean over samples and horizons."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (bce * weight[:, None]).mean()
