"""Run configuration: the defaults every run shares. The runs in `run.py` are a few overrides
each."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    exp_name: str = "onset"
    out_dir: str = "results"  # one directory per invocation of run.py; this run lives under it
    # data: one Hub config per year (the authors' train / test partitions); validation is held
    # out from the training partition by day
    train_years: list[int] = field(default_factory=lambda: [2014, 2015])
    test_years: list[int] = field(default_factory=lambda: [2014, 2015])
    val_fraction: float = 0.15
    n_in: int = 2  # observed scans per sample (the oldest n_in); the rest are targets
    horizons: int = 2
    # model
    weights: str | None = None  # detector state dict; default: local copy, else the Hub
    encoder_trainable: bool = True
    # loss: weighted cross-entropy per horizon, the authors' sample weights
    sample_weights: dict[str, float] = field(
        default_factory=lambda: {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5}
    )
    # training
    epochs: int = 5  # cap; early stopping usually ends sooner
    patience: int = 2  # stop after this many epochs without a validation head-AUC gain (0: off)
    batch_size: int = 16
    learning_rate: float = 1e-4
    encoder_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    # loader workers: each holds one decoded shard (about 1 GB); ONSET_WORKERS overrides, for a
    # machine with more cores and memory than the laptop default
    workers: int = field(default_factory=lambda: int(os.environ.get("ONSET_WORKERS", "4")))
    seed: int = 1234
    # evaluation
    bootstrap: int = 1000

    def __post_init__(self) -> None:
        if self.n_in + self.horizons > 4:
            raise ValueError("a TorNet sample has four scans: n_in + horizons must be <= 4")
