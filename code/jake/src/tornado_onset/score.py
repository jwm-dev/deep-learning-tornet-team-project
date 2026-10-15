"""Evaluation on the test partition, per horizon, and the metrics behind it.

Step zero, `score_detector`, runs the frozen published detector as designed, one scan at a time,
over the test set and writes two probabilities per sample and horizon: the detector on the
newest observed scan (persistence, the floor) and on the real future scan (truth, the ceiling).
That directory is made once and shared. `evaluate` scores a trained model: it writes the head's
probabilities and reads the floor and ceiling back from the detector directory.

Every predictions CSV carries the per-scan label pattern (e.g. `0011`) and `pre_onset` /
`ongoing` flags. `summary.json` has AUC / AUCPR per view (the paper's NC, RC, WC) for all cases
and for the pre-onset subset (no input scan tornadic, last target scan tornadic; nulls kept),
with bootstrap intervals and paired bootstraps against persistence.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tornado_onset.config import Config
from tornado_onset.data import OnsetDataset, collate, shard_paths
from tornado_onset.tornet import TorNetDetector

# which null categories are kept alongside confirmed tornadoes
VIEWS = {
    "NC": ("all nulls vs confirmed", {"TOR", "NUL", "WRN"}),
    "RC": ("random nulls vs confirmed", {"TOR", "NUL"}),
    "WC": ("warnings vs confirmed", {"TOR", "WRN"}),
}


# --- metrics ---------------------------------------------------------------------------------


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; nan if only one class is present."""
    y = np.asarray(y, bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def _ci(values: list[float]) -> list[float]:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def views(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """AUC and AUCPR with bootstrap intervals, per view that has both classes."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, (_desc, cats) in VIEWS.items():
        sub = df[df["category"].isin(cats)]
        y, p = sub["label"].to_numpy(bool), sub["prob"].to_numpy(float)
        if not 0 < y.sum() < len(y):
            continue
        entry: dict = {
            "n": len(sub),
            "n_confirmed": int(y.sum()),
            "auc": roc_auc(y, p),
            "aucpr": average_precision(y, p),
        }
        if n_boot:
            idx = rng.integers(0, len(sub), (n_boot, len(sub)))
            entry["auc_ci95"] = _ci([roc_auc(y[i], p[i]) for i in idx])
            entry["aucpr_ci95"] = _ci([average_precision(y[i], p[i]) for i in idx])
        out[key] = entry
    return out


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Bootstrap the AUC and AUCPR difference (b - a) over the same files, per view."""
    m = a.merge(b, on="filename", suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    out: dict = {"n_common": len(m)}
    for key, (_desc, cats) in VIEWS.items():
        sub = m[m["category_a"].isin(cats)]
        y = sub["label_a"].to_numpy(bool)
        pa, pb = sub["prob_a"].to_numpy(float), sub["prob_b"].to_numpy(float)
        idx = rng.integers(0, len(sub), (n_boot, len(sub)))
        out[key] = {
            "delta_auc": roc_auc(y, pb) - roc_auc(y, pa),
            "delta_auc_ci95": _ci([roc_auc(y[i], pb[i]) - roc_auc(y[i], pa[i]) for i in idx]),
            "delta_aucpr": average_precision(y, pb) - average_precision(y, pa),
            "delta_aucpr_ci95": _ci(
                [average_precision(y[i], pb[i]) - average_precision(y[i], pa[i]) for i in idx]
            ),
        }
    return out


# --- predictions -----------------------------------------------------------------------------


def pattern_flags(frame_labels: np.ndarray, n_in: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """pre_onset: no input scan tornadic and the last scan tornadic; ongoing: an input scan
    tornadic; and the label pattern as a string."""
    inputs = frame_labels[:, :n_in].max(axis=1) > 0
    last = frame_labels[:, -1] > 0
    patterns = ["".join(str(int(v)) for v in row) for row in frame_labels]
    return (~inputs) & last, inputs, patterns


def test_loader(cfg: Config) -> DataLoader:
    paths = shard_paths("test", cfg.test_years)
    return DataLoader(
        OnsetDataset(paths, cfg.n_in, cfg.horizons, cfg.sample_weights),
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict(
    cfg: Config,
    device: torch.device,
    sources: Callable[[dict, torch.Tensor, torch.Tensor, int], dict[str, np.ndarray]],
) -> dict[str, list[pd.DataFrame]]:
    """Run over the test set; `sources(batch, x, coords, h)` returns {name: logits (B,)} for
    horizon h. Returns {name: [one predictions frame per horizon]}."""
    rows: dict[str, list[list[dict]]] = {}
    for batch in test_loader(cfg):
        x, coords = batch["x"].to(device), batch["coords"].to(device)
        pre_onset, ongoing, patterns = pattern_flags(batch["frame_labels"].numpy(), cfg.n_in)
        for h in range(cfg.horizons):
            logits = sources(batch, x, coords, h)
            for name in logits:
                rows.setdefault(name, [[] for _ in range(cfg.horizons)])
            for j, meta in enumerate(batch["meta"]):
                base = {
                    "filename": meta["filename"],
                    "label": int(batch["labels"][j, h]),
                    "category": meta["category"],
                    "ef_number": meta["ef_number"],
                    "event_id": meta["event_id"],
                    "time": int(meta["times"][h]),
                    "minutes": float(batch["minutes"][j, h]),
                    "pattern": patterns[j],
                    "pre_onset": bool(pre_onset[j]),
                    "ongoing": bool(ongoing[j]),
                }
                for name, lg in logits.items():
                    v = float(lg[j])
                    rows[name][h].append({**base, "logit": v, "prob": 1 / (1 + math.exp(-v))})
    return {name: [pd.DataFrame(r) for r in per_h] for name, per_h in rows.items()}


def scoped(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-onset evaluation set: pre-onset tornado cases plus every null."""
    return df[df["pre_onset"] | (df["category"] != "TOR")]


def summarize(
    cfg: Config, preds: dict[str, list[pd.DataFrame]], out_dir: str, against: str = "persistence"
) -> dict:
    """Write each source's predictions CSVs and summary.json: per horizon, views over all cases
    and over the pre-onset set, and every other source paired against `against`."""
    os.makedirs(out_dir, exist_ok=True)
    summary: dict = {"test_years": cfg.test_years, "n_in": cfg.n_in, "horizons": []}
    for h in range(cfg.horizons):
        frames = {name: per_h[h] for name, per_h in preds.items()}
        for name, df in frames.items():
            df.to_csv(os.path.join(out_dir, f"predictions_{name}_h{h}.csv"), index=False)
        ref = next(iter(frames.values()))
        pre = {name: scoped(df) for name, df in frames.items()}
        entry = {
            "horizon": h,
            "lead_minutes_median": float(ref["minutes"].median()),
            "n": len(ref),
            "positives": int(ref["label"].sum()),
            "pre_onset_cases": int(ref["pre_onset"].sum()),
            "ongoing_cases": int(ref["ongoing"].sum()),
            "all": {name: views(df, cfg.bootstrap, cfg.seed) for name, df in frames.items()},
            "pre_onset": {name: views(df, cfg.bootstrap, cfg.seed) for name, df in pre.items()},
            f"pre_onset_paired_vs_{against}": {
                name: paired_difference(pre[against], df, cfg.bootstrap, cfg.seed)
                for name, df in pre.items()
                if name != against
            },
        }
        summary["horizons"].append(entry)
        line = " ".join(
            f"{name}={entry['pre_onset'].get(name, {}).get('WC', {}).get('auc', float('nan')):.4f}"
            for name in frames
        )
        logging.info(
            "h%d lead %.0f min, pre-onset WC AUC: %s", h, entry["lead_minutes_median"], line
        )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def score_detector(cfg: Config, judge: TorNetDetector, device: torch.device, out_dir: str) -> dict:
    """Step zero: the published detector on the newest observed scan (persistence) and on each
    real future scan (truth), written once for every later run to compare against."""

    def sources(batch: dict, x: torch.Tensor, coords: torch.Tensor, h: int) -> dict:
        future = batch["future"][:, h].to(device)
        return {
            "persistence": judge(x[:, -1], coords).squeeze(1).cpu().numpy(),
            "truth": judge(future, coords).squeeze(1).cpu().numpy(),
        }

    return summarize(cfg, predict(cfg, device, sources), out_dir)


def read_detector(detector_dir: str, horizons: int) -> dict[str, list[pd.DataFrame]]:
    return {
        name: [
            pd.read_csv(os.path.join(detector_dir, f"predictions_{name}_h{h}.csv"))
            for h in range(horizons)
        ]
        for name in ("persistence", "truth")
    }


@torch.no_grad()
def evaluate(
    cfg: Config, model: torch.nn.Module, device: torch.device, out_dir: str, detector_dir: str
) -> dict:
    """Score a trained model's head and compare it with the cached floor and ceiling."""
    model.eval()

    def sources(batch: dict, x: torch.Tensor, coords: torch.Tensor, h: int) -> dict:
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = model(x, coords)
        return {"head": logits.float()[:, h].cpu().numpy()}

    preds = {**read_detector(detector_dir, cfg.horizons), **predict(cfg, device, sources)}
    for h in range(cfg.horizons):
        if not preds["head"][h]["filename"].equals(preds["persistence"][h]["filename"]):
            raise RuntimeError(f"{detector_dir} was scored on a different test set")
    return summarize(cfg, preds, out_dir)


def headline(summary: dict) -> dict:
    """Pre-onset WC AUC per source and horizon, for results.json."""
    return {
        "pre_onset_WC_AUC": {
            f"h{e['horizon']}": {
                name: v.get("WC", {}).get("auc") for name, v in e["pre_onset"].items()
            }
            for e in summary["horizons"]
        }
    }
