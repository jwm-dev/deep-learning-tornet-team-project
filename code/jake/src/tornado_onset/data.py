"""TorNet samples from the Hugging Face Hub, turned into the detector's tensors.

The dataset `jakeryderv/tornet` holds every TorNet v1.1 sample as Parquet: four radar scans
about five minutes apart, each (azimuth=120, range=240, tilt=2) for six products in physical
units (NaN where empty), the range-folded mask, a label per scan, scan times and event metadata.
Shards are fetched once into the Hub file cache (`snapshot_download`, 256 samples and about
170 MB each) and read directly with pyarrow, one shard at a time: a shard decompresses to about
0.8 GB, so a worker holds one and `datasets`' uncompressed Arrow cache (3 MB per sample) is
never built. Training streams shards in random order and shuffles within each; a worker's
batches draw from the shards it holds, so mixing is at the shard level (one shard is a few
hours of one year). Row selection (day-based validation, a subset) is by filename.

The published detector consumes, per scan, the six products normalised to roughly [-1, 1] with
empty gates set to ``BACKGROUND``, the two range-folded channels appended (14 channels), and two
CoordConv coordinate channels (range and inverse range). Everything here is channels-first.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils.data import IterableDataset, get_worker_info

HF_DATASET = "jakeryderv/tornet"
ALL_VARIABLES = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]
CHANNEL_MIN_MAX = {
    "DBZ": (-20.0, 60.0),
    "VEL": (-60.0, 60.0),
    "KDP": (-2.0, 5.0),
    "RHOHV": (0.2, 1.04),
    "ZDR": (-1.0, 8.0),
    "WIDTH": (0.0, 9.0),
}
BACKGROUND = -3.0  # the detector's fill value for empty gates, in normalised units
N_SWEEPS = 2
N_RADAR_CHANNELS = N_SWEEPS * len(ALL_VARIABLES)  # 12
N_INPUT_CHANNELS = N_RADAR_CHANNELS + N_SWEEPS  # + 2 range-folded channels = 14


def channel_scale(name: str) -> tuple[float, float]:
    """(offset, scale) such that normalised = (raw - offset) / scale, as the detector does."""
    lo, hi = CHANNEL_MIN_MAX[name]
    return (lo + hi) / 2, (hi - lo) / 2


def normalize_fields(frames: dict[str, np.ndarray], background: float = BACKGROUND) -> np.ndarray:
    """{product: (..., A, R, tilt)} -> (..., A, R, 12), normalised and NaN-filled, in the
    detector's channel order (products in ALL_VARIABLES order, both tilts of each)."""
    out = []
    for v in ALL_VARIABLES:
        offset, scale = channel_scale(v)
        x = (frames[v].astype(np.float32) - offset) / scale
        out.append(np.where(np.isnan(x), background, x))
    return np.concatenate(out, axis=-1)


def coordinates(
    az_limits: tuple[float, float],
    rng_limits: tuple[float, float],
    shape: tuple[int, int] = (120, 240),
    min_range_m: float = 2125.0,
) -> np.ndarray:
    """The authors' CoordConv coordinates, (2, A, R): scaled range and its inverse."""
    scale = 1e-5
    rng_lower = (rng_limits[0] + 250) * scale
    rng_upper = (rng_limits[1] - 250) * scale
    az_lower = (90 - az_limits[0]) * np.pi / 180
    az_upper = (90 - az_limits[1]) * np.pi / 180
    az = np.linspace(az_lower, az_upper, shape[0])
    rg = np.linspace(rng_lower, rng_upper, shape[1])
    r, _ = np.meshgrid(rg, az, indexing="xy")  # (A, R)
    r = np.where(r >= min_range_m * scale, r, min_range_m * scale)
    return np.stack((r, 1 / r), axis=0).astype(np.float32)


@dataclass
class Sample:
    """One TorNet sample, every scan: fields (T, 14, A, R) in the detector's representation,
    coords (2, A, R), one label per scan, times in epoch seconds, and event metadata."""

    filename: str
    fields: np.ndarray
    coords: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    category: str
    ef_number: int
    event_id: int


def sample_from_row(row: dict[str, Any]) -> Sample:
    """A row of the Hub dataset (numpy format) -> the detector's tensors."""
    frames = {v: np.asarray(row[v], dtype=np.float32) for v in ALL_VARIABLES}  # (T, A, R, tilt)
    rfm = np.asarray(row["range_folded_mask"], dtype=np.float32)
    fields = np.concatenate([normalize_fields(frames), rfm], axis=-1)  # (T, A, R, 14)
    ef = int(row["ef_number"])
    labels = np.asarray(row["frame_labels"], dtype=np.float32).copy()
    labels[-1] = float(ef >= 0)  # the authors' v1.0 consistency fix
    az = (float(row["az_lower"]), float(row["az_upper"]))
    rng = (float(row["rng_lower"]), float(row["rng_upper"]))
    return Sample(
        filename=str(row["filename"]),
        fields=np.ascontiguousarray(fields.transpose(0, 3, 1, 2)),
        coords=coordinates(az, rng, fields.shape[1:3]),
        labels=labels,
        times=np.asarray(row["time"], dtype=np.int64),
        category=str(row["category"]),
        ef_number=ef,
        event_id=int(row["event_id"]),
    )


def sample_weight(category: str, ef_number: int, w: dict[str, float]) -> float:
    """The authors' rule: warnings wW, nulls wN, EF0 w0, EF1 w1, EF2+ w2."""
    if category == "WRN":
        return float(w.get("wW", 0.5))
    if ef_number < 0:
        return float(w.get("wN", 1.0))
    if ef_number == 0:
        return float(w.get("w0", 1.0))
    if ef_number == 1:
        return float(w.get("w1", 1.0))
    return float(w.get("w2", 2.0))


ARRAY_FIELDS = [*ALL_VARIABLES, "range_folded_mask"]
SHAPE = (4, 120, 240, 2)


def shard_paths(partition: str, years: list[int]) -> list[str]:
    """Local paths of the Parquet shards of one partition and the given years, fetched into the
    Hub cache on first use, sorted."""
    import glob

    from huggingface_hub import snapshot_download

    allow = [f"data/{partition}/{y}/*.parquet" for y in years]
    root = snapshot_download(HF_DATASET, repo_type="dataset", allow_patterns=allow)
    paths = sorted(p for pat in allow for p in glob.glob(os.path.join(root, pat)))
    if not paths:
        raise FileNotFoundError(f"no shards for {partition} {years} in {HF_DATASET}")
    return paths


def metadata(partition: str, years: list[int]) -> pd.DataFrame:
    """The scalar fields of every sample of the partition-years (from the small per-year
    metadata files), with a `day` column (UTC date of the first scan)."""
    from huggingface_hub import snapshot_download

    allow = [f"metadata/{partition}-{y}.parquet" for y in years]
    root = snapshot_download(HF_DATASET, repo_type="dataset", allow_patterns=allow)
    df = pd.concat([pd.read_parquet(os.path.join(root, a)) for a in allow], ignore_index=True)
    first = np.array([t[0] for t in df["time"]], dtype="int64")
    df["day"] = np.datetime_as_string(first.astype("datetime64[s]"), unit="D")
    return df


def split_by_day(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Hold out whole days for validation so consecutive scans of one storm never straddle the
    split. Returns (train filenames, val filenames)."""
    days = np.array(sorted(df["day"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(days)
    val_days = set(days[: round(len(days) * val_fraction)])
    is_val = df["day"].isin(val_days).to_numpy()
    return set(df["filename"][~is_val]), set(df["filename"][is_val])


def _flat(column) -> np.ndarray:
    """A nested list<...<halffloat|uint8>> Arrow column -> an owned (N, *SHAPE) numpy array."""
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    n = len(arr)
    if isinstance(arr, pa.ExtensionArray):  # `datasets` registered its Array4D type
        arr = arr.storage
    while isinstance(arr, pa.ListArray | pa.FixedSizeListArray):
        arr = arr.flatten()
    return np.array(arr.to_numpy(zero_copy_only=False), copy=True).reshape(n, *SHAPE)


def read_shard(path: str, columns: list[str] | None = None) -> dict[str, Any]:
    """One shard -> {"filename": [...], scalar columns: lists, array columns: (N, *SHAPE)}.
    Columns are read one at a time and copied out of Arrow, with the system allocator so the
    memory goes back to the OS: a worker holds one decoded shard (~0.8 GB), not an Arrow table
    on top of it."""
    import pyarrow.parquet as pq

    pa.set_memory_pool(pa.system_memory_pool())
    f = pq.ParquetFile(path)
    names = columns or f.schema_arrow.names
    out: dict[str, Any] = {}
    for name in names:
        col = f.read(columns=[name]).column(0)
        out[name] = _flat(col) if name in ARRAY_FIELDS else col.to_pylist()
        del col
    f.close()
    pa.default_memory_pool().release_unused()
    return out


def row_of(shard: dict[str, Any], i: int) -> dict[str, Any]:
    return {k: v[i] for k, v in shard.items()}


class OnsetDataset(IterableDataset):
    """Samples streamed from Parquet shards: the oldest `n_in` scans in, the next `horizons`
    out. With `shuffle`, shard order is reshuffled every epoch (`set_epoch`) and rows within a
    shard are shuffled; DataLoader workers take disjoint shards. Shards holding none of the
    `keep` filenames are skipped after reading only their filename column, so a day-based
    validation split (days are contiguous in the shards) decodes a fraction of the shards.

    Per item:
        x            (n_in, 14, A, R)      the observed scans
        coords       (2, A, R)
        future       (horizons, 14, A, R)  target scans
        labels       (horizons,)           tornado label of each target scan
        frame_labels (n_in + horizons,)    every scan's label, for scoping the evaluation
        minutes      (horizons,)           lead from the last input scan, real timestamps
        weight       ()                    the authors' sample weight
        meta         filename, category, ef_number, event_id, time (last input), times
    """

    def __init__(
        self,
        paths: list[str],
        n_in: int = 2,
        horizons: int = 2,
        weights: dict[str, float] | None = None,
        keep: set[str] | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if n_in + horizons > 4:
            raise ValueError("a TorNet sample has four scans")
        self.paths = list(paths)
        self.n_in, self.horizons = n_in, horizons
        self.weights = weights or {}
        self.keep = keep
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _my_paths(self) -> list[str]:
        paths = self.paths
        if self.shuffle:
            order = np.random.default_rng(self.seed + self.epoch).permutation(len(paths))
            paths = [paths[i] for i in order]
        info = get_worker_info()
        if info is not None:
            paths = paths[info.id :: info.num_workers]
        return paths

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        rng = np.random.default_rng(self.seed + 1000 * self.epoch + (info.id if info else 0))
        n, h = self.n_in, self.horizons
        for path in self._my_paths():
            if self.keep is not None:  # the filename column alone is cheap: skip shards with
                names = read_shard(path, ["filename"])["filename"]  # nothing to yield
                idx = [i for i, f in enumerate(names) if f in self.keep]
                if not idx:
                    continue
            shard = read_shard(path)
            if self.keep is None:
                idx = list(range(len(shard["filename"])))
            if self.shuffle:
                rng.shuffle(idx)
            for i in idx:
                s = sample_from_row(row_of(shard, i))
                t = s.times
                yield {
                    "x": torch.from_numpy(s.fields[:n]),
                    "coords": torch.from_numpy(s.coords),
                    "future": torch.from_numpy(s.fields[n : n + h]),
                    "labels": torch.from_numpy(s.labels[n : n + h]),
                    "frame_labels": torch.from_numpy(s.labels[: n + h]),
                    "minutes": torch.from_numpy(
                        ((t[n : n + h] - t[n - 1]) / 60).astype(np.float32)
                    ),
                    "weight": torch.tensor(sample_weight(s.category, s.ef_number, self.weights)),
                    "meta": {
                        "filename": s.filename,
                        "category": s.category,
                        "ef_number": s.ef_number,
                        "event_id": s.event_id,
                        "time": int(t[n - 1]),
                        "times": t[n : n + h].tolist(),
                    },
                }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "meta"}
    out["meta"] = [b["meta"] for b in batch]
    return out
