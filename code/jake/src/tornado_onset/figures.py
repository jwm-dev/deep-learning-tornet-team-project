"""The summary and the figures of one results directory, built from its runs' files alone.

    results/<stamp>/
      detector/          floor and ceiling predictions, summary.json          (step 0)
      <run>/             params.json, history.csv, results.json, scores/      (steps 1..)
      summary.csv        one row per run and horizon
      summary.txt        the same as a table
      figures/           ladder.png, roc_pr.png, training.png, delta.png

`summarize_root` writes all of it and is rerunnable: `uv run run.py summary results/<stamp>`.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

SOURCE_LABEL = {
    "persistence": "floor: detector on the newest scan",
    "truth": "ceiling: detector on the future scan",
}


def _load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def runs_in(root: str) -> list[str]:
    """Trained run directories under root, in ladder order (params.json records the order of
    creation through the directory mtime; names are stable so sort by that)."""
    out = []
    for d in os.listdir(root):
        if d != "detector" and os.path.exists(os.path.join(root, d, "scores", "summary.json")):
            out.append(d)
    return sorted(out, key=lambda d: os.path.getmtime(os.path.join(root, d, "params.json")))


def summary_rows(root: str) -> pd.DataFrame:
    """One row per run and horizon: pre-onset WC AUC of floor, ceiling and head with intervals,
    the head's paired gain over the floor, epochs; then the same per test year (from the
    predictions files, point estimates only) with `year` set."""
    det = _load(os.path.join(root, "detector", "summary.json"))
    if det is None:
        raise FileNotFoundError(f"no detector step under {root}")
    rows = []

    def wc(entry: dict, name: str, key: str = "auc"):
        return entry["pre_onset"].get(name, {}).get("WC", {}).get(key)

    for e in det["horizons"]:
        rows.append(
            {
                "step": 0,
                "run": "detector",
                "horizon": e["horizon"],
                "lead_min": e["lead_minutes_median"],
                "pre_onset_cases": e["pre_onset_cases"],
                "floor_auc": wc(e, "persistence"),
                "floor_ci": wc(e, "persistence", "auc_ci95"),
                "ceiling_auc": wc(e, "truth"),
                "ceiling_ci": wc(e, "truth", "auc_ci95"),
                "ceiling_minus_floor": e["pre_onset_paired_vs_persistence"]["truth"]["WC"][
                    "delta_auc"
                ],
            }
        )
    for step, d in enumerate(runs_in(root), start=1):
        summary = _load(os.path.join(root, d, "scores", "summary.json")) or {}
        res = _load(os.path.join(root, d, "results.json")) or {}
        for e in summary["horizons"]:
            paired = e["pre_onset_paired_vs_persistence"]["head"]["WC"]
            rows.append(
                {
                    "step": step,
                    "run": d,
                    "horizon": e["horizon"],
                    "lead_min": e["lead_minutes_median"],
                    "pre_onset_cases": e["pre_onset_cases"],
                    "floor_auc": wc(e, "persistence"),
                    "ceiling_auc": wc(e, "truth"),
                    "head_auc": wc(e, "head"),
                    "head_ci": wc(e, "head", "auc_ci95"),
                    "head_aucpr": wc(e, "head", "aucpr"),
                    "head_minus_floor": paired["delta_auc"],
                    "head_minus_floor_ci": paired["delta_auc_ci95"],
                    "best_epoch": res.get("best_epoch"),
                    "epochs_run": res.get("epochs_run"),
                    "trainable_params": res.get("trainable"),
                }
            )
    return pd.concat([pd.DataFrame(rows), per_year_rows(root)], ignore_index=True)


def per_year_rows(root: str) -> pd.DataFrame:
    """Pre-onset WC AUC per test year for the floor, the ceiling and every run's head."""
    from tornado_onset.score import roc_auc

    rows = []
    runs = runs_in(root)
    for h in range(_n_horizons(root)):
        src = {
            "floor": os.path.join(root, "detector", f"predictions_persistence_h{h}.csv"),
            "ceiling": os.path.join(root, "detector", f"predictions_truth_h{h}.csv"),
            **{d: os.path.join(root, d, "scores", f"predictions_head_h{h}.csv") for d in runs},
        }
        frames = {name: _wc_pre_onset(pd.read_csv(path)) for name, path in src.items()}
        years = sorted(pd.to_datetime(frames["floor"]["time"], unit="s").dt.year.unique())
        for year in years:
            row: dict = {"horizon": h, "year": int(year)}
            for name, df in frames.items():
                sub = df[pd.to_datetime(df["time"], unit="s").dt.year == year]
                row["pre_onset_cases"] = int(sub["pre_onset"].sum())
                key = {"floor": "floor_auc", "ceiling": "ceiling_auc"}.get(name, f"{name}_auc")
                row[key] = roc_auc(sub["label"].to_numpy(bool), sub["prob"].to_numpy(float))
            rows.append(row)
    return pd.DataFrame(rows)


def _n_horizons(root: str) -> int:
    det = _load(os.path.join(root, "detector", "summary.json")) or {"horizons": []}
    return len(det["horizons"])


def format_table(df: pd.DataFrame) -> str:
    df = df[df["run"].notna()] if "year" in df else df  # pooled rows only
    lines = []
    det = df[df["run"] == "detector"]
    lines.append("step 0, the published detector on one scan, pre-onset WC AUC:")
    for r in det.to_dict("records"):
        lines.append(
            f"  h{r['horizon']} (+{r['lead_min']:.0f} min, {r['pre_onset_cases']} cases):"
            f"  floor {r['floor_auc']:.3f}   ceiling {r['ceiling_auc']:.3f}"
        )
    lines.append("")
    lines.append("step run               h  lead   head   head-floor [95%]        best ep")
    for r in df[df["run"] != "detector"].to_dict("records"):
        lo, hi = r["head_minus_floor_ci"]
        lines.append(
            f"{int(r['step']):>4} {r['run']:<17} {r['horizon']}  {r['lead_min']:3.0f}   {r['head_auc']:.3f}  "
            f"{r['head_minus_floor']:+.3f} [{lo:+.3f}, {hi:+.3f}]   {int(r['best_epoch'])}"
        )
    return "\n".join(lines)


# --- figures ---------------------------------------------------------------------------------


def _curves(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ROC (fpr, tpr) and PR (recall, precision) of one predictions frame."""
    y = df["label"].to_numpy(bool)
    order = np.argsort(-df["prob"].to_numpy(float), kind="stable")
    hits = y[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    tpr = np.concatenate([[0.0], tp / max(y.sum(), 1)])
    fpr = np.concatenate([[0.0], fp / max((~y).sum(), 1)])
    precision = tp / np.arange(1, len(hits) + 1)
    return fpr, tpr, tpr[1:], precision


def _wc_pre_onset(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["pre_onset"] | (df["category"] != "TOR")) & df["category"].isin(["TOR", "WRN"])]


def make_figures(root: str, df: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    runs = runs_in(root)
    horizons = sorted(df["horizon"].unique())
    df = df[df["run"].notna()] if "year" in df else df  # pooled rows only
    det = df[df["run"] == "detector"].set_index("horizon")
    trained = df[df["run"] != "detector"]
    colors = plt.get_cmap("tab10")
    written = []

    # 1. the ladder: head AUC per step between floor and ceiling, one panel per horizon
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), horizons, strict=True):
        d = det.loc[h]
        sub = trained[trained["horizon"] == h]
        ax.axhline(d["floor_auc"], color="gray", ls="--", label="floor (detector, newest scan)")
        ax.axhline(d["ceiling_auc"], color="black", ls=":", label="ceiling (detector, future scan)")
        x = sub["step"].to_numpy()
        y = sub["head_auc"].to_numpy(float)
        ci = np.array(sub["head_ci"].tolist(), dtype=float).reshape(-1, 2)
        ax.errorbar(
            x,
            y,
            yerr=[y - ci[:, 0], ci[:, 1] - y],
            fmt="o",
            color=colors(0),
            capsize=3,
            label="head (95% bootstrap)",
        )
        ax.set_xticks(x, sub["run"].tolist(), rotation=20)
        ax.set_title(f"+{d['lead_min']:.0f} min, {int(d['pre_onset_cases'])} pre-onset cases")
        ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("pre-onset WC AUC")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    written.append(_save(fig, fig_dir, "ladder.png"))

    # 2. paired gain over the floor, with intervals
    fig, ax = plt.subplots(figsize=(7, 1 + 0.6 * len(trained)))
    ypos = np.arange(len(trained))
    for i, r in enumerate(trained.to_dict("records")):
        lo, hi = r["head_minus_floor_ci"]
        c = colors(horizons.index(r["horizon"]))
        ax.errorbar(
            r["head_minus_floor"],
            i,
            xerr=[[r["head_minus_floor"] - lo], [hi - r["head_minus_floor"]]],
            fmt="o",
            color=c,
            capsize=3,
        )
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(
        ypos, [f"{r['run']} (+{r['lead_min']:.0f} min)" for r in trained.to_dict("records")]
    )
    ax.invert_yaxis()
    ax.set_xlabel("head AUC minus floor AUC (pre-onset WC, paired bootstrap 95%)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    written.append(_save(fig, fig_dir, "delta.png"))

    # 3. ROC and PR curves on the pre-onset WC set: floor, ceiling, each run
    fig, axes = plt.subplots(2, len(horizons), figsize=(5 * len(horizons), 9), squeeze=False)
    for j, h in enumerate(horizons):
        series = {
            "floor": pd.read_csv(
                os.path.join(root, "detector", f"predictions_persistence_h{h}.csv")
            ),
            "ceiling": pd.read_csv(os.path.join(root, "detector", f"predictions_truth_h{h}.csv")),
            **{
                d: pd.read_csv(os.path.join(root, d, "scores", f"predictions_head_h{h}.csv"))
                for d in runs
            },
        }
        for k, (name, frame) in enumerate(series.items()):
            fpr, tpr, rec, prec = _curves(_wc_pre_onset(frame))
            style = {
                "floor": dict(color="gray", ls="--"),
                "ceiling": dict(color="black", ls=":"),
            }.get(name, dict(color=colors(k - 2)))
            axes[0, j].plot(fpr, tpr, label=name, **style)
            axes[1, j].plot(rec, prec, label=name, **style)
        axes[0, j].plot([0, 1], [0, 1], color="lightgray", lw=0.8)
        axes[0, j].set(
            title=f"ROC, +{det.loc[h, 'lead_min']:.0f} min",
            xlabel="false positive rate",
            ylabel="true positive rate",
        )
        axes[1, j].set(
            title=f"precision-recall, +{det.loc[h, 'lead_min']:.0f} min",
            xlabel="recall",
            ylabel="precision",
        )
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    written.append(_save(fig, fig_dir, "roc_pr.png"))

    # 4. training curves: validation head AUC and losses per epoch, best epoch marked
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, d in enumerate(runs):
        hist = pd.read_csv(os.path.join(root, d, "history.csv"))
        val_auc = hist[[c for c in hist.columns if c.startswith("val_AUC_h")]].mean(axis=1)
        axes[0].plot(hist["epoch"], val_auc, "o-", color=colors(k), label=d)
        axes[1].plot(hist["epoch"], hist["train_loss"], "o-", color=colors(k), label=f"{d} train")
        axes[1].plot(hist["epoch"], hist["val_loss"], "o--", color=colors(k), label=f"{d} val")
        if "best" in hist:
            best = hist[hist["best"].fillna(False).astype(bool)]
            if len(best):
                axes[0].plot(
                    best["epoch"].iloc[-1], val_auc[best.index[-1]], "*", color=colors(k), ms=14
                )
    axes[0].set(
        title="validation head AUC (mean of horizons); star = checkpoint kept",
        xlabel="epoch",
        ylabel="AUC",
    )
    axes[1].set(title="weighted BCE", xlabel="epoch", ylabel="loss")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    written.append(_save(fig, fig_dir, "training.png"))
    return written


def _save(fig, fig_dir: str, name: str) -> str:
    import matplotlib.pyplot as plt

    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summarize_root(root: str) -> str:
    """Write summary.csv, summary.txt and the figures; return the table."""
    df = summary_rows(root)
    df.to_csv(os.path.join(root, "summary.csv"), index=False)
    table = format_table(df)
    with open(os.path.join(root, "summary.txt"), "w") as f:
        f.write(table + "\n")
    try:
        for path in make_figures(root, df):
            logging.info("wrote %s", path)
    except Exception:  # a plotting bug must never cost the run; the data is on disk
        logging.exception("figures failed; rerun with `run.py summary %s` after fixing", root)
    return table
