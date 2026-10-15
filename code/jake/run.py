"""Tornado onset: the published detector and two runs that build on it, on TorNet 2014-2015.

    uv run run.py all                          # every step in order, into results/<stamp>/
    uv run run.py finetuned results/X          # one step into an existing directory
    uv run run.py summary results/X            # rebuild summary.csv/.txt and the figures

    step  run         what changes                                question
    0     detector    nothing trained: the published detector     what the detector already gives
                      as designed, one scan, scored on the        on the onset question. Newest
                      test set                                    observed scan: the floor. Real
                                                                  future scan: the ceiling.
    1     frozen      encoder frozen, both observed scans in,     can the detector's own features,
                      features concatenated, a new head trained   read over two scans, be re-aimed
                      on the +5 / +10 min labels                  at onset
    2     finetuned   encoder unfrozen at a tenth of the head's   does adapting the features to the
                      learning rate                               target buy more

Each invocation writes one self-contained directory, results/<stamp>/: detector/ (floor and
ceiling predictions), one directory per trained run (params.json, history.csv, checkpoints/,
scores/ with the head predictions and summary.json, results.json), summary.csv, summary.txt
and figures/ (ladder, paired gain, ROC and PR curves, training curves). Each trained run uses
the 2014-2015 training partitions (validation held out by day), keeps the checkpoint with the
best validation head AUC and scores it against step zero. Model weights and data come from the
Hugging Face Hub and are cached on first use.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from typing import Any

from tornado_onset.config import Config
from tornado_onset.figures import summarize_root
from tornado_onset.train import run, run_detector

RUNS: dict[str, dict[str, Any]] = {
    "frozen": {"encoder_trainable": False},
    "finetuned": {"encoder_trainable": True},
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", choices=["all", *RUNS, "detector", "summary"])
    p.add_argument(
        "dir",
        nargs="?",
        help="results directory to write into (default results/<timestamp>) or, for summary, "
        "the finished one to rebuild; a single run into an existing directory reuses its "
        "detector step",
    )
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = a.dir or os.path.join("results", datetime.datetime.now().strftime("%y%m%d_%H%M"))
    if a.run != "summary":
        names = list(RUNS) if a.run == "all" else [] if a.run == "detector" else [a.run]
        run_detector(Config(exp_name="detector", out_dir=out))
        for name in names:
            run(Config(exp_name=name, **RUNS[name], out_dir=out))
    print(summarize_root(out))


if __name__ == "__main__":
    main()
