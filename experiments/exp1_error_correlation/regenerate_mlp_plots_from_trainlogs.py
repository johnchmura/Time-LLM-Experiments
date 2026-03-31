"""
Regenerate MLP training-curve PNGs from existing mlp_*_trainlog.json files.

MLP only: trainlogs contain per-epoch history. XGBoost validation curves are not
stored in saved xgb_*.json models; regenerate those by re-running training or
use PNGs from the original run.

Does not import torch or xgboost.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_mlp_training_plot(history: List[Dict[str, Any]], out_path: str) -> None:
    if not history:
        return
    _ensure_dir(os.path.dirname(out_path))
    epochs = [h["epoch"] for h in history]
    loss = [h["train_loss"] for h in history]
    roc = [h["val_roc_auc"] for h in history]
    pr = [h["val_pr_auc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, loss, color="C0")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].set_title("MLP train loss")

    axes[1].plot(epochs, roc, color="C1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("MLP val ROC-AUC")

    axes[2].plot(epochs, pr, color="C2")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("PR-AUC")
    axes[2].set_title("MLP val PR-AUC")

    fig.suptitle("MLP training curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_trainlog_filename(name: str) -> Optional[Tuple[str, str]]:
    if not name.startswith("mlp_") or not name.endswith("_trainlog.json"):
        return None
    stem = name[len("mlp_") : -len("_trainlog.json")]
    for fs in ("all_features", "l1_selected"):
        suf = "_" + fs
        if stem.endswith(suf):
            return stem[: -len(suf)], fs
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Write plots/<target>/<feature_set>/mlp_training.png from mlp_*_trainlog.json"
    )
    p.add_argument(
        "--dir",
        type=str,
        default="experiments/results/exp1_error_correlation/ml_models",
        help="Directory containing mlp_*_trainlog.json files",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing PNGs",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each file processed",
    )
    args = p.parse_args()
    base = os.path.abspath(args.dir)
    if not os.path.isdir(base):
        print(f"Not a directory: {base}", file=sys.stderr)
        return 1

    pattern = os.path.join(base, "mlp_*_trainlog.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files matched: {pattern}", file=sys.stderr)
        return 1

    n_ok = 0
    n_skip = 0
    for trainlog_path in paths:
        name = os.path.basename(trainlog_path)
        parsed = parse_trainlog_filename(name)
        if parsed is None:
            print(f"Skip (unrecognized name): {name}", file=sys.stderr)
            n_skip += 1
            continue
        tname, fset = parsed
        out_path = os.path.join(base, "plots", tname, fset, "mlp_training.png")
        if args.verbose or args.dry_run:
            print(f"{trainlog_path} -> {out_path}")

        with open(trainlog_path, "r") as f:
            data = json.load(f)
        history = data.get("history") or []
        if not history:
            print(f"Skip (empty history): {name}", file=sys.stderr)
            n_skip += 1
            continue

        if args.dry_run:
            n_ok += 1
            continue

        _save_mlp_training_plot(history, out_path)
        n_ok += 1

    if args.dry_run:
        print(f"Dry-run: would write {n_ok} PNG(s), skip {n_skip}")
    else:
        print(f"Wrote {n_ok} PNG(s), skip {n_skip}")
    return 0 if n_ok > 0 or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
