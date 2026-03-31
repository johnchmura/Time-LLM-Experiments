"""
Train one logistic regression per feature in features.npz.

Targets:
- high_error_{median,q75,q90}: errors > threshold
- directional_correct: sign(pred_t - pred_{t-1}) matches sign(true_t - true_{t-1})

Example:
  python experiments/exp1_error_correlation/train_logreg_by_feature.py \
    --npz_path experiments/results/exp1_error_correlation/features.npz \
    --out_dir experiments/results/exp1_error_correlation/logreg
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


SKIP_KEYS = {"errors", "channel_ids", "directional_correct"}


def _finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _as_binary(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.int64)
    u = np.unique(y)
    if not np.all((u == 0) | (u == 1)):
        raise ValueError(f"Target must be binary 0/1, got values: {u}")
    return y


def _eval_binary(y_true: np.ndarray, prob1: np.ndarray) -> Dict[str, float]:
    pred = (prob1 >= 0.5).astype(np.int64)
    out: Dict[str, float] = {}
    out["roc_auc"] = float(roc_auc_score(y_true, prob1))
    out["accuracy"] = float(accuracy_score(y_true, pred))
    out["precision"] = float(precision_score(y_true, pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, pred, zero_division=0))
    out["pos_rate"] = float(np.mean(y_true))
    return out


def _fit_eval_one(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> Dict[str, Any]:
    x, y = _finite_xy(x, y)
    y = _as_binary(y)
    if len(x) < 50:
        return {"skipped": True, "reason": "too_few_samples", "n": int(len(x))}
    if len(np.unique(y)) < 2:
        return {"skipped": True, "reason": "single_class", "n": int(len(x))}

    x_tr, x_te, y_tr, y_te = train_test_split(
        x.reshape(-1, 1),
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    clf = LogisticRegression(
        penalty="l2",
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(x_tr, y_tr)
    prob1 = clf.predict_proba(x_te)[:, 1]

    metrics = _eval_binary(y_te, prob1)
    metrics.update(
        {
            "skipped": False,
            "n": int(len(x)),
            "n_train": int(len(x_tr)),
            "n_test": int(len(x_te)),
            "coef": float(clf.coef_.ravel()[0]),
            "intercept": float(clf.intercept_.ravel()[0]),
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1: logistic regression by feature")
    p.add_argument(
        "--npz_path",
        type=str,
        default="experiments/results/exp1_error_correlation/features.npz",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/results/exp1_error_correlation/logreg",
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    npz = np.load(args.npz_path, allow_pickle=False)
    files = list(npz.files)
    if "errors" not in files:
        raise ValueError("NPZ missing required key: errors")

    errors = npz["errors"].astype(np.float64)
    thresholds = {
        "median": float(np.nanmedian(errors)),
        "q75": float(np.nanquantile(errors, 0.75)),
        "q90": float(np.nanquantile(errors, 0.90)),
    }

    targets: Dict[str, np.ndarray] = {
        f"high_error_{k}": (errors > t).astype(np.int64) for k, t in thresholds.items()
    }
    if "directional_correct" in files:
        targets["directional_correct"] = npz["directional_correct"].astype(np.int64)

    summary: Dict[str, Any] = {
        "npz_path": args.npz_path,
        "test_size": args.test_size,
        "seed": args.seed,
        "thresholds": thresholds,
        "targets_present": sorted(list(targets.keys())),
        "features": {},
    }

    feature_keys = [k for k in files if k not in SKIP_KEYS and k not in targets]
    feature_keys = sorted(feature_keys)

    for feat in feature_keys:
        x = npz[feat].astype(np.float64)
        res_by_target: Dict[str, Any] = {}
        for tname, y in targets.items():
            if len(x) != len(y):
                res_by_target[tname] = {
                    "skipped": True,
                    "reason": "length_mismatch",
                    "n_x": int(len(x)),
                    "n_y": int(len(y)),
                }
                continue
            res_by_target[tname] = _fit_eval_one(
                x,
                y,
                test_size=args.test_size,
                seed=args.seed,
            )
        summary["features"][feat] = res_by_target

    out_path = os.path.join(args.out_dir, "logreg_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

