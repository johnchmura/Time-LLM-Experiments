"""
Train multivariate L1-regularized logistic regression on all Exp1 NPZ features.

Runs separately for:
- high_error_median
- high_error_q75
- high_error_q90

Selects C by 5-fold stratified CV optimizing ROC-AUC.

Example:
  python experiments/exp1_error_correlation/train_logreg_l1_multivar.py \
    --npz_path experiments/results/exp1_error_correlation/features.npz \
    --out_dir experiments/results/exp1_error_correlation/logreg_multivar \
    --max_rows 300000
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["high_error_median", "high_error_q75", "high_error_q90"]
SKIP_KEYS = {
    "errors",
    "channel_ids",
    "directional_correct",
    "sample_id",
    "dataset_row_index",
    "horizon_t",
    "y_true",
    "y_pred",
    "y_prev",
    "pred_prev",
    "direction_true",
    "direction_pred",
    "timestamp",
    "time_features",
}


def _load_thresholds_from_summary(summary_path: str) -> Optional[Dict[str, float]]:
    if not summary_path:
        return None
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r") as f:
        d = json.load(f)
    th = d.get("thresholds", None)
    if not isinstance(th, dict):
        return None
    need = {"median", "q75", "q90"}
    if not need.issubset(set(th.keys())):
        return None
    return {k: float(th[k]) for k in ["median", "q75", "q90"]}


def _compute_thresholds(errors: np.ndarray) -> Dict[str, float]:
    return {
        "median": float(np.nanmedian(errors)),
        "q75": float(np.nanquantile(errors, 0.75)),
        "q90": float(np.nanquantile(errors, 0.90)),
    }


def _finite_rows_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(axis=1)


def _subsample_rows_stratified(
    x: np.ndarray, y: np.ndarray, max_rows: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y) <= max_rows:
        idx = np.arange(len(y))
        return x, y, idx
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n1 = int(round(max_rows * (len(idx1) / max(len(y), 1))))
    n1 = max(1, min(len(idx1), n1))
    n0 = max_rows - n1
    n0 = max(1, min(len(idx0), n0))
    sel = np.concatenate([rng.choice(idx0, n0, replace=False), rng.choice(idx1, n1, replace=False)])
    rng.shuffle(sel)
    return x[sel], y[sel], sel


def _eval_holdout(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    pred = (p >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "pos_rate": float(np.mean(y_true)),
    }


@dataclass(frozen=True)
class Args:
    npz_path: str
    out_dir: str
    seed: int
    cv_folds: int
    c_min: float
    c_max: float
    c_steps: int
    max_rows: int
    test_size: float
    summary_path: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Exp1: multivariate L1 logistic regression")
    p.add_argument(
        "--npz_path",
        type=str,
        default="experiments/results/exp1_error_correlation/features.npz",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/results/exp1_error_correlation/logreg_multivar",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--c_min", type=float, default=1e-3)
    p.add_argument("--c_max", type=float, default=1e2)
    p.add_argument("--c_steps", type=int, default=12)
    p.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional stratified subsample size (0 means use all rows).",
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument(
        "--summary_path",
        type=str,
        default="experiments/results/exp1_error_correlation/logreg/logreg_summary.json",
        help="Optional path to reuse thresholds for median/q75/q90.",
    )
    a = p.parse_args()
    return Args(
        npz_path=a.npz_path,
        out_dir=a.out_dir,
        seed=a.seed,
        cv_folds=a.cv_folds,
        c_min=a.c_min,
        c_max=a.c_max,
        c_steps=a.c_steps,
        max_rows=a.max_rows,
        test_size=a.test_size,
        summary_path=a.summary_path,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    npz = np.load(args.npz_path, allow_pickle=False)
    files = list(npz.files)
    if "errors" not in files:
        raise ValueError("NPZ missing required key: errors")

    errors = npz["errors"].astype(np.float64)

    feature_keys = [k for k in files if k not in SKIP_KEYS]
    feature_keys = [k for k in feature_keys if k != "errors"]
    feature_keys = sorted(feature_keys)

    x = np.stack([npz[k].astype(np.float64) for k in feature_keys], axis=1)
    mask = _finite_rows_mask(x) & np.isfinite(errors)
    x = x[mask]
    errors = errors[mask]

    thresholds = _load_thresholds_from_summary(args.summary_path) or _compute_thresholds(errors)
    y_by_target = {
        "high_error_median": (errors > thresholds["median"]).astype(np.int64),
        "high_error_q75": (errors > thresholds["q75"]).astype(np.int64),
        "high_error_q90": (errors > thresholds["q90"]).astype(np.int64),
    }

    c_grid = np.logspace(np.log10(args.c_min), np.log10(args.c_max), args.c_steps)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    out: Dict[str, Any] = {
        "npz_path": args.npz_path,
        "out_dir": args.out_dir,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "c_grid": [float(v) for v in c_grid],
        "max_rows": int(args.max_rows),
        "test_size": float(args.test_size),
        "thresholds": thresholds,
        "n_rows_total": int(len(x)),
        "n_features": int(x.shape[1]),
        "feature_keys": feature_keys,
        "targets": {},
    }

    for tname in TARGETS:
        y_full = y_by_target[tname]
        x_t, y_t, sel_idx = _subsample_rows_stratified(x, y_full, args.max_rows, args.seed)

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

        search = GridSearchCV(
            pipe,
            param_grid={"clf__C": c_grid},
            scoring="roc_auc",
            cv=cv,
            n_jobs=10,
            refit=True,
        )
        search.fit(x_t, y_t)

        best_pipe = search.best_estimator_
        best_clf: LogisticRegression = best_pipe.named_steps["clf"]
        coefs = best_clf.coef_.ravel()
        nz = np.flatnonzero(coefs != 0.0)
        selected = sorted(
            [{"feature": feature_keys[i], "coef": float(coefs[i])} for i in nz],
            key=lambda d: abs(d["coef"]),
            reverse=True,
        )

        x_tr, x_te, y_tr, y_te = train_test_split(
            x_t,
            y_t,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y_t,
        )
        best_pipe.fit(x_tr, y_tr)
        p = best_pipe.predict_proba(x_te)[:, 1]
        holdout = _eval_holdout(y_te, p)

        out["targets"][tname] = {
            "n_rows_used": int(len(x_t)),
            "pos_rate_used": float(np.mean(y_t)),
            "best_C": float(search.best_params_["clf__C"]),
            "cv_mean_auc": float(search.best_score_),
            "n_selected": int(len(selected)),
            "selected": selected,
            "holdout": holdout,
        }

    out_path = os.path.join(args.out_dir, "multivar_l1_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

