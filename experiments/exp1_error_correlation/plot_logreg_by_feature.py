"""
Plot logistic regression performance per feature from Experiment 1 artifacts.

Inputs:
- features.npz (errors + feature arrays)
- logreg_summary.json (coef/intercept + thresholds)

Outputs:
- one directory per feature under out_root
- index.json under out_root summarizing files

Example:
  python experiments/exp1_error_correlation/plot_logreg_by_feature.py \
    --npz_path experiments/results/exp1_error_correlation/features.npz \
    --summary_path experiments/results/exp1_error_correlation/logreg/logreg_summary.json \
    --out_root experiments/results/exp1_error_correlation/logreg/plots
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


TARGETS = ["high_error_median", "high_error_q75", "high_error_q90"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _subsample_idx(n: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def _binned_curve(x: np.ndarray, y: np.ndarray, *, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]
    edges = np.linspace(0, len(x_s), bins + 1).astype(int)
    xs, ys = [], []
    for i in range(bins):
        a, b = edges[i], edges[i + 1]
        if b - a <= 0:
            continue
        xs.append(float(np.mean(x_s[a:b])))
        ys.append(float(np.mean(y_s[a:b])))
    return np.array(xs), np.array(ys)


def _plot_roc(y: np.ndarray, p: np.ndarray, out_path: str) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(y, p)
    auc = float(roc_auc_score(y, p))
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"roc_auc": auc}


def _plot_calibration(y: np.ndarray, p: np.ndarray, out_path: str, *, bins: int) -> Dict[str, float]:
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    inds = np.digitize(p, edges[1:-1], right=False)
    mp, ep, counts = [], [], []
    for b in range(bins):
        m = inds == b
        if not np.any(m):
            continue
        mp.append(float(np.mean(p[m])))
        ep.append(float(np.mean(y[m])))
        counts.append(int(np.sum(m)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="ideal")
    ax.plot(mp, ep, marker="o", linewidth=1.5, label="empirical")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical positive rate")
    ax.set_title("Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ece = 0.0
    n = len(y)
    for mpp, epp, c in zip(mp, ep, counts):
        ece += (c / max(n, 1)) * abs(epp - mpp)
    return {"ece": float(ece)}


def _plot_confusion(y: np.ndarray, p: np.ndarray, out_path: str) -> Dict[str, float]:
    pred = (p >= 0.5).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    acc = float(accuracy_score(y, pred))
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion @0.5  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"accuracy": acc, "precision": prec, "recall": rec}


def _plot_risk_vs_error(
    errors: np.ndarray, p: np.ndarray, out_path: str, *, bins: int
) -> None:
    xs, ys = _binned_curve(p, errors, bins=bins)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("mean absolute error")
    ax.set_title("Risk vs absolute error (binned)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_vs_error(
    x: np.ndarray,
    errors: np.ndarray,
    p: np.ndarray,
    out_path: str,
    *,
    max_points: int,
    seed: int,
) -> None:
    idx = _subsample_idx(len(x), max_points=max_points, seed=seed)
    xs = x[idx]
    es = errors[idx]
    ps = p[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(xs, es, c=ps, s=4, alpha=0.10, rasterized=True)
    ax.set_xlabel("feature value")
    ax.set_ylabel("absolute error")
    ax.set_title("Feature vs absolute error (colored by predicted risk)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("predicted risk")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class Args:
    npz_path: str
    summary_path: str
    out_root: str
    seed: int
    max_points: int
    max_eval_points: int
    overwrite: bool
    calibration_bins: int
    risk_bins: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Exp1: plot logreg by feature")
    p.add_argument(
        "--npz_path",
        type=str,
        default="experiments/results/exp1_error_correlation/features.npz",
    )
    p.add_argument(
        "--summary_path",
        type=str,
        default="experiments/results/exp1_error_correlation/logreg/logreg_summary.json",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="experiments/results/exp1_error_correlation/logreg/plots",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_points", type=int, default=50_000)
    p.add_argument("--max_eval_points", type=int, default=300_000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--calibration_bins", type=int, default=15)
    p.add_argument("--risk_bins", type=int, default=25)
    a = p.parse_args()
    return Args(
        npz_path=a.npz_path,
        summary_path=a.summary_path,
        out_root=a.out_root,
        seed=a.seed,
        max_points=a.max_points,
        max_eval_points=a.max_eval_points,
        overwrite=bool(a.overwrite),
        calibration_bins=a.calibration_bins,
        risk_bins=a.risk_bins,
    )


def _maybe_skip(path: str, overwrite: bool) -> bool:
    return (not overwrite) and os.path.exists(path)


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_root)

    with open(args.summary_path, "r") as f:
        summary = json.load(f)

    npz = np.load(args.npz_path, allow_pickle=False)
    errors = npz["errors"].astype(np.float64)

    thresholds = summary.get("thresholds", {})
    if not all(k in thresholds for k in ["median", "q75", "q90"]):
        raise ValueError("summary missing required thresholds: median/q75/q90")

    y_by_target: Dict[str, np.ndarray] = {
        "high_error_median": (errors > float(thresholds["median"])).astype(np.int64),
        "high_error_q75": (errors > float(thresholds["q75"])).astype(np.int64),
        "high_error_q90": (errors > float(thresholds["q90"])).astype(np.int64),
    }

    index: Dict[str, Any] = {
        "npz_path": args.npz_path,
        "summary_path": args.summary_path,
        "out_root": args.out_root,
        "seed": args.seed,
        "max_points": args.max_points,
        "max_eval_points": args.max_eval_points,
        "thresholds": thresholds,
        "targets": TARGETS,
        "features": {},
    }

    for feat, per_target in summary.get("features", {}).items():
        if feat not in npz.files:
            continue
        x_all = npz[feat].astype(np.float64)

        feat_dir = os.path.join(args.out_root, feat)
        _ensure_dir(feat_dir)

        feat_index: Dict[str, Any] = {"targets": {}}
        for tname in TARGETS:
            m = per_target.get(tname, {})
            if m.get("skipped", False):
                feat_index["targets"][tname] = {"skipped": True, "reason": m.get("reason")}
                continue
            if "coef" not in m or "intercept" not in m:
                feat_index["targets"][tname] = {"skipped": True, "reason": "missing_coef"}
                continue

            coef = float(m["coef"])
            intercept = float(m["intercept"])
            y_all = y_by_target[tname]

            mask = _finite_mask(x_all, errors, y_all.astype(np.float64))
            x = x_all[mask]
            e = errors[mask]
            y = y_all[mask]

            z = intercept + coef * x
            p = _sigmoid(z)

            eval_idx = _subsample_idx(len(x), max_points=args.max_eval_points, seed=args.seed)
            x_e = x[eval_idx]
            e_e = e[eval_idx]
            y_e = y[eval_idx]
            p_e = p[eval_idx]

            out_files: Dict[str, str] = {}
            roc_path = os.path.join(feat_dir, f"roc_curve_{tname}.png")
            cal_path = os.path.join(feat_dir, f"calibration_{tname}.png")
            risk_path = os.path.join(feat_dir, f"risk_vs_error_{tname}.png")
            scat_path = os.path.join(feat_dir, f"feature_vs_error_{tname}.png")
            conf_path = os.path.join(feat_dir, f"confusion_{tname}.png")

            metrics: Dict[str, Any] = {"n_total": int(len(x)), "n_eval": int(len(x_e))}

            if not _maybe_skip(roc_path, args.overwrite):
                metrics.update(_plot_roc(y_e, p_e, roc_path))
            out_files["roc_curve"] = os.path.basename(roc_path)

            if not _maybe_skip(cal_path, args.overwrite):
                metrics.update(_plot_calibration(y_e, p_e, cal_path, bins=args.calibration_bins))
            out_files["calibration"] = os.path.basename(cal_path)

            if not _maybe_skip(conf_path, args.overwrite):
                metrics.update(_plot_confusion(y_e, p_e, conf_path))
            out_files["confusion"] = os.path.basename(conf_path)

            if not _maybe_skip(risk_path, args.overwrite):
                _plot_risk_vs_error(e_e, p_e, risk_path, bins=args.risk_bins)
            out_files["risk_vs_error"] = os.path.basename(risk_path)

            if not _maybe_skip(scat_path, args.overwrite):
                _plot_feature_vs_error(
                    x_e,
                    e_e,
                    p_e,
                    scat_path,
                    max_points=args.max_points,
                    seed=args.seed,
                )
            out_files["feature_vs_error"] = os.path.basename(scat_path)

            feat_index["targets"][tname] = {
                "skipped": False,
                "coef": coef,
                "intercept": intercept,
                "metrics": metrics,
                "files": out_files,
            }

        index["features"][feat] = feat_index

    index_path = os.path.join(args.out_root, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved: {index_path}")


if __name__ == "__main__":
    main()

