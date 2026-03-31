from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "final_layer_norm",
    "mean_layer_norm",
    "layer_norm_var",
    "centroid_distance",
    "knn_distance",
    "reprog_norm",
    "reprog_dim_var",
    "reprog_centroid_distance",
    "reprog_knn_distance",
    "reprog_entropy",
    "reprog_max_attn",
    "reprog_entropy_head_var",
    "pre_head_norm",
    "temporal_h_delta",
    "pred_delta",
    "llm_attn_entropy",
    "llm_attn_max_weight",
    "llm_attn_entropy_head_var",
]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _calibrate_probs(raw_cal: np.ndarray, y_cal: np.ndarray, raw_eval: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return np.clip(raw_eval, 0.0, 1.0)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.clip(raw_cal, 0.0, 1.0), y_cal.astype(np.float64))
        return np.clip(iso.predict(np.clip(raw_eval, 0.0, 1.0)), 0.0, 1.0)
    if method == "platt":
        clf = LogisticRegression(max_iter=2000)
        clf.fit(raw_cal.reshape(-1, 1), y_cal.astype(np.int64))
        return clf.predict_proba(raw_eval.reshape(-1, 1))[:, 1]
    raise ValueError(f"Unknown calibration method: {method}")


def _build_feature_matrix(npz: np.lib.npyio.NpzFile, feature_keys: List[str]) -> np.ndarray:
    x = np.stack([npz[k].astype(np.float64) for k in feature_keys], axis=1)
    return x


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _random_baseline_curves(
    directional_correct: np.ndarray,
    abs_error: np.ndarray,
    coverages: List[float],
    *,
    seed: int,
    repeats: int,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    n = len(abs_error)
    mda_curve, mae_curve = [], []
    for cov in coverages:
        keep_n = max(1, int(round(cov * n)))
        m_vals = []
        a_vals = []
        for _ in range(repeats):
            idx = rng.choice(n, size=keep_n, replace=False)
            m_vals.append(float(np.mean(directional_correct[idx])))
            a_vals.append(float(np.mean(abs_error[idx])))
        mda_curve.append(float(np.mean(m_vals)))
        mae_curve.append(float(np.mean(a_vals)))
    return {"mda": mda_curve, "mae": mae_curve}


def _oracle_curves(
    directional_correct: np.ndarray,
    abs_error: np.ndarray,
    coverages: List[float],
) -> Dict[str, List[float]]:
    n = len(abs_error)
    order_mda = np.argsort(1 - directional_correct)  # keep correct first
    order_mae = np.argsort(abs_error)  # keep low-error first
    mda_curve, mae_curve = [], []
    for cov in coverages:
        keep_n = max(1, int(round(cov * n)))
        idx_m = order_mda[:keep_n]
        idx_a = order_mae[:keep_n]
        mda_curve.append(float(np.mean(directional_correct[idx_m])))
        mae_curve.append(float(np.mean(abs_error[idx_a])))
    return {"mda": mda_curve, "mae": mae_curve}


def _block_bootstrap_delta(
    directional_correct: np.ndarray,
    abs_error: np.ndarray,
    model_keep_mask: np.ndarray,
    *,
    block_size: int,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    n = len(abs_error)
    if n == 0:
        return {
            "delta_mda_mean": float("nan"),
            "delta_mda_ci_low": float("nan"),
            "delta_mda_ci_high": float("nan"),
            "delta_mae_mean": float("nan"),
            "delta_mae_ci_low": float("nan"),
            "delta_mae_ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    full_mda = float(np.mean(directional_correct))
    full_mae = float(np.mean(abs_error))
    d_mda, d_mae = [], []
    b = max(1, min(block_size, n))
    reps = int(np.ceil(n / b))
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - b + 1), size=reps)
        idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
        keep = model_keep_mask[idx]
        if np.sum(keep) == 0:
            continue
        mda_sel = float(np.mean(directional_correct[idx][keep]))
        mae_sel = float(np.mean(abs_error[idx][keep]))
        d_mda.append(mda_sel - full_mda)
        d_mae.append(full_mae - mae_sel)
    if not d_mda:
        return {
            "delta_mda_mean": float("nan"),
            "delta_mda_ci_low": float("nan"),
            "delta_mda_ci_high": float("nan"),
            "delta_mae_mean": float("nan"),
            "delta_mae_ci_low": float("nan"),
            "delta_mae_ci_high": float("nan"),
        }
    dm = np.asarray(d_mda)
    da = np.asarray(d_mae)
    return {
        "delta_mda_mean": float(np.mean(dm)),
        "delta_mda_ci_low": float(np.quantile(dm, 0.025)),
        "delta_mda_ci_high": float(np.quantile(dm, 0.975)),
        "delta_mae_mean": float(np.mean(da)),
        "delta_mae_ci_low": float(np.quantile(da, 0.025)),
        "delta_mae_ci_high": float(np.quantile(da, 0.975)),
    }


def _fit_directional_classifier(x_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> Any:
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
                    random_state=seed,
                ),
            ),
        ]
    )
    grid = {"clf__C": np.logspace(-3, 2, 10)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=8, refit=True)
    gs.fit(x_tr, y_tr)
    return gs.best_estimator_, float(gs.best_score_), float(gs.best_params_["clf__C"])


def _fit_abs_regressor(x_tr: np.ndarray, y_tr: np.ndarray) -> Any:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    model.fit(x_tr, y_tr)
    return model


@dataclass(frozen=True)
class Args:
    val_npz: str
    test_npz: str
    out_dir: str
    coverages: List[float]
    primary_coverage: float
    risk_cal_size: float
    seed: int
    calibration: str
    bootstrap_iters: int
    bootstrap_block_size: int
    random_repeats: int
    protocol_label: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Selective evaluation for MDA and MAE")
    p.add_argument(
        "--val_npz",
        type=str,
        default="experiments/results/exp1_error_correlation/features_val.npz",
    )
    p.add_argument(
        "--test_npz",
        type=str,
        default="experiments/results/exp1_error_correlation/features_test.npz",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/results/exp1_error_correlation/selective_eval",
    )
    p.add_argument("--coverages", type=str, default="1.0,0.95,0.90,0.80,0.70")
    p.add_argument("--primary_coverage", type=float, default=0.90)
    p.add_argument("--risk_cal_size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration", type=str, default="isotonic", choices=["none", "isotonic", "platt"])
    p.add_argument("--bootstrap_iters", type=int, default=1000)
    p.add_argument("--bootstrap_block_size", type=int, default=1024)
    p.add_argument("--random_repeats", type=int, default=250)
    p.add_argument(
        "--protocol_label",
        type=str,
        default="",
        help="Optional label for split protocol (e.g., standard, strict_gap_seq_len).",
    )
    a = p.parse_args()
    return Args(
        val_npz=a.val_npz,
        test_npz=a.test_npz,
        out_dir=a.out_dir,
        coverages=_parse_float_list(a.coverages),
        primary_coverage=float(a.primary_coverage),
        risk_cal_size=float(a.risk_cal_size),
        seed=int(a.seed),
        calibration=a.calibration,
        bootstrap_iters=int(a.bootstrap_iters),
        bootstrap_block_size=int(a.bootstrap_block_size),
        random_repeats=int(a.random_repeats),
        protocol_label=a.protocol_label.strip(),
    )


def _coverage_thresholds_from_cal(risk_cal: np.ndarray, coverages: List[float]) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for c in coverages:
        out[c] = float(np.quantile(risk_cal, c))
    return out


def _curve_from_thresholds(
    risk: np.ndarray,
    thresholds: Dict[float, float],
    directional_correct: np.ndarray,
    abs_error: np.ndarray,
) -> Dict[str, List[float]]:
    covs = sorted(thresholds.keys(), reverse=True)
    mda, mae, achieved_cov = [], [], []
    for c in covs:
        keep = risk <= thresholds[c]
        if np.sum(keep) == 0:
            mda.append(float("nan"))
            mae.append(float("nan"))
            achieved_cov.append(0.0)
            continue
        mda.append(float(np.mean(directional_correct[keep])))
        mae.append(float(np.mean(abs_error[keep])))
        achieved_cov.append(float(np.mean(keep)))
    return {"coverages": covs, "mda": mda, "mae": mae, "achieved_coverage": achieved_cov}


def _plot_curves(
    out_path: str,
    model_curves: Dict[str, Dict[str, List[float]]],
    random_curve: Dict[str, List[float]],
    oracle_curve: Dict[str, List[float]],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, cur in model_curves.items():
        ax1.plot(cur["coverages"], cur["mda"], marker="o", label=name)
        ax2.plot(cur["coverages"], cur["mae"], marker="o", label=name)
    covs = sorted(model_curves[next(iter(model_curves))]["coverages"], reverse=True)
    ax1.plot(covs, random_curve["mda"], linestyle="--", label="random")
    ax1.plot(covs, oracle_curve["mda"], linestyle=":", label="oracle")
    ax2.plot(covs, random_curve["mae"], linestyle="--", label="random")
    ax2.plot(covs, oracle_curve["mae"], linestyle=":", label="oracle")
    ax1.set_title("Coverage vs Selective-MDA")
    ax1.set_xlabel("target coverage")
    ax1.set_ylabel("MDA")
    ax2.set_title("Coverage vs Selective-MAE")
    ax2.set_xlabel("target coverage")
    ax2.set_ylabel("MAE")
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_dir)

    val = np.load(args.val_npz, allow_pickle=False)
    test = np.load(args.test_npz, allow_pickle=False)

    feature_keys = [k for k in DEFAULT_FEATURES if k in val.files and k in test.files]
    if not feature_keys:
        raise ValueError("No shared known feature keys found in val/test NPZ.")

    x_val = _build_feature_matrix(val, feature_keys)
    x_test = _build_feature_matrix(test, feature_keys)

    dir_val = val["directional_correct"].astype(np.int64)
    dir_test = test["directional_correct"].astype(np.int64)
    abs_val = val["errors"].astype(np.float64)
    abs_test = test["errors"].astype(np.float64)

    # Keep finite rows only.
    m_val = np.isfinite(x_val).all(axis=1) & np.isfinite(abs_val) & np.isfinite(dir_val.astype(np.float64))
    m_test = np.isfinite(x_test).all(axis=1) & np.isfinite(abs_test) & np.isfinite(dir_test.astype(np.float64))
    x_val, dir_val, abs_val = x_val[m_val], dir_val[m_val], abs_val[m_val]
    x_test, dir_test, abs_test = x_test[m_test], dir_test[m_test], abs_test[m_test]

    y_dir_err = (1 - dir_val).astype(np.int64)
    idx_tr, idx_cal = train_test_split(
        np.arange(len(y_dir_err)),
        test_size=args.risk_cal_size,
        random_state=args.seed,
        stratify=y_dir_err,
    )
    x_tr, x_cal = x_val[idx_tr], x_val[idx_cal]
    y_tr_dir, y_cal_dir = y_dir_err[idx_tr], y_dir_err[idx_cal]
    y_tr_abs, y_cal_abs = abs_val[idx_tr], abs_val[idx_cal]

    # Directional error classifier.
    clf, cv_auc, best_c = _fit_directional_classifier(x_tr, y_tr_dir, args.seed)
    raw_cal = clf.predict_proba(x_cal)[:, 1]
    raw_test = clf.predict_proba(x_test)[:, 1]
    risk_dir_cal = _calibrate_probs(raw_cal, y_cal_dir, raw_cal, args.calibration)
    risk_dir_test = _calibrate_probs(raw_cal, y_cal_dir, raw_test, args.calibration)

    # Absolute error regressor.
    reg = _fit_abs_regressor(x_tr, y_tr_abs)
    risk_abs_cal = reg.predict(x_cal).astype(np.float64)
    risk_abs_test = reg.predict(x_test).astype(np.float64)

    thresholds_dir = _coverage_thresholds_from_cal(risk_dir_cal, args.coverages)
    thresholds_abs = _coverage_thresholds_from_cal(risk_abs_cal, args.coverages)

    model_curves = {
        "directional_classifier": _curve_from_thresholds(risk_dir_test, thresholds_dir, dir_test, abs_test),
        "abs_error_regression": _curve_from_thresholds(risk_abs_test, thresholds_abs, dir_test, abs_test),
    }

    random_curve = _random_baseline_curves(
        dir_test,
        abs_test,
        sorted(args.coverages, reverse=True),
        seed=args.seed,
        repeats=args.random_repeats,
    )
    oracle_curve = _oracle_curves(dir_test, abs_test, sorted(args.coverages, reverse=True))

    # Primary-coverage bootstrap CIs for improvement over random baseline expectation.
    p_cov = min(args.coverages, key=lambda x: abs(x - args.primary_coverage))
    thr_dir_primary = thresholds_dir[p_cov]
    keep_primary = risk_dir_test <= thr_dir_primary
    ci_primary = _block_bootstrap_delta(
        dir_test,
        abs_test,
        keep_primary,
        block_size=args.bootstrap_block_size,
        n_boot=args.bootstrap_iters,
        seed=args.seed,
    )

    # Diagnostics.
    auc_cal = _safe_auc(y_cal_dir, risk_dir_cal)
    ap_cal = _safe_ap(y_cal_dir, risk_dir_cal)
    auc_test = _safe_auc((1 - dir_test).astype(np.int64), risk_dir_test)
    ap_test = _safe_ap((1 - dir_test).astype(np.int64), risk_dir_test)

    _plot_curves(
        os.path.join(args.out_dir, "coverage_curves.png"),
        model_curves,
        random_curve,
        oracle_curve,
    )

    summary = {
        "val_npz": args.val_npz,
        "test_npz": args.test_npz,
        "protocol_label": args.protocol_label or None,
        "feature_keys": feature_keys,
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "risk_train_size": int(len(idx_tr)),
        "risk_cal_size": int(len(idx_cal)),
        "coverages": sorted(args.coverages, reverse=True),
        "primary_coverage": float(p_cov),
        "thresholds": {
            "directional_classifier": {str(k): float(v) for k, v in thresholds_dir.items()},
            "abs_error_regression": {str(k): float(v) for k, v in thresholds_abs.items()},
        },
        "directional_classifier": {
            "cv_auc": cv_auc,
            "best_C": best_c,
            "calibration": args.calibration,
            "cal_auc": auc_cal,
            "cal_ap": ap_cal,
            "test_auc": auc_test,
            "test_ap": ap_test,
        },
        "curves": {
            "models": model_curves,
            "random": random_curve,
            "oracle": oracle_curve,
        },
        "primary_delta_vs_random": ci_primary,
        "artifacts": {
            "coverage_plot": "coverage_curves.png",
        },
    }

    with open(os.path.join(args.out_dir, "selective_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "paper_primary_endpoints.json"), "w") as f:
        json.dump(
            {
                "protocol_label": args.protocol_label or None,
                "primary_coverage": float(p_cov),
                "delta_vs_random": ci_primary,
                "directional_classifier_curve_point": {
                    "coverage": float(p_cov),
                    "mda": float(model_curves["directional_classifier"]["mda"][sorted(args.coverages, reverse=True).index(p_cov)]),
                    "mae": float(model_curves["directional_classifier"]["mae"][sorted(args.coverages, reverse=True).index(p_cov)]),
                },
            },
            f,
            indent=2,
        )
    print(f"Saved selective evaluation artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()

