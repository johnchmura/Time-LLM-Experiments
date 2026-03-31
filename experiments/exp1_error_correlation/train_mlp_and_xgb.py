"""
Train MLP (PyTorch) and XGBoost models on Exp1 features for three targets,
comparing all features vs L1-selected features.

Targets:
- high_error_median: errors > median(errors)
- high_error_q75: errors > q75(errors)
- high_error_q90: errors > q90(errors)

Feature sets:
- all_features: all 18 NPZ features
- l1_selected: target-specific feature list from multivar_l1_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import xgboost as xgb


TARGETS = ["high_error_median", "high_error_q75", "high_error_q90"]
FEATURE_SET_ALL = "all_features"
FEATURE_SET_L1 = "l1_selected"
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


def _now_s() -> float:
    return time.time()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _finite_rows_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(axis=1)


def _compute_thresholds(errors: np.ndarray) -> Dict[str, float]:
    return {
        "median": float(np.nanmedian(errors)),
        "q75": float(np.nanquantile(errors, 0.75)),
        "q90": float(np.nanquantile(errors, 0.90)),
    }


def _eval_binary(y_true: np.ndarray, prob1: np.ndarray) -> Dict[str, float]:
    prob1 = np.asarray(prob1, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = (prob1 >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob1)),
        "pr_auc": float(average_precision_score(y_true, prob1)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "pos_rate": float(np.mean(y_true)),
    }


def _load_l1_selected(summary_path: str) -> Dict[str, List[str]]:
    with open(summary_path, "r") as f:
        d = json.load(f)
    out: Dict[str, List[str]] = {}
    targets = d.get("targets", {})
    for tname in TARGETS:
        td = targets.get(tname, {})
        sel = td.get("selected", [])
        names = [s["feature"] for s in sel if isinstance(s, dict) and "feature" in s]
        out[tname] = names
    return out


def _standardize_train_test(
    x_tr: np.ndarray, x_te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = x_tr.mean(axis=0, dtype=np.float64)
    sd = x_tr.std(axis=0, dtype=np.float64)
    sd = np.where(sd < 1e-12, 1.0, sd)
    x_tr_s = ((x_tr - mu) / sd).astype(np.float32, copy=False)
    x_te_s = ((x_te - mu) / sd).astype(np.float32, copy=False)
    return x_tr_s, x_te_s, mu.astype(np.float32), sd.astype(np.float32)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.15)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.15)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.10)

        self.out = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(self.act(self.bn1(self.fc1(x))))
        r = self.fc2(h)
        h = self.drop2(self.act(self.bn2(r + h)))
        h = self.drop3(self.act(self.bn3(self.fc3(h))))
        return self.out(h).squeeze(-1)


@dataclass(frozen=True)
class Args:
    train_npz_path: str
    test_npz_path: str
    npz_path: str
    l1_summary_path: str
    out_dir: str
    seed: int
    test_size: float
    xgb_n_estimators: int
    xgb_early_stopping: int
    xgb_n_jobs: int
    mlp_epochs: int
    mlp_batch_size: int
    mlp_lr: float
    mlp_weight_decay: float
    mlp_patience: int
    device: str
    max_eval_rows: int
    xgb_fit_verbose: bool
    no_tqdm: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Exp1: train MLP + XGBoost on feature NPZ")
    p.add_argument(
        "--train_npz_path",
        type=str,
        default="",
        help="Optional NPZ for risk-model training (if set, must also set --test_npz_path).",
    )
    p.add_argument(
        "--test_npz_path",
        type=str,
        default="",
        help="Optional NPZ for held-out evaluation (if set, must also set --train_npz_path).",
    )
    p.add_argument(
        "--npz_path",
        type=str,
        default="experiments/results/exp1_error_correlation/features.npz",
    )
    p.add_argument(
        "--l1_summary_path",
        type=str,
        default="experiments/results/exp1_error_correlation/logreg_multivar/multivar_l1_summary.json",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/results/exp1_error_correlation/ml_models",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)

    p.add_argument("--xgb_n_estimators", type=int, default=4000)
    p.add_argument("--xgb_early_stopping", type=int, default=75)
    p.add_argument("--xgb_n_jobs", type=int, default=10)

    p.add_argument("--mlp_epochs", type=int, default=80)
    p.add_argument("--mlp_batch_size", type=int, default=16384)
    p.add_argument("--mlp_lr", type=float, default=1e-3)
    p.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    p.add_argument("--mlp_patience", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--max_eval_rows",
        type=int,
        default=300_000,
        help="Cap rows used for evaluation metrics to speed up training loops (0 means all).",
    )
    p.add_argument(
        "--xgb_quiet",
        action="store_true",
        help="Disable XGBoost fit verbose output (default prints eval progress).",
    )
    p.add_argument(
        "--no_tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    a = p.parse_args()
    return Args(
        train_npz_path=a.train_npz_path,
        test_npz_path=a.test_npz_path,
        npz_path=a.npz_path,
        l1_summary_path=a.l1_summary_path,
        out_dir=a.out_dir,
        seed=a.seed,
        test_size=a.test_size,
        xgb_n_estimators=a.xgb_n_estimators,
        xgb_early_stopping=a.xgb_early_stopping,
        xgb_n_jobs=a.xgb_n_jobs,
        mlp_epochs=a.mlp_epochs,
        mlp_batch_size=a.mlp_batch_size,
        mlp_lr=a.mlp_lr,
        mlp_weight_decay=a.mlp_weight_decay,
        mlp_patience=a.mlp_patience,
        device=a.device,
        max_eval_rows=a.max_eval_rows,
        xgb_fit_verbose=not a.xgb_quiet,
        no_tqdm=a.no_tqdm,
    )


def _pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_npz_xy(
    npz_path: str, feature_keys: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    npz = np.load(npz_path, allow_pickle=False)
    errors = npz["errors"].astype(np.float64)
    if feature_keys is None:
        feature_keys = sorted([k for k in npz.files if k not in SKIP_KEYS])
    missing = [k for k in feature_keys if k not in npz.files]
    if missing:
        raise ValueError(f"Missing features in {npz_path}: {missing[:5]}")
    x_all = np.stack([npz[k].astype(np.float32) for k in feature_keys], axis=1)
    mask = _finite_rows_mask(x_all) & np.isfinite(errors)
    return x_all[mask], errors[mask], feature_keys


def _maybe_eval_subset(
    y: np.ndarray, prob: np.ndarray, seed: int, max_rows: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y) <= max_rows:
        return y, prob
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_rows, replace=False)
    return y[idx], prob[idx]


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


def _save_xgb_training_plot(model: xgb.XGBClassifier, out_path: str) -> None:
    if not hasattr(model, "evals_result"):
        return
    try:
        er = model.evals_result()
    except Exception:
        return
    val = er.get("validation_0", {})
    if not val:
        return
    _ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(8, 5))
    if "auc" in val:
        ax.plot(val["auc"], label="val AUC", color="C1")
    if "aucpr" in val:
        ax.plot(val["aucpr"], label="val AUCPR", color="C2")
    ax.set_xlabel("boosting round")
    ax.set_ylabel("metric")
    ax.set_title("XGBoost validation learning curves")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_xgboost(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    early_stopping_rounds: int,
    n_jobs: int,
    max_eval_rows: int,
    fit_verbose: bool,
) -> Tuple[Dict[str, Any], xgb.XGBClassifier]:
    x_tr2, x_va, y_tr2, y_va = train_test_split(
        x_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr
    )
    pos = float(np.sum(y_tr2 == 1))
    neg = float(np.sum(y_tr2 == 0))
    spw = (neg / max(pos, 1.0)) if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric=["auc", "aucpr"],
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
        scale_pos_weight=spw,
        early_stopping_rounds=early_stopping_rounds,
    )

    model.fit(
        x_tr2,
        y_tr2,
        eval_set=[(x_va, y_va)],
        verbose=fit_verbose,
    )

    prob = model.predict_proba(x_te)[:, 1]
    y_s, p_s = _maybe_eval_subset(y_te, prob, seed=seed, max_rows=max_eval_rows)
    metrics = _eval_binary(y_s, p_s)
    metrics["best_iteration"] = int(getattr(model, "best_iteration", -1))
    metrics["scale_pos_weight"] = float(spw)
    return metrics, model


def _torch_auc_metrics(y_true: np.ndarray, prob1: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, prob1)),
        "pr_auc": float(average_precision_score(y_true, prob1)),
    }


def train_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    monitor: str,
    max_eval_rows: int,
    show_progress: bool,
    epoch_desc: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], nn.Module]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_tr2, x_va, y_tr2, y_va = train_test_split(
        x_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr
    )

    x_tr_s, x_va_s, mu, sd = _standardize_train_test(x_tr2, x_va)
    x_te_s = ((x_te - mu) / sd).astype(np.float32, copy=False)

    x_tr_t = torch.from_numpy(x_tr_s)
    y_tr_t = torch.from_numpy(y_tr2.astype(np.float32))
    x_va_t = torch.from_numpy(x_va_s)
    y_va_t = torch.from_numpy(y_va.astype(np.float32))
    x_te_t = torch.from_numpy(x_te_s)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = ResidualMLP(in_dim=x_tr.shape[1]).to(device)

    pos = float(np.sum(y_tr2 == 1))
    neg = float(np.sum(y_tr2 == 0))
    pw = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1.0
    best_state = None
    bad = 0
    history: List[Dict[str, Any]] = []

    def _predict_prob(xb: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(xb.to(device)).float().cpu().numpy()
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

    epoch_iter = range(1, epochs + 1)
    if show_progress:
        epoch_iter = tqdm(
            epoch_iter,
            desc=epoch_desc or "MLP epochs",
            leave=False,
            unit="ep",
        )

    for ep in epoch_iter:
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * int(len(xb))
            n += int(len(xb))

        prob_va = _predict_prob(x_va_t)
        y_va_np = y_va.astype(np.int64)
        y_s, p_s = _maybe_eval_subset(y_va_np, prob_va, seed=seed + ep, max_rows=max_eval_rows)
        m2 = _torch_auc_metrics(y_s, p_s)
        score = m2["pr_auc"] if monitor == "pr_auc" else m2["roc_auc"]

        history.append(
            {
                "epoch": ep,
                "train_loss": float(total_loss / max(n, 1)),
                "val_roc_auc": float(m2["roc_auc"]),
                "val_pr_auc": float(m2["pr_auc"]),
            }
        )

        if score > best_score + 1e-4:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    prob_te = _predict_prob(x_te_t)
    y_te_np = y_te.astype(np.int64)
    y_s, p_s = _maybe_eval_subset(y_te_np, prob_te, seed=seed, max_rows=max_eval_rows)
    metrics = _eval_binary(y_s, p_s)

    cfg = {
        "epochs_ran": int(history[-1]["epoch"]) if history else 0,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "patience": int(patience),
        "monitor": monitor,
        "pos_weight": float(pw),
        "device": str(device),
    }
    return metrics, {"config": cfg, "history": history}, model


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_dir)
    device = _pick_device(args.device)

    using_explicit_split = bool(args.train_npz_path) or bool(args.test_npz_path)
    if using_explicit_split:
        if not args.train_npz_path or not args.test_npz_path:
            raise ValueError("When using explicit NPZ split mode, set both --train_npz_path and --test_npz_path.")
        x_train_all, errors_train, feature_keys = _load_npz_xy(args.train_npz_path, feature_keys=None)
        x_test_all, errors_test, _ = _load_npz_xy(args.test_npz_path, feature_keys=feature_keys)
        thresholds = _compute_thresholds(errors_train)
        y_by_target_train = {
            "high_error_median": (errors_train > thresholds["median"]).astype(np.int64),
            "high_error_q75": (errors_train > thresholds["q75"]).astype(np.int64),
            "high_error_q90": (errors_train > thresholds["q90"]).astype(np.int64),
        }
        y_by_target_test = {
            "high_error_median": (errors_test > thresholds["median"]).astype(np.int64),
            "high_error_q75": (errors_test > thresholds["q75"]).astype(np.int64),
            "high_error_q90": (errors_test > thresholds["q90"]).astype(np.int64),
        }
        n_rows_all = int(len(x_train_all) + len(x_test_all))
    else:
        x_all, errors, feature_keys = _load_npz_xy(args.npz_path, feature_keys=None)
        thresholds = _compute_thresholds(errors)
        y_by_target = {
            "high_error_median": (errors > thresholds["median"]).astype(np.int64),
            "high_error_q75": (errors > thresholds["q75"]).astype(np.int64),
            "high_error_q90": (errors > thresholds["q90"]).astype(np.int64),
        }
        n_rows_all = int(len(x_all))

    l1_sel = _load_l1_selected(args.l1_summary_path)

    summary: Dict[str, Any] = {
        "npz_path": args.npz_path,
        "train_npz_path": args.train_npz_path,
        "test_npz_path": args.test_npz_path,
        "explicit_split_mode": using_explicit_split,
        "l1_summary_path": args.l1_summary_path,
        "out_dir": args.out_dir,
        "seed": args.seed,
        "test_size": args.test_size,
        "thresholds": thresholds,
        "n_rows": n_rows_all,
        "feature_keys_all": feature_keys,
        "feature_sets": [FEATURE_SET_ALL, FEATURE_SET_L1],
        "targets": {},
    }

    show_tqdm = not args.no_tqdm
    target_loop = tqdm(TARGETS, desc="Targets (XGB+MLP per feature set)", unit="target") if show_tqdm else TARGETS

    for tname in target_loop:
        if using_explicit_split:
            x_tr_all = x_train_all
            x_te_all = x_test_all
            y_tr = y_by_target_train[tname]
            y_te = y_by_target_test[tname]
            y = np.concatenate([y_tr, y_te])
        else:
            y = y_by_target[tname]
            x_tr_idx, x_te_idx = train_test_split(
                np.arange(len(y)),
                test_size=args.test_size,
                random_state=args.seed,
                stratify=y,
            )
            x_tr_all = x_all[x_tr_idx]
            x_te_all = x_all[x_te_idx]
            y_tr = y[x_tr_idx]
            y_te = y[x_te_idx]

        sel_names = l1_sel.get(tname, [])
        sel_names = [n for n in sel_names if n in feature_keys]
        sel_idx = [feature_keys.index(n) for n in sel_names]
        if len(sel_idx) == 0:
            sel_idx = list(range(len(feature_keys)))
            sel_names = feature_keys[:]

        x_tr_l1 = x_tr_all[:, sel_idx]
        x_te_l1 = x_te_all[:, sel_idx]

        target_out: Dict[str, Any] = {
            "pos_rate": float(np.mean(y)),
            "feature_sets": {
                FEATURE_SET_ALL: {"n_features": int(x_tr_all.shape[1]), "feature_keys": feature_keys},
                FEATURE_SET_L1: {"n_features": int(x_tr_l1.shape[1]), "feature_keys": sel_names},
            },
            "models": {},
        }

        fset_pairs = list(
            {
                FEATURE_SET_ALL: (x_tr_all, x_te_all, feature_keys),
                FEATURE_SET_L1: (x_tr_l1, x_te_l1, sel_names),
            }.items()
        )
        fset_loop = (
            tqdm(fset_pairs, desc=f"{tname} feature sets", leave=False, unit="set")
            if show_tqdm
            else fset_pairs
        )

        for fset, (x_tr, x_te, fkeys) in fset_loop:
            plot_dir = os.path.join(args.out_dir, "plots", tname, fset)
            _ensure_dir(plot_dir)
            xgb_plot_rel = os.path.join("plots", tname, fset, "xgb_training.png")
            mlp_plot_rel = os.path.join("plots", tname, fset, "mlp_training.png")
            xgb_plot_path = os.path.join(args.out_dir, xgb_plot_rel)
            mlp_plot_path = os.path.join(args.out_dir, mlp_plot_rel)

            # XGBoost
            t0 = _now_s()
            xgb_metrics, xgb_model = train_xgboost(
                x_tr,
                y_tr,
                x_te,
                y_te,
                seed=args.seed,
                n_estimators=args.xgb_n_estimators,
                early_stopping_rounds=args.xgb_early_stopping,
                n_jobs=args.xgb_n_jobs,
                max_eval_rows=args.max_eval_rows,
                fit_verbose=args.xgb_fit_verbose,
            )
            xgb_path = os.path.join(args.out_dir, f"xgb_{tname}_{fset}.json")
            xgb_model.save_model(xgb_path)
            _save_xgb_training_plot(xgb_model, xgb_plot_path)

            # MLP
            monitor = "pr_auc" if tname == "high_error_q90" else "roc_auc"
            mlp_t0 = _now_s()
            mlp_metrics, mlp_trainlog, mlp_model = train_mlp(
                x_tr,
                y_tr,
                x_te,
                y_te,
                seed=args.seed,
                device=device,
                epochs=args.mlp_epochs,
                batch_size=args.mlp_batch_size,
                lr=args.mlp_lr,
                weight_decay=args.mlp_weight_decay,
                patience=args.mlp_patience,
                monitor=monitor,
                max_eval_rows=args.max_eval_rows,
                show_progress=show_tqdm,
                epoch_desc=f"MLP {tname} {fset}",
            )
            mlp_path = os.path.join(args.out_dir, f"mlp_{tname}_{fset}.pt")
            torch.save({"state_dict": mlp_model.state_dict(), "feature_keys": fkeys}, mlp_path)

            trainlog_path = os.path.join(args.out_dir, f"mlp_{tname}_{fset}_trainlog.json")
            with open(trainlog_path, "w") as f:
                json.dump(mlp_trainlog, f, indent=2)

            hist = mlp_trainlog.get("history", [])
            _save_mlp_training_plot(hist, mlp_plot_path)

            target_out["models"][fset] = {
                "xgboost": {
                    "metrics": xgb_metrics,
                    "model_path": os.path.basename(xgb_path),
                    "plot_path": xgb_plot_rel.replace("\\", "/"),
                    "train_s": float(_now_s() - t0),
                    "params": {
                        "n_estimators": int(args.xgb_n_estimators),
                        "early_stopping_rounds": int(args.xgb_early_stopping),
                        "n_jobs": int(args.xgb_n_jobs),
                    },
                },
                "mlp": {
                    "metrics": mlp_metrics,
                    "model_path": os.path.basename(mlp_path),
                    "trainlog_path": os.path.basename(trainlog_path),
                    "plot_path": mlp_plot_rel.replace("\\", "/"),
                    "train_s": float(_now_s() - mlp_t0),
                    "params": {
                        "epochs": int(args.mlp_epochs),
                        "batch_size": int(args.mlp_batch_size),
                        "lr": float(args.mlp_lr),
                        "weight_decay": float(args.mlp_weight_decay),
                        "patience": int(args.mlp_patience),
                        "device": str(device),
                    },
                },
            }

        summary["targets"][tname] = target_out

    out_path = os.path.join(args.out_dir, "ml_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

