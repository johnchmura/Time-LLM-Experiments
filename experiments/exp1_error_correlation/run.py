"""
Experiment 1: Do LLM internal signals correlate with prediction error?

Architecture note:
Time-LLM uses a linear head (FlattenHead) over flattened patch activations from
the LLM's last hidden state -- it does NOT use autoregressive next-token generation.
There are no vocabulary logits for the forecast. "Output features" are implemented
as head-level surrogates (pre-head activation norms, prediction deltas). Each
y_hat_t is a linear function of ALL patch positions; the patch-to-horizon alignment
(t -> p) used here is a heuristic attribution.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import TimeLLM
from data_provider.data_factory import data_dict
from utils.tools import load_content

FEATURE_NAMES = [
    'final_layer_norm',
    'mean_layer_norm',
    'layer_norm_var',
    'centroid_distance',
    'knn_distance',
    'reprog_norm',
    'reprog_dim_var',
    'reprog_centroid_distance',
    'reprog_knn_distance',
    'reprog_entropy',
    'reprog_max_attn',
    'reprog_entropy_head_var',
    'pre_head_norm',
    'temporal_h_delta',
    'pred_delta',
    'llm_attn_entropy',
    'llm_attn_max_weight',
    'llm_attn_entropy_head_var',
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Exp 1: LLM signal-error correlation')

    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to model checkpoint file')
    p.add_argument('--output_dir', type=str,
                   default='experiments/results/exp1_error_correlation')

    # Data (ETTh1 defaults matching scripts/TimeLLM_ETTh1.sh)
    p.add_argument('--data', type=str, default='ETTh1')
    p.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    p.add_argument('--data_path', type=str, default='ETTh1.csv')
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--freq', type=str, default='h')
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--percent', type=int, default=100)
    p.add_argument('--seasonal_patterns', type=str, default='Monthly')

    # Model (ETTh1 training defaults)
    p.add_argument('--task_name', type=str, default='long_term_forecast')
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--enc_in', type=int, default=7)
    p.add_argument('--dec_in', type=int, default=7)
    p.add_argument('--c_out', type=int, default=7)
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--d_ff', type=int, default=128)
    p.add_argument('--factor', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--llm_model', type=str, default='LLAMA')
    p.add_argument('--llm_dim', type=int, default=4096)
    p.add_argument('--llm_layers', type=int, default=16)
    p.add_argument('--prompt_domain', type=int, default=0)
    p.add_argument('--num_tokens', type=int, default=1000)

    # Experiment
    p.add_argument('--batch_size', type=int, default=4,
                   help='Batch size for inference (reduce if OOM)')
    p.add_argument('--max_train_samples', type=int, default=5000,
                   help='Cap for training reference set (sampled uniformly across channels)')
    p.add_argument('--knn_k', type=int, default=5)
    p.add_argument('--alignment', type=str, default='spread',
                   choices=['spread', 'last_patch'],
                   help='Patch-to-horizon mapping strategy')
    p.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    p.add_argument('--splits', type=str, default='',
                   help='Comma-separated eval splits (overrides --split), e.g. "val,test"')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--output_basename', type=str, default='features',
                   help='Per-split NPZ prefix: {output_basename}_{split}.npz')
    p.add_argument('--no_plots', action='store_true',
                   help='Skip feature-vs-error plots (faster extraction-only runs)')
    p.add_argument('--with_checkpoint_hash', action='store_true',
                   help='Compute SHA256 of checkpoint for provenance (can be slow for large files)')
    p.add_argument(
        '--strict_gap_steps',
        type=int,
        default=0,
        help='If >0, for val/test keep only rows with dataset_row_index >= strict_gap_steps '
             '(use seq_len for strict no-overlap boundary protocol).',
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device(requested):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def map_horizon_to_patch(t, pred_len, patch_nums, alignment='spread'):
    if alignment == 'last_patch':
        return patch_nums - 1
    return min(t * patch_nums // pred_len, patch_nums - 1)


def entropy(probs, dim=-1, eps=1e-10):
    """Shannon entropy along *dim* for a probability tensor."""
    p = probs.float().clamp(min=eps)
    return -(p * p.log()).sum(dim=dim)


def build_dataset(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    return Data(
        root_path=args.root_path, data_path=args.data_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target,
        timeenc=timeenc, freq=args.freq,
        percent=args.percent, seasonal_patterns=args.seasonal_patterns,
    )


def parse_eval_splits(args):
    if args.splits.strip():
        raw = [x.strip() for x in args.splits.split(',') if x.strip()]
    else:
        raw = [args.split]
    allowed = {'val', 'test', 'train'}
    bad = [s for s in raw if s not in allowed]
    if bad:
        raise ValueError(f"Unsupported split(s): {bad}. Allowed: {sorted(allowed)}")
    # Preserve order while deduplicating.
    out = []
    seen = set()
    for s in raw:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def maybe_sha256(path, enabled):
    if not enabled:
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def git_commit_hash():
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _split_date_index(args, split_flag, ds):
    """
    Return datetime index for this split's `data_x` rows if reconstructable.
    For datasets without a direct `date` column, returns None.
    """
    csv_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
    except Exception:
        return None
    if 'date' not in df.columns:
        return None

    cls = ds.__class__.__name__
    type_map = {'train': 0, 'val': 1, 'test': 2}
    if split_flag not in type_map:
        return None
    st = type_map[split_flag]

    if cls == 'Dataset_ETT_hour':
        b1s = [0, 12 * 30 * 24 - args.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - args.seq_len]
        b2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        b1, b2 = b1s[st], b2s[st]
        if st == 0:
            b2 = (b2 - args.seq_len) * args.percent // 100 + args.seq_len
    elif cls == 'Dataset_ETT_minute':
        b1s = [0, 12 * 30 * 24 * 4 - args.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - args.seq_len]
        b2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        b1, b2 = b1s[st], b2s[st]
        if st == 0:
            b2 = (b2 - args.seq_len) * args.percent // 100 + args.seq_len
    elif cls == 'Dataset_Custom':
        n = len(df)
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        b1s = [0, n_train - args.seq_len, n - n_test - args.seq_len]
        b2s = [n_train, n_train + n_val, n]
        b1, b2 = b1s[st], b2s[st]
        if st == 0:
            b2 = (b2 - args.seq_len) * args.percent // 100 + args.seq_len
    else:
        return None

    dates = pd.to_datetime(df['date'].iloc[b1:b2], errors='coerce').to_numpy()
    if dates.shape[0] == 0:
        return None
    return dates


def load_model(args, device):
    args.content = load_content(args)
    args.use_eager_attention = True
    model = TimeLLM.Model(args).float().to(device)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _autocast_ctx(device):
    """Autocast context matching the bf16 cast inside patch_embedding."""
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _collate(samples):
    """Stack a list of (x, y, xm, ym) tuples into batch tensors."""
    def _to_tensor(x):
        # Dataset_* in this repo returns numpy arrays; convert them so torch.stack works.
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return torch.as_tensor(x)

    xs = [_to_tensor(s[0]) for s in samples]
    ys = [_to_tensor(s[1]) for s in samples]
    xms = [_to_tensor(s[2]) for s in samples]
    yms = [_to_tensor(s[3]) for s in samples]
    return (
        torch.stack(xs),
        torch.stack(ys),
        torch.stack(xms),
        torch.stack(yms),
    )


# ---------------------------------------------------------------------------
# Phase A: training reference statistics
# ---------------------------------------------------------------------------

def collect_train_references(model, args, device):
    """
    Collect final-layer hidden-state and reprogramming-output centroids plus
    k-NN indices from a balanced subset of the training split.

    Reference vectors are the *mean across patches* of each sample's
    final-layer hidden state (d_ff-dim) and reprogramming output (d_llm-dim).
    """
    ds = build_dataset(args, 'train')
    tot_len = ds.tot_len
    enc_in = ds.enc_in
    per_ch = args.max_train_samples // enc_in

    indices = []
    for fid in range(enc_in):
        start = fid * tot_len
        step = max(1, tot_len // per_ch)
        indices.extend(list(range(start, start + tot_len, step))[:per_ch])
    indices = sorted(indices)

    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False,
                        collate_fn=_collate)

    h_by_ch = {f: [] for f in range(enc_in)}
    r_by_ch = {f: [] for f in range(enc_in)}

    ptr = 0
    print(f"Collecting train references: {len(indices)} samples ...")
    with torch.no_grad(), _autocast_ctx(device):
        for bx, by, bxm, bym in tqdm(loader, desc='Train refs'):
            B = bx.shape[0]
            bx = bx.float().to(device)
            bxm = bxm.float().to(device)
            bym = bym.float().to(device)
            di = torch.zeros_like(by[:, -args.pred_len:, :]).float()
            di = torch.cat([by[:, :args.label_len, :], di], dim=1).float().to(device)

            aux = model(bx, bxm, di, bym, return_aux=True)

            h_last = aux['h_last'][:, 0, :, :].float().mean(dim=-1).cpu().numpy()
            r_out = aux['reprog_out'][:, 0, :, :].float().mean(dim=1).cpu().numpy()

            for j in range(B):
                orig_idx = indices[ptr + j]
                fid = orig_idx // tot_len
                h_by_ch[fid].append(h_last[j])
                r_by_ch[fid].append(r_out[j])
            ptr += B

    centroids, r_centroids = {}, {}
    knn, r_knn = {}, {}
    for fid in range(enc_in):
        h_arr = np.stack(h_by_ch[fid])
        r_arr = np.stack(r_by_ch[fid])
        centroids[fid] = h_arr.mean(axis=0)
        r_centroids[fid] = r_arr.mean(axis=0)
        nn_h = NearestNeighbors(n_neighbors=min(args.knn_k, len(h_arr)),
                                metric='euclidean')
        nn_h.fit(h_arr)
        knn[fid] = nn_h
        nn_r = NearestNeighbors(n_neighbors=min(args.knn_k, len(r_arr)),
                                metric='euclidean')
        nn_r.fit(r_arr)
        r_knn[fid] = nn_r

    return dict(centroids=centroids, reprog_centroids=r_centroids,
                knn=knn, reprog_knn=r_knn)


# ---------------------------------------------------------------------------
# Phase B: evaluation feature extraction
# ---------------------------------------------------------------------------

def extract_batch_features(aux, batch_y, batch_ym, feat_ids, args, refs, patch_nums):
    """
    Compute all scalar features for one batch.

    Returns
    -------
    features : dict  {name: np.array [B * pred_len]}
    errors   : np.array [B * pred_len]
    targets  : dict  {name: np.array [B * pred_len]}
    metadata : dict  {name: np.array [...]} flattened to [B * pred_len] except
                      `time_features` with shape [B * pred_len, D_time]
    """
    pred = aux['pred'].detach().cpu().float()       # [B, pred_len, 1]
    h_layers = aux['h_layers']                      # list of [B, 1, d_ff, P]
    h_last = aux['h_last']                          # [B, 1, d_ff, P]
    pre_head = aux['pre_head']                      # [B, 1, d_ff, P]
    reprog_out = aux['reprog_out']                   # [B, 1, P, d_llm]
    reprog_attn_t = aux['reprog_attn']               # [B, H_repr, P, S]
    llm_attns = aux['llm_attns']                     # list of [B, H_llm, seq, seq]
    prompt_len = aux['prompt_len']

    B = pred.shape[0]
    pred_len = args.pred_len
    P = patch_nums

    f_dim = -1 if args.features == 'MS' else 0
    true = batch_y[:, -pred_len:, f_dim:].float()   # [B, pred_len, 1]
    errors = torch.abs(true - pred).squeeze(-1).numpy()  # [B, pred_len]

    patch_map = np.array([map_horizon_to_patch(t, pred_len, P, args.alignment)
                          for t in range(pred_len)])

    # ── per-layer norms [n_layers, B, P] ──
    layer_norms = np.stack([
        hl[:, 0, :, :].float().norm(dim=1).numpy() for hl in h_layers
    ])
    lnm = layer_norms[:, :, patch_map]              # [n_layers, B, pred_len]

    features = {}
    features['final_layer_norm'] = lnm[-1].ravel()
    features['mean_layer_norm'] = lnm.mean(axis=0).ravel()
    features['layer_norm_var'] = lnm.var(axis=0).ravel()

    # ── pre-head norm ──
    ph_norms = pre_head[:, 0, :, :].float().norm(dim=1).numpy()  # [B, P]
    features['pre_head_norm'] = ph_norms[:, patch_map].ravel()

    # ── reprogramming output features ──
    rp = reprog_out[:, 0, :, :].float().numpy()      # [B, P, d_llm]
    features['reprog_norm'] = np.linalg.norm(rp, axis=-1)[:, patch_map].ravel()
    features['reprog_dim_var'] = rp.var(axis=-1)[:, patch_map].ravel()

    # ── reprogramming attention features ──
    ra = reprog_attn_t.float()                        # [B, H, P, S]
    ra_ent = entropy(ra, dim=-1).numpy()              # [B, H, P]
    ra_mx = ra.max(dim=-1).values.numpy()
    features['reprog_entropy'] = ra_ent.mean(axis=1)[:, patch_map].ravel()
    features['reprog_max_attn'] = ra_mx.mean(axis=1)[:, patch_map].ravel()
    features['reprog_entropy_head_var'] = ra_ent.var(axis=1)[:, patch_map].ravel()

    # ── temporal hidden-state delta ──
    h_np = h_last[:, 0, :, :].float().numpy()        # [B, d_ff, P]
    h_m = h_np[:, :, patch_map]                       # [B, d_ff, pred_len]
    h_diff = np.diff(h_m, axis=-1, prepend=h_m[:, :, :1])
    h_delta = np.linalg.norm(h_diff, axis=1)          # [B, pred_len]
    h_delta[:, 0] = 0.0
    features['temporal_h_delta'] = h_delta.ravel()

    # ── prediction delta ──
    pred_np = pred.squeeze(-1).numpy()
    p_diff = np.abs(np.diff(pred_np, axis=-1, prepend=pred_np[:, :1]))
    p_diff[:, 0] = 0.0
    features['pred_delta'] = p_diff.ravel()

    # ── directional accuracy (vs_prev_step) ──
    true_np = true.squeeze(-1).detach().cpu().numpy()
    pred_step = np.diff(pred_np, axis=-1, prepend=pred_np[:, :1])
    true_step = np.diff(true_np, axis=-1, prepend=true_np[:, :1])
    direction_correct = (np.sign(pred_step) == np.sign(true_step)).astype(np.int64)
    targets = {'directional_correct': direction_correct.ravel()}
    true_prev = np.concatenate([true_np[:, :1], true_np[:, :-1]], axis=1)
    pred_prev = np.concatenate([pred_np[:, :1], pred_np[:, :-1]], axis=1)
    metadata = {
        'y_true': true_np.ravel(),
        'y_pred': pred_np.ravel(),
        'y_prev': true_prev.ravel(),
        'pred_prev': pred_prev.ravel(),
        'direction_true': np.sign(true_step).astype(np.int8).ravel(),
        'direction_pred': np.sign(pred_step).astype(np.int8).ravel(),
        'error_abs': errors.ravel(),
    }
    if isinstance(batch_ym, torch.Tensor):
        tmark = batch_ym[:, -pred_len:, :].detach().cpu().float().numpy()
        metadata['time_features'] = tmark.reshape(B * pred_len, -1)

    # ── sample-level geometry (broadcast to pred_len) ──
    h_mean = h_np.mean(axis=-1)                       # [B, d_ff]
    r_mean = rp.mean(axis=1)                           # [B, d_llm]

    cdist = np.zeros(B)
    kdist = np.zeros(B)
    rcdist = np.zeros(B)
    rkdist = np.zeros(B)

    for fid in np.unique(feat_ids):
        m = feat_ids == fid
        if fid in refs['centroids']:
            cdist[m] = np.linalg.norm(h_mean[m] - refs['centroids'][fid], axis=1)
        if fid in refs['knn']:
            d, _ = refs['knn'][fid].kneighbors(h_mean[m])
            kdist[m] = d.mean(axis=1)
        if fid in refs['reprog_centroids']:
            rcdist[m] = np.linalg.norm(r_mean[m] - refs['reprog_centroids'][fid], axis=1)
        if fid in refs['reprog_knn']:
            d, _ = refs['reprog_knn'][fid].kneighbors(r_mean[m])
            rkdist[m] = d.mean(axis=1)

    features['centroid_distance'] = np.repeat(cdist, pred_len)
    features['knn_distance'] = np.repeat(kdist, pred_len)
    features['reprog_centroid_distance'] = np.repeat(rcdist, pred_len)
    features['reprog_knn_distance'] = np.repeat(rkdist, pred_len)

    # ── LLM self-attention features ──
    if llm_attns:
        ent_l, mx_l = [], []
        for at in llm_attns:
            af = at.float()
            aq = af[:, :, prompt_len:prompt_len + P, :]  # [B, H, P, seq]
            ent_l.append(entropy(aq, dim=-1).numpy())     # [B, H, P]
            mx_l.append(aq.max(dim=-1).values.numpy())
        ent_all = np.stack(ent_l)                         # [L, B, H, P]
        mx_all = np.stack(mx_l)
        features['llm_attn_entropy'] = ent_all.mean(axis=(0, 2))[:, patch_map].ravel()
        features['llm_attn_max_weight'] = mx_all.mean(axis=(0, 2))[:, patch_map].ravel()
        features['llm_attn_entropy_head_var'] = (
            ent_all.var(axis=2).mean(axis=0)[:, patch_map].ravel()
        )
    else:
        n = B * pred_len
        features['llm_attn_entropy'] = np.full(n, np.nan)
        features['llm_attn_max_weight'] = np.full(n, np.nan)
        features['llm_attn_entropy_head_var'] = np.full(n, np.nan)

    return features, errors.ravel(), targets, metadata


def run_evaluation(model, args, device, refs, split_flag):
    """Iterate over the eval split, extract features and errors."""
    ds = build_dataset(args, split_flag)
    split_dates = _split_date_index(args, split_flag, ds)
    tot_len = ds.tot_len
    all_indices = np.arange(len(ds), dtype=np.int64)
    if args.strict_gap_steps > 0 and split_flag in {'val', 'test'}:
        row_idx = all_indices % tot_len
        all_indices = all_indices[row_idx >= int(args.strict_gap_steps)]
    if len(all_indices) == 0:
        raise ValueError(
            f"No samples remain for split={split_flag} after strict_gap_steps={args.strict_gap_steps}. "
            f"tot_len={tot_len}"
        )
    subset = Subset(ds, all_indices.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    patch_nums = model.patch_nums

    accum = {k: [] for k in FEATURE_NAMES}
    all_errors, all_ch = [], []
    all_targets = {}
    all_meta = {
        'sample_id': [],
        'dataset_row_index': [],
        'horizon_t': [],
        'y_true': [],
        'y_pred': [],
        'y_prev': [],
        'pred_prev': [],
        'direction_true': [],
        'direction_pred': [],
        'time_features': [],
    }

    idx = 0
    ptr = 0
    with torch.no_grad(), _autocast_ctx(device):
        for bx, by, bxm, bym in tqdm(loader, desc='Eval'):
            B = bx.shape[0]
            batch_orig_idx = all_indices[ptr:ptr + B]
            ptr += B
            fids = (batch_orig_idx // tot_len).astype(np.int64)
            idx += B

            bx = bx.float().to(device)
            bxm = bxm.float().to(device)
            bym_d = bym.float().to(device)
            di = torch.zeros_like(by[:, -args.pred_len:, :]).float()
            di = torch.cat([by[:, :args.label_len, :], di], dim=1).float().to(device)

            aux = model(bx, bxm, di, bym_d, return_aux=True)
            feats, errs, targets, meta = extract_batch_features(
                aux, by, bym, fids, args, refs, patch_nums
            )

            for k in FEATURE_NAMES:
                accum[k].append(feats[k])
            all_errors.append(errs)
            ch_rep = np.repeat(fids, args.pred_len)
            all_ch.append(ch_rep)
            for tk, tv in targets.items():
                all_targets.setdefault(tk, []).append(tv)

            # Provenance per flattened horizon row.
            sample_ids = np.arange(idx - B, idx)
            all_meta['sample_id'].append(np.repeat(sample_ids, args.pred_len))
            all_meta['dataset_row_index'].append(np.repeat(batch_orig_idx % tot_len, args.pred_len))
            all_meta['horizon_t'].append(np.tile(np.arange(args.pred_len), B))
            all_meta['y_true'].append(meta['y_true'])
            all_meta['y_pred'].append(meta['y_pred'])
            all_meta['y_prev'].append(meta['y_prev'])
            all_meta['pred_prev'].append(meta['pred_prev'])
            all_meta['direction_true'].append(meta['direction_true'])
            all_meta['direction_pred'].append(meta['direction_pred'])
            if 'time_features' in meta:
                all_meta['time_features'].append(meta['time_features'])

    for k in FEATURE_NAMES:
        accum[k] = np.concatenate(accum[k])
    for tk in list(all_targets.keys()):
        all_targets[tk] = np.concatenate(all_targets[tk])
    out_meta = {}
    for mk, mv in all_meta.items():
        if not mv:
            continue
        if mk == 'time_features':
            out_meta[mk] = np.concatenate(mv, axis=0)
        else:
            out_meta[mk] = np.concatenate(mv)
    if split_dates is not None and 'dataset_row_index' in out_meta and 'horizon_t' in out_meta:
        # Map each flattened horizon row to target timestamp index in split-local data_x.
        ts_idx = out_meta['dataset_row_index'].astype(np.int64) + args.seq_len + out_meta['horizon_t'].astype(np.int64)
        ts_idx = np.clip(ts_idx, 0, len(split_dates) - 1)
        out_meta['timestamp'] = split_dates[ts_idx].astype('datetime64[s]').astype(str)
    return (
        accum,
        np.concatenate(all_errors),
        np.concatenate(all_ch),
        all_targets,
        out_meta,
        int(len(all_indices)),
        int(len(ds)),
    )


# ---------------------------------------------------------------------------
# Phase C: correlation analysis
# ---------------------------------------------------------------------------

def _safe_corr(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12:
        return dict(pearson_r=float('nan'), pearson_p=float('nan'),
                    spearman_r=float('nan'), spearman_p=float('nan'))
    pr, pp = scipy_stats.pearsonr(x[valid], y[valid])
    sr, sp = scipy_stats.spearmanr(x[valid], y[valid])
    return dict(pearson_r=float(pr), pearson_p=float(pp),
                spearman_r=float(sr), spearman_p=float(sp))


def compute_correlations(features, errors, channel_ids, enc_in):
    metrics = {'pooled': {}, 'per_channel': {}}
    for fname in FEATURE_NAMES:
        fv = features[fname]
        metrics['pooled'][fname] = _safe_corr(fv, errors)
        for fid in range(enc_in):
            m = channel_ids == fid
            ch = str(fid)
            if ch not in metrics['per_channel']:
                metrics['per_channel'][ch] = {}
            metrics['per_channel'][ch][fname] = _safe_corr(fv[m], errors[m])
    return metrics


# ---------------------------------------------------------------------------
# Phase D: plots
# ---------------------------------------------------------------------------

def make_plots(features, errors, output_dir, figure_subdir='figures'):
    fig_dir = os.path.join(output_dir, figure_subdir)
    os.makedirs(fig_dir, exist_ok=True)

    for fname in FEATURE_NAMES:
        fv = features[fname]
        valid = np.isfinite(fv) & np.isfinite(errors)
        if valid.sum() < 20:
            continue
        fv_v, ev_v = fv[valid], errors[valid]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # -- scatter (subsample for readability) --
        n = len(fv_v)
        if n > 10_000:
            sel = np.random.default_rng(42).choice(n, 10_000, replace=False)
            xs, ys = fv_v[sel], ev_v[sel]
        else:
            xs, ys = fv_v, ev_v
        ax1.scatter(xs, ys, alpha=0.08, s=3, rasterized=True)
        ax1.set_xlabel(fname)
        ax1.set_ylabel('|error|')
        ax1.set_title(f'{fname} vs error')

        # -- binned means (deciles) --
        try:
            edges = np.unique(np.percentile(fv_v, np.linspace(0, 100, 11)))
            if len(edges) > 1:
                dig = np.digitize(fv_v, edges[1:-1])
                bm_f, bm_e, bm_se = [], [], []
                for b in range(len(edges) - 1):
                    mb = dig == b
                    if mb.sum() > 0:
                        bm_f.append(fv_v[mb].mean())
                        bm_e.append(ev_v[mb].mean())
                        bm_se.append(ev_v[mb].std() / max(np.sqrt(mb.sum()), 1))
                ax2.errorbar(bm_f, bm_e, yerr=bm_se, fmt='o-', capsize=3,
                             markersize=5, linewidth=1.5)
                ax2.set_xlabel(f'{fname} (bin mean)')
                ax2.set_ylabel('mean |error|')
                ax2.set_title(f'{fname} — binned means')
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'{fname}.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved {len(FEATURE_NAMES)} plots to {fig_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    eval_splits = parse_eval_splits(args)
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        cfg = dict(vars(args))
        cfg['resolved_splits'] = eval_splits
        json.dump(cfg, f, indent=2)

    # ── load model ──
    model = load_model(args, device)
    print(f"Model loaded  |  patch_nums={model.patch_nums}  d_ff={args.d_ff}  "
          f"llm_layers={args.llm_layers}  alignment={args.alignment}")

    # ── Phase A ──
    refs = collect_train_references(model, args, device)

    # Save reference statistics for one-shot extraction provenance.
    ref_out = {}
    for fid, c in refs['centroids'].items():
        ref_out[f'centroid_{fid}'] = c
    for fid, c in refs['reprog_centroids'].items():
        ref_out[f'reprog_centroid_{fid}'] = c
    np.savez_compressed(os.path.join(args.output_dir, 'train_reference_stats.npz'), **ref_out)

    provenance = {
        'checkpoint_path': os.path.abspath(args.checkpoint),
        'checkpoint_sha256': maybe_sha256(args.checkpoint, args.with_checkpoint_hash),
        'git_commit': git_commit_hash(),
        'feature_names': FEATURE_NAMES,
        'resolved_splits': eval_splits,
        'args': vars(args),
    }
    with open(os.path.join(args.output_dir, 'provenance.json'), 'w') as f:
        json.dump(provenance, f, indent=2)

    data_contract = {'splits': {}, 'feature_names': FEATURE_NAMES}

    for split_flag in eval_splits:
        print(f"\n=== Evaluating split: {split_flag} ===")
        (
            features,
            errors,
            channel_ids,
            targets,
            metadata,
            n_used,
            n_total,
        ) = run_evaluation(
            model, args, device, refs, split_flag
        )

        metrics = compute_correlations(features, errors, channel_ids, args.enc_in)
        with open(os.path.join(args.output_dir, f'metrics_{split_flag}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        out_npz = os.path.join(args.output_dir, f'{args.output_basename}_{split_flag}.npz')
        payload = dict(
            errors=errors,
            channel_ids=channel_ids,
            sample_id=metadata.get('sample_id'),
            dataset_row_index=metadata.get('dataset_row_index'),
            horizon_t=metadata.get('horizon_t'),
            y_true=metadata.get('y_true'),
            y_pred=metadata.get('y_pred'),
            y_prev=metadata.get('y_prev'),
            pred_prev=metadata.get('pred_prev'),
            direction_true=metadata.get('direction_true'),
            direction_pred=metadata.get('direction_pred'),
            **features,
            **targets,
        )
        if 'timestamp' in metadata:
            payload['timestamp'] = metadata['timestamp']
        if 'time_features' in metadata:
            payload['time_features'] = metadata['time_features']
        np.savez_compressed(out_npz, **payload)

        # Backward-compatible single-split output.
        if len(eval_splits) == 1:
            np.savez_compressed(
                os.path.join(args.output_dir, 'features.npz'),
                **payload,
            )

        if not args.no_plots:
            make_plots(features, errors, args.output_dir, figure_subdir=f'figures_{split_flag}')

        print("\n=== Correlation Summary (pooled) ===")
        print(f"  {'feature':35s} {'Pearson r':>10s}  {'p':>10s}  "
              f"{'Spearman r':>10s}  {'p':>10s}")
        print("  " + "-" * 80)
        for fname in FEATURE_NAMES:
            m = metrics['pooled'].get(fname, {})
            pr = m.get('pearson_r', float('nan'))
            pp = m.get('pearson_p', float('nan'))
            sr = m.get('spearman_r', float('nan'))
            sp = m.get('spearman_p', float('nan'))
            print(f"  {fname:35s} {pr:10.4f}  {pp:10.2e}  {sr:10.4f}  {sp:10.2e}")

        split_contract = {}
        for k, v in payload.items():
            if v is None:
                continue
            arr = np.asarray(v)
            split_contract[k] = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}
        split_contract['_rows_used'] = n_used
        split_contract['_rows_total_before_gap_filter'] = n_total
        data_contract['splits'][split_flag] = split_contract

    with open(os.path.join(args.output_dir, 'data_contract.json'), 'w') as f:
        json.dump(data_contract, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
