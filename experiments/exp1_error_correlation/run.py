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
import json
import os
import sys
from pathlib import Path

import numpy as np
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
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='auto')

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

def extract_batch_features(aux, batch_y, feat_ids, args, refs, patch_nums):
    """
    Compute all scalar features for one batch.

    Returns
    -------
    features : dict  {name: np.array [B * pred_len]}
    errors   : np.array [B * pred_len]
    targets  : dict  {name: np.array [B * pred_len]}
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

    return features, errors.ravel(), targets


def run_evaluation(model, args, device, refs):
    """Iterate over the eval split, extract features and errors."""
    ds = build_dataset(args, args.split)
    tot_len = ds.tot_len
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    patch_nums = model.patch_nums

    accum = {k: [] for k in FEATURE_NAMES}
    all_errors, all_ch = [], []
    all_targets = {}

    idx = 0
    with torch.no_grad(), _autocast_ctx(device):
        for bx, by, bxm, bym in tqdm(loader, desc='Eval'):
            B = bx.shape[0]
            fids = np.array([(idx + j) // tot_len for j in range(B)])
            idx += B

            bx = bx.float().to(device)
            bxm = bxm.float().to(device)
            bym_d = bym.float().to(device)
            di = torch.zeros_like(by[:, -args.pred_len:, :]).float()
            di = torch.cat([by[:, :args.label_len, :], di], dim=1).float().to(device)

            aux = model(bx, bxm, di, bym_d, return_aux=True)
            feats, errs, targets = extract_batch_features(
                aux, by, fids, args, refs, patch_nums
            )

            for k in FEATURE_NAMES:
                accum[k].append(feats[k])
            all_errors.append(errs)
            all_ch.append(np.repeat(fids, args.pred_len))
            for tk, tv in targets.items():
                all_targets.setdefault(tk, []).append(tv)

    for k in FEATURE_NAMES:
        accum[k] = np.concatenate(accum[k])
    for tk in list(all_targets.keys()):
        all_targets[tk] = np.concatenate(all_targets[tk])
    return accum, np.concatenate(all_errors), np.concatenate(all_ch), all_targets


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

def make_plots(features, errors, output_dir):
    fig_dir = os.path.join(output_dir, 'figures')
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
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── load model ──
    model = load_model(args, device)
    print(f"Model loaded  |  patch_nums={model.patch_nums}  d_ff={args.d_ff}  "
          f"llm_layers={args.llm_layers}  alignment={args.alignment}")

    # ── Phase A ──
    refs = collect_train_references(model, args, device)

    # ── Phase B ──
    features, errors, channel_ids, targets = run_evaluation(model, args, device, refs)

    # ── Phase C ──
    metrics = compute_correlations(features, errors, channel_ids, args.enc_in)
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        os.path.join(args.output_dir, 'features.npz'),
        errors=errors, channel_ids=channel_ids, **features, **targets,
    )

    # ── Phase D ──
    make_plots(features, errors, args.output_dir)

    # ── summary ──
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

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
