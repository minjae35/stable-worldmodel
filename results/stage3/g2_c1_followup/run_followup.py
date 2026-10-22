"""G2 C1 follow-up: natural-distribution readout and label-free 2x2 prep.

No world-model training. Does not write into G0, G1, or C1 directories.
Object state is used only to score clips after they have been chosen.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.pop('HF_ENDPOINT', None)

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from stable_worldmodel.wm.spatial.model import FrozenDINOv2  # noqa: E402

OUT = ROOT / 'results/stage3/g2_c1_followup'
WEIGHTS = ROOT / 'results/stage3/g0_dinov2/weights'
G1_CLIPS = ROOT / 'results/stage3/g2_c1_audit/g1_clips_classified.json'
POOL = ROOT / 'results/stage3/g2_c1_audit/candidate_pool.json'
PATHS = {
    'pusht': '/workspace/datasets/lewm-pusht/train.lance',
    'tworoom': '/workspace/datasets/lewm-tworooms/train.lance',
    'cube': '/workspace/datasets/lewm-cube/train.lance',
}
SEED = 3072
FRAMESKIP = 5
SPAN = 10
CLIPS_PER_EP = 4
N_TRAIN, N_VAL, N_TEST = 256, 64, 64
PROBE_FIT = 216
PROBE_SELECT = 40
PROBE_FRAMES = 8
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]


def log(message: str):
    print(message, flush=True)


def json_dump(path: Path, payload):
    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    path.write_text(json.dumps(convert(payload), indent=2))


def load_columns(path: str, columns: list[str], batch_size: int = 200_000):
    import lance

    dataset = lance.dataset(path)
    total = dataset.count_rows()
    parts = {name: [] for name in columns}
    bar = tqdm(total=total, desc=f'load {Path(path).parent.name}', unit='row', mininterval=0.5)
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        n = batch.num_rows
        for name in columns:
            column = batch.column(name)
            if 'list' in str(column.type):
                width = column.type.list_size
                flat = column.flatten().to_numpy(zero_copy_only=False)
                parts[name].append(np.asarray(flat, dtype=np.float32).reshape(n, width))
            else:
                parts[name].append(np.asarray(column.to_numpy(zero_copy_only=False)))
        bar.update(n)
    bar.close()
    return {
        name: parts[name][0] if len(parts[name]) == 1 else np.concatenate(parts[name], axis=0)
        for name in columns
    }


def episode_bounds(episode_idx: np.ndarray):
    change = np.flatnonzero(np.diff(episode_idx)) + 1
    starts = np.concatenate([[0], change]).astype(np.int64)
    ends = np.concatenate([change, [len(episode_idx)]]).astype(np.int64)
    return starts, ends


def wrapped_deg(a0, a1):
    delta = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
    return np.abs(np.degrees(delta))


def split_ids(eligible: np.ndarray, n_train: int, n_val: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(eligible)
    need = n_train + n_val + n_test
    if len(order) < need:
        raise RuntimeError(f'need {need} episodes, have {len(order)}')
    chosen = order[:need]
    return {
        'train': chosen[:n_train].astype(int).tolist(),
        'val': chosen[n_train : n_train + n_val].astype(int).tolist(),
        'test': chosen[n_train + n_val :].astype(int).tolist(),
    }


def valid_starts(length: int):
    last = int(length) - SPAN
    if last < 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, last + 1, dtype=np.int64)


def uniform_starts(length: int, k: int):
    starts = valid_starts(length)
    if len(starts) == 0:
        return starts
    pick = np.unique(np.round(np.linspace(0, len(starts) - 1, k)).astype(np.int64))
    return starts[pick]


def sample_s0(length: int, rng: np.random.Generator):
    starts = valid_starts(length)
    replace = len(starts) < CLIPS_PER_EP
    return rng.choice(starts, size=CLIPS_PER_EP, replace=replace).astype(int).tolist()


def sample_s1(length: int, rng: np.random.Generator):
    starts = valid_starts(length)
    if len(starts) == 0:
        return []
    edges = np.linspace(0, len(starts), CLIPS_PER_EP + 1)
    chosen = []
    for i in range(CLIPS_PER_EP):
        lo = int(np.floor(edges[i]))
        hi = int(np.floor(edges[i + 1]))
        hi = max(hi, lo + 1)
        hi = min(hi, len(starts))
        lo = min(lo, hi - 1)
        chosen.append(int(starts[int(rng.integers(lo, hi))]))
    return chosen


def normalized_time(start: int, length: int):
    denom = max(int(length) - SPAN, 1)
    return float(start) / float(denom)


def quartile(start: int, length: int):
    return int(min(3, np.floor(normalized_time(start, length) * 4)))


def make_manifest(env: str, starts, ends, split: dict, seed: int):
    rng = np.random.default_rng(seed)
    manifest = {'env': env, 'split': split, 's0': {}, 's1': {}}
    for name, episodes in split.items():
        s0_rows, s1_rows = [], []
        for ep in episodes:
            length = int(ends[ep] - starts[ep])
            base = int(starts[ep])
            for sampler, bucket in ((sample_s0, s0_rows), (sample_s1, s1_rows)):
                for start in sampler(length, rng):
                    bucket.append(
                        {
                            'episode': int(ep),
                            'start': int(start),
                            'row': base + int(start),
                            'length': length,
                            'normalized_time': normalized_time(start, length),
                            'quartile': quartile(start, length),
                        }
                    )
        manifest['s0'][name] = s0_rows
        manifest['s1'][name] = s1_rows
    return manifest


def coverage_stats(clips, motion):
    """motion is a dict of arrays aligned with clips. State was not used to choose them."""
    times = np.array([c['normalized_time'] for c in clips], dtype=np.float64)
    quartiles = np.array([c['quartile'] for c in clips], dtype=np.int64)
    episodes = np.array([c['episode'] for c in clips], dtype=np.int64)
    per_ep = []
    occupied = []
    for ep in np.unique(episodes):
        q = quartiles[episodes == ep]
        per_ep.append(len(np.unique(q)))
        occupied.append(q)
    occupied = np.concatenate(occupied) if occupied else np.zeros(0, dtype=np.int64)
    stats = {
        'n_clips': int(len(clips)),
        'n_episodes': int(len(np.unique(episodes))) if len(clips) else 0,
        'mean_quartiles_per_episode': float(np.mean(per_ep)) if per_ep else None,
        'fraction_episodes_covering_all_quartiles': float(np.mean(np.array(per_ep) == 4)) if per_ep else None,
        'quartile_fraction': {str(i): float(np.mean(quartiles == i)) if len(quartiles) else None for i in range(4)},
        'time_mean': float(times.mean()) if len(times) else None,
        'time_std': float(times.std()) if len(times) else None,
    }
    for key, values in motion.items():
        values = np.asarray(values, dtype=np.float64)
        stats[key] = {
            'p50': float(np.median(values)),
            'p75': float(np.percentile(values, 75)),
            'p90': float(np.percentile(values, 90)),
            'mean': float(values.mean()),
        }
    return stats


def pusht_motion(state, clips):
    da, db, dang = [], [], []
    for clip in clips:
        s0 = state[clip['row']]
        s1 = state[clip['row'] + FRAMESKIP]
        da.append(float(np.linalg.norm(s1[:2] - s0[:2])))
        db.append(float(np.linalg.norm(s1[2:4] - s0[2:4])))
        dang.append(float(wrapped_deg(np.array([s0[4]]), np.array([s1[4]]))[0]))
    da, db, dang = np.asarray(da), np.asarray(db), np.asarray(dang)
    rates = {
        'static': float(np.mean((da < 5) & (db < 2) & (dang < 5))),
        'agent_only': float(np.mean((da >= 15) & (db < 2) & (dang < 5))),
        'block_2_15': float(np.mean((db >= 2) & (db < 15))),
        'block_ge_15': float(np.mean(db >= 15)),
        'block_translate_ge_5': float(np.mean(db >= 5)),
        'block_rotate_ge_10': float(np.mean(dang >= 10)),
    }
    # Episode-level: at least one of the clips shows the event.
    episodes = np.array([c['episode'] for c in clips])
    episode_rates = {}
    for name, mask in {
        'block_2_15': (db >= 2) & (db < 15),
        'block_ge_15': db >= 15,
        'block_translate_ge_5': db >= 5,
        'block_rotate_ge_10': dang >= 10,
        'agent_only': (da >= 15) & (db < 2) & (dang < 5),
    }.items():
        hit = []
        for ep in np.unique(episodes):
            hit.append(bool(np.any(mask[episodes == ep])))
        episode_rates[name] = float(np.mean(hit)) if hit else None
    stats = coverage_stats(clips, {'agent_delta': da, 'block_delta': db, 'angle_delta_deg': dang})
    stats['clip_rates'] = rates
    stats['episode_hit_rates'] = episode_rates
    return stats


def vector_motion(values, clips, name, thresholds):
    delta = []
    for clip in clips:
        a = values[clip['row']]
        b = values[clip['row'] + FRAMESKIP]
        delta.append(float(np.linalg.norm(b - a)))
    delta = np.asarray(delta, dtype=np.float64)
    episodes = np.array([c['episode'] for c in clips])
    rates = {}
    episode_rates = {}
    for label, lo, hi in thresholds:
        if hi is None:
            mask = delta >= lo
        else:
            mask = (delta >= lo) & (delta < hi)
        rates[label] = float(np.mean(mask)) if len(mask) else None
        hit = [bool(np.any(mask[episodes == ep])) for ep in np.unique(episodes)]
        episode_rates[label] = float(np.mean(hit)) if hit else None
    stats = coverage_stats(clips, {name: delta})
    stats['clip_rates'] = rates
    stats['episode_hit_rates'] = episode_rates
    return stats


def g1_reference():
    clips = json.loads(G1_CLIPS.read_text())
    da = np.array([c['agent_delta'] for c in clips])
    db = np.array([c['block_delta'] for c in clips])
    dang = np.array([c['angle_delta_deg'] for c in clips])
    return {
        'n': int(len(clips)),
        'clips_per_episode': 1,
        'median_agent_delta': float(np.median(da)),
        'median_block_delta': float(np.median(db)),
        'median_angle_delta_deg': float(np.median(dang)),
        'clip_rates': {
            'static': float(np.mean((da < 5) & (db < 2) & (dang < 5))),
            'agent_only': float(np.mean((da >= 15) & (db < 2) & (dang < 5))),
            'block_2_15': float(np.mean((db >= 2) & (db < 15))),
            'block_ge_15': float(np.mean(db >= 15)),
            'block_translate_ge_5': float(np.mean(db >= 5)),
            'block_rotate_ge_10': float(np.mean(dang >= 10)),
        },
    }


def preprocess(images, device):
    x = torch.from_numpy(np.ascontiguousarray(images))
    x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div(255)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def decode_image(blob):
    raw = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    with Image.open(io.BytesIO(raw)) as img:
        array = np.asarray(img.convert('RGB'))
    if array.shape != (224, 224, 3):
        raise RuntimeError(f'expected 224x224, got {array.shape}')
    return array


def fetch_images(path, rows):
    import lance

    dataset = lance.dataset(path)
    images = [None] * len(rows)
    for start in tqdm(range(0, len(rows), 128), desc='decode', unit='batch', mininterval=0.5):
        chunk = [int(r) for r in rows[start : start + 128]]
        blobs = dataset.take(chunk, columns=['pixels'])['pixels'].to_pylist()
        for i, blob in enumerate(blobs):
            images[start + i] = decode_image(blob)
    return np.stack(images)


@torch.no_grad()
def encode_images(encoder, images, device, batch=64):
    parts = []
    for start in tqdm(range(0, len(images), batch), desc='dino', unit='batch', mininterval=0.5):
        pixels = preprocess(images[start : start + batch], device)
        tokens = encoder.encode(pixels)[:, 0].half().cpu()
        if tokens.shape[1:] != (256, 384):
            raise RuntimeError(tuple(tokens.shape))
        parts.append(tokens)
    return torch.cat(parts, dim=0)


def pool4(tokens, batch=512):
    parts = []
    for start in range(0, len(tokens), batch):
        chunk = tokens[start : start + batch].float().reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(chunk, (4, 4))
        parts.append(pooled.permute(0, 2, 3, 1).reshape(chunk.shape[0], -1).cpu())
    return torch.cat(parts, dim=0)


def dual_ridge(x_fit, y_fit, x_eval, alpha):
    x_mu = x_fit.mean(0, keepdim=True)
    x_sd = x_fit.std(0, keepdim=True).clamp_min(1e-6)
    y_mu = y_fit.mean(0, keepdim=True)
    y_sd = y_fit.std(0, keepdim=True).clamp_min(1e-6)
    xt = (x_fit - x_mu) / x_sd
    yt = (y_fit - y_mu) / y_sd
    gram = xt @ xt.T
    coef = torch.linalg.solve(gram + float(alpha) * torch.eye(len(xt), device=xt.device), yt)
    pred = ((x_eval - x_mu) / x_sd) @ xt.T @ coef
    return pred * y_sd + y_mu


def pca_ridge(x_fit, y_fit, x_eval, k, alpha):
    x_mu = x_fit.mean(0, keepdim=True)
    x_sd = x_fit.std(0, keepdim=True).clamp_min(1e-6)
    y_mu = y_fit.mean(0, keepdim=True)
    y_sd = y_fit.std(0, keepdim=True).clamp_min(1e-6)
    xt = (x_fit - x_mu) / x_sd
    yt = (y_fit - y_mu) / y_sd
    _u, _s, v = torch.pca_lowrank(xt, q=k)
    z_fit = xt @ v
    z_eval = ((x_eval - x_mu) / x_sd) @ v
    gram = z_fit.T @ z_fit + float(alpha) * torch.eye(k, device=xt.device)
    weight = torch.linalg.solve(gram, z_fit.T @ yt)
    return (z_eval @ weight) * y_sd + y_mu


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def l2_mean(pred, true):
    return float(np.linalg.norm(pred - true, axis=1).mean())


def angle_mae(pred_sc, true_sc):
    pred = np.arctan2(pred_sc[:, 0], pred_sc[:, 1])
    true = np.arctan2(true_sc[:, 0], true_sc[:, 1])
    err = np.arctan2(np.sin(pred - true), np.cos(pred - true))
    return float(np.degrees(np.abs(err)).mean())


def score_error(pred, true, kind):
    if kind == 'angle':
        return angle_mae(pred, true)
    return l2_mean(pred, true)


def r2_score(pred, true):
    residual = ((true - pred) ** 2).sum()
    total = ((true - true.mean(0)) ** 2).sum()
    if total <= 0:
        return None
    return float(1.0 - residual / total)


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 5 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(score, positive):
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ties so equal scores do not inflate the rank.
    sorted_score = score[order]
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        if stop > start + 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
        start = stop
    sum_pos = float(ranks[positive].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class TinySpatialReadout(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(384, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(4),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 16, out_dim),
        )

    def forward(self, tokens):
        x = tokens.reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
        return self.net(x)


def train_tiny(tokens, targets, groups, kind, steps=250):
    device = torch.device('cuda')
    torch.manual_seed(SEED)
    model = TinySpatialReadout(targets.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    train_i = np.asarray(groups['train'])
    val_i = np.asarray(groups['val'])
    y_mu = targets[train_i].mean(0)
    y_sd = targets[train_i].std(0).clamp_min(1e-6)
    best_state, best_val, bad = None, 1e9, 0
    rng = np.random.default_rng(SEED)
    history = []
    for step in range(steps):
        model.train()
        take = train_i[rng.choice(len(train_i), size=min(128, len(train_i)), replace=False)]
        batch_x = tokens[take].float().to(device)
        batch_y = ((targets[take] - y_mu) / y_sd).to(device)
        pred = model(batch_x)
        loss = (pred - batch_y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == steps - 1:
            err = eval_tiny(model, tokens, targets, val_i, y_mu, y_sd, kind, device)
            history.append({'step': step, 'val': err})
            if err < best_val - 1e-4:
                best_val = err
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if step > 30 and bad >= 8:
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    result = {'best_val': best_val, 'steps_ran': history[-1]['step'] if history else 0}
    preds = {}
    for name, index in groups.items():
        pred = predict_tiny(model, tokens, np.asarray(index), y_mu, y_sd, device)
        true = targets[np.asarray(index)].numpy()
        result[name] = {'mae': score_error(pred, true, kind), 'mean_abs': mae(pred, true)}
        preds[name] = pred
    result['pred'] = preds
    return result


@torch.no_grad()
def predict_tiny(model, tokens, index, y_mu, y_sd, device, batch=256):
    model.eval()
    outs = []
    for start in range(0, len(index), batch):
        take = index[start : start + batch]
        raw = model(tokens[take].float().to(device)).float().cpu()
        outs.append(raw * y_sd + y_mu)
    return torch.cat(outs, dim=0).numpy()


@torch.no_grad()
def eval_tiny(model, tokens, targets, index, y_mu, y_sd, kind, device):
    pred = predict_tiny(model, tokens, index, y_mu, y_sd, device)
    return score_error(pred, targets[index].numpy(), kind)


def select_ridge(features, targets, groups, kind):
    device = features.device
    fit_i = torch.as_tensor(groups['train'], device=device)
    val_i = torch.as_tensor(groups['val'], device=device)
    y_val = targets[val_i].float().cpu().numpy()
    best = {'alpha': None, 'val': 1e9}
    for alpha in ALPHAS:
        pred = dual_ridge(features[fit_i], targets[fit_i], features[val_i], alpha).float().cpu().numpy()
        err = score_error(pred, y_val, kind)
        if err < best['val']:
            best = {'alpha': alpha, 'val': err}
    out = {'alpha': best['alpha'], 'val': best['val'], 'pred': {}}
    for name, index in groups.items():
        pred = dual_ridge(
            features[fit_i], targets[fit_i], features[torch.as_tensor(index, device=device)], best['alpha']
        ).float().cpu().numpy()
        true = targets[torch.as_tensor(index, device=device)].float().cpu().numpy()
        out[name] = {'mae': score_error(pred, true, kind), 'mean_abs': mae(pred, true), 'r2': r2_score(pred, true)}
        out['pred'][name] = pred
    return out


def select_pca(features, targets, groups, kind):
    device = features.device
    fit_i = torch.as_tensor(groups['train'], device=device)
    val_i = torch.as_tensor(groups['val'], device=device)
    y_val = targets[val_i].float().cpu().numpy()
    best = {'k': None, 'alpha': None, 'val': 1e9}
    for k in (32, 128):
        for alpha in ALPHAS:
            pred = pca_ridge(features[fit_i], targets[fit_i], features[val_i], k, alpha).float().cpu().numpy()
            err = score_error(pred, y_val, kind)
            if err < best['val']:
                best = {'k': k, 'alpha': alpha, 'val': err}
    out = {'k': best['k'], 'alpha': best['alpha'], 'val': best['val'], 'pred': {}}
    for name, index in groups.items():
        pred = pca_ridge(
            features[fit_i],
            targets[fit_i],
            features[torch.as_tensor(index, device=device)],
            best['k'],
            best['alpha'],
        ).float().cpu().numpy()
        true = targets[torch.as_tensor(index, device=device)].float().cpu().numpy()
        out[name] = {'mae': score_error(pred, true, kind), 'mean_abs': mae(pred, true), 'r2': r2_score(pred, true)}
        out['pred'][name] = pred
    return out


def delta_arrays(pred_now, pred_fut, true_now, true_fut, kind):
    if kind == 'angle':
        def ang(sc):
            return np.arctan2(sc[:, 0], sc[:, 1])

        true_d = np.arctan2(np.sin(ang(true_fut) - ang(true_now)), np.cos(ang(true_fut) - ang(true_now)))
        pred_d = np.arctan2(np.sin(ang(pred_fut) - ang(pred_now)), np.cos(ang(pred_fut) - ang(pred_now)))
        err = np.degrees(np.abs(np.arctan2(np.sin(pred_d - true_d), np.cos(pred_d - true_d))))
        motion = np.degrees(np.abs(true_d))
        pred_mag = np.degrees(np.abs(pred_d))
        return err, motion, pred_mag
    err = np.linalg.norm((pred_fut - pred_now) - (true_fut - true_now), axis=1)
    motion = np.linalg.norm(true_fut - true_now, axis=1)
    pred_mag = np.linalg.norm(pred_fut - pred_now, axis=1)
    return err, motion, pred_mag


def binned(err, motion, pred_mag, edges):
    rows = []
    bounds = [0.0, *edges, 1e9]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (motion >= lo) & (motion < hi)
        n = int(mask.sum())
        if n < 5:
            continue
        rows.append(
            {
                'lo': lo,
                'hi': None if hi > 1e8 else hi,
                'n': n,
                'mean_motion': float(motion[mask].mean()),
                'median_motion': float(np.median(motion[mask])),
                'delta_mae': float(err[mask].mean()),
                'median_predicted_motion': float(np.median(pred_mag[mask])),
                'beats_zero': bool(err[mask].mean() < motion[mask].mean()),
                'spearman': spearman(pred_mag[mask], motion[mask]),
            }
        )
    return rows


def absolute_bins(pred, true, motion, edges, kind):
    rows = []
    bounds = [0.0, *edges, 1e9]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (motion >= lo) & (motion < hi)
        if int(mask.sum()) < 5:
            continue
        if kind == 'angle':
            err = angle_mae(pred[mask], true[mask])
        else:
            err = l2_mean(pred[mask], true[mask])
        rows.append(
            {
                'lo': lo,
                'hi': None if hi > 1e8 else hi,
                'n': int(mask.sum()),
                'absolute_mae': err,
                'mean_motion': float(motion[mask].mean()),
            }
        )
    return rows


def pack_method(name, fit_now, fit_fut, true_now, true_fut, kind, edges):
    pred_now = fit_now['pred']['test']
    pred_fut = fit_fut['pred']['test']
    err, motion, pred_mag = delta_arrays(pred_now, pred_fut, true_now, true_fut, kind)
    small = (motion >= 2) & (motion < 15) if kind == 'xy' else (motion >= 5) & (motion < 15)
    still = motion < (2 if kind == 'xy' else 5)
    report = {
        'name': name,
        'train_mae': fit_fut['train']['mae'],
        'val_mae': fit_fut['val']['mae'],
        'test_absolute_mae': fit_fut['test']['mae'],
        'test_r2': r2_score(pred_fut, true_fut),
        'ceiling_mae': fit_fut['test']['mae'],
        'feature_persistence_mae': score_error(pred_now, true_fut, kind),
        'true_persistence_mae': score_error(true_now, true_fut, kind),
        'delta_mae': float(err.mean()),
        'true_motion_mae': float(motion.mean()),
        'fraction_delta_closer_than_zero': float(np.mean(err < motion)),
        'spearman_motion': spearman(pred_mag, motion),
        'auroc_small_vs_still': auroc(pred_mag[small | still], small[small | still]),
        'bins': binned(err, motion, pred_mag, edges),
        'absolute_bins': absolute_bins(pred_fut, true_fut, motion, edges, kind),
        'median_predicted_motion_still': float(np.median(pred_mag[still])) if np.any(still) else None,
        'median_predicted_motion_small': float(np.median(pred_mag[small])) if np.any(small) else None,
    }
    report['train_val_gap'] = report['train_mae'] - report['val_mae']
    return report


def distinguishable(report, kind):
    """Held-out decision rule. Thresholds are criteria, not measurements."""
    if kind == 'xy':
        wanted = [(2.0, 8.0), (8.0, 15.0)]
    else:
        wanted = [(5.0, 15.0)]
    found = []
    for lo, hi in wanted:
        match = [row for row in report['bins'] if row['lo'] == lo and row['hi'] == hi]
        if not match:
            return False
        row = match[0]
        useful = row['n'] >= 30 and row['beats_zero'] and row['delta_mae'] <= 0.75 * row['mean_motion']
        found.append(useful)
    auroc_ok = report['auroc_small_vs_still'] is not None and report['auroc_small_vs_still'] >= 0.7
    return bool(all(found) and auroc_ok)


def cell_of(xy):
    pix = np.asarray(xy, dtype=np.float64) * (224.0 / 512.0)
    cell = np.floor(pix / 14.0).astype(np.int64)
    return np.clip(cell, 0, 15)


def window_report(tokens_now, tokens_fut, state_now, state_fut):
    grid_now = tokens_now.float().reshape(-1, 16, 16, 384)
    grid_fut = tokens_fut.float().reshape(-1, 16, 16, 384)
    block = np.linalg.norm(state_fut[:, 2:4] - state_now[:, 2:4], axis=1)
    agent = np.linalg.norm(state_fut[:, 0:2] - state_now[:, 0:2], axis=1)
    block_l2 = region_l2(grid_now, grid_fut, state_now[:, 2:4])
    agent_l2 = region_l2(grid_now, grid_fut, state_now[:, 0:2])
    global_l2 = torch.linalg.vector_norm(grid_fut - grid_now, dim=-1).mean(dim=(1, 2)).numpy()
    small = (block >= 2) & (block < 15)
    still = block < 2
    large = block >= 15

    def pack(name, values):
        return {
            'name': name,
            'spearman_block': spearman(values, block),
            'spearman_agent': spearman(values, agent),
            'auroc_block_small_vs_still': auroc(values[small | still], small[small | still]),
            'median_still': float(np.median(values[still])) if np.any(still) else None,
            'median_small': float(np.median(values[small])) if np.any(small) else None,
            'median_large': float(np.median(values[large])) if np.any(large) else None,
        }

    return {
        'note': (
            'Window is a 3x3 token neighborhood around state_xy * 224/512. '
            'State locates the evaluation region only. It is not a training loss. '
            'The same window can be applied to predicted DINO tokens. '
            'An RGB centroid cannot.'
        ),
        'block_window': pack('block_window', block_l2),
        'agent_window': pack('agent_window', agent_l2),
        'global_mean': pack('global_mean', global_l2),
    }


def region_l2(grid_now, grid_fut, xy, radius=1):
    cells = cell_of(xy)
    values = np.empty(len(cells), dtype=np.float64)
    for i in range(len(cells)):
        x = int(cells[i, 0])
        y = int(cells[i, 1])
        xs = slice(max(0, x - radius), min(16, x + radius + 1))
        ys = slice(max(0, y - radius), min(16, y + radius + 1))
        diff = (grid_fut[i, ys, xs] - grid_now[i, ys, xs]).reshape(-1, 384)
        values[i] = float(torch.linalg.vector_norm(diff, dim=1).mean())
    return values


PALETTE = np.array(
    [[255, 255, 255], [144, 238, 144], [65, 105, 225], [119, 136, 153]],
    dtype=np.float32,
)


def rgb_centroid_check(images, state):
    mapped_block = state[:, 2:4] * (224.0 / 512.0)
    mapped_agent = state[:, 0:2] * (224.0 / 512.0)
    block_err, agent_err, gray_count = [], [], []
    for image, block_xy, agent_xy in zip(images, mapped_block, mapped_agent):
        flat = image.reshape(-1, 3).astype(np.float32)
        dist = ((flat[:, None, :] - PALETTE[None, :, :]) ** 2).sum(-1)
        label = dist.argmin(axis=1).reshape(image.shape[:2])
        for class_id, target, bucket in ((3, block_xy, block_err), (2, agent_xy, agent_err)):
            ys, xs = np.nonzero(label == class_id)
            gray_count.append(int(len(xs))) if class_id == 3 else None
            if len(xs) < 20:
                bucket.append(None)
                continue
            centroid = np.array([xs.mean(), ys.mean()])
            bucket.append(float(np.linalg.norm(centroid - target)))
    def summarize(values):
        kept = np.array([v for v in values if v is not None], dtype=np.float64)
        return {
            'n': int(len(kept)),
            'median_px': float(np.median(kept)) if len(kept) else None,
            'p90_px': float(np.percentile(kept, 90)) if len(kept) else None,
            'fraction_within_10px': float(np.mean(kept <= 10)) if len(kept) else None,
            'fraction_within_14px': float(np.mean(kept <= 14)) if len(kept) else None,
        }
    return {
        'note': (
            'Nearest-color centroid on real RGB. This checks the pixel mapping. '
            'It cannot be computed from DINO tokens.'
        ),
        'n_images': int(len(images)),
        'block': summarize(block_err),
        'agent': summarize(agent_err),
    }


def probe_indices(starts, ends, split):
    rng = np.random.default_rng(SEED + 1)
    train = np.array(split['train'])
    order = rng.permutation(train)
    fit_eps = set(order[:PROBE_FIT].tolist())
    select_eps = set(order[PROBE_FIT : PROBE_FIT + PROBE_SELECT].tolist())
    test_eps = set(split['test'])
    if fit_eps & select_eps or fit_eps & test_eps or select_eps & test_eps:
        raise RuntimeError('probe episodes overlap')
    rows = []
    for ep, group in (
        *((ep, 'train') for ep in sorted(fit_eps)),
        *((ep, 'val') for ep in sorted(select_eps)),
        *((ep, 'test') for ep in sorted(test_eps)),
    ):
        length = int(ends[ep] - starts[ep])
        base = int(starts[ep])
        for start in uniform_starts(length, PROBE_FRAMES):
            rows.append(
                {
                    'episode': int(ep),
                    'start': int(start),
                    'row': base + int(start),
                    'group': group,
                }
            )
    return rows


def save_plots(readout, coverage):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in (
        (axes[0], 'block_xy', 'Block XY delta error'),
        (axes[1], 'block_angle', 'Block angle delta error (deg)'),
    ):
        methods = readout[key]['methods']
        names = list(methods)
        # Plot the middle method's bins as bars, and overlay others as points.
        base = methods[names[0]]['bins']
        labels = [f"{int(r['lo'])}-{r['hi'] if r['hi'] is not None else 'inf'}" for r in base]
        x = np.arange(len(base))
        width = 0.25
        for i, name in enumerate(names):
            rows = { (r['lo'], r['hi']): r for r in methods[name]['bins'] }
            delta = [rows[(r['lo'], r['hi'])]['delta_mae'] if (r['lo'], r['hi']) in rows else np.nan for r in base]
            ax.bar(x + (i - 1) * width, delta, width=width, label=name)
        ax.plot(x, [r['mean_motion'] for r in base], 'k--', label='true motion')
        ax.set_xticks(x, labels, rotation=30)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'readout_bins.png', dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ['static', 'agent_only', 'block_2_15', 'block_ge_15', 'block_rotate_ge_10']
    g1 = coverage['g1']['clip_rates']
    s0 = coverage['pusht']['s0']['all']['clip_rates']
    s1 = coverage['pusht']['s1']['all']['clip_rates']
    x = np.arange(len(labels))
    axes[0].bar(x - 0.25, [g1[k] for k in labels], width=0.25, label='G1')
    axes[0].bar(x, [s0[k] for k in labels], width=0.25, label='S0')
    axes[0].bar(x + 0.25, [s1[k] for k in labels], width=0.25, label='S1')
    axes[0].set_xticks(x, labels, rotation=30)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('PushT clip rates')
    axes[0].legend()
    qs = ['0', '1', '2', '3']
    axes[1].plot(qs, [coverage['pusht']['s0']['all']['quartile_fraction'][q] for q in qs], label='S0')
    axes[1].plot(qs, [coverage['pusht']['s1']['all']['quartile_fraction'][q] for q in qs], label='S1')
    axes[1].set_ylim(0, 0.5)
    axes[1].set_title('Clip time quartile')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / 'sampling_coverage.png', dpi=120)
    plt.close(fig)


def flatten_clips(manifest, sampler):
    rows = []
    for split in ('train', 'val', 'test'):
        rows.extend(manifest[sampler][split])
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'STATUS').write_text('RUNNING\n')
    log('C1 follow-up start')
    pusht = load_columns(PATHS['pusht'], ['episode_idx', 'state'])
    starts, ends = episode_bounds(pusht['episode_idx'].astype(np.int64))
    lengths = ends - starts
    eligible = np.flatnonzero(lengths >= SPAN)
    split = split_ids(eligible, N_TRAIN, N_VAL, N_TEST, SEED)
    manifest = make_manifest('pusht', starts, ends, split, SEED)
    state = pusht['state']
    coverage = {'g1': g1_reference(), 'pusht': {}}
    for sampler in ('s0', 's1'):
        all_clips = flatten_clips(manifest, sampler)
        coverage['pusht'][sampler] = {
            'all': pusht_motion(state, all_clips),
            'test': pusht_motion(state, manifest[sampler]['test']),
        }
    log(
        'pusht coverage '
        f"S0 block>=5 {coverage['pusht']['s0']['all']['clip_rates']['block_translate_ge_5']:.3f} "
        f"S1 {coverage['pusht']['s1']['all']['clip_rates']['block_translate_ge_5']:.3f} "
        f"G1 {coverage['g1']['clip_rates']['block_translate_ge_5']:.3f}"
    )

    tw = load_columns(PATHS['tworoom'], ['episode_idx', 'pos_agent'])
    tw_starts, tw_ends = episode_bounds(tw['episode_idx'].astype(np.int64))
    tw_split = split_ids(np.flatnonzero((tw_ends - tw_starts) >= SPAN), N_TRAIN, N_VAL, N_TEST, SEED)
    tw_manifest = make_manifest('tworoom', tw_starts, tw_ends, tw_split, SEED)
    coverage['tworoom'] = {}
    for sampler in ('s0', 's1'):
        clips = flatten_clips(tw_manifest, sampler)
        coverage['tworoom'][sampler] = {
            'all': vector_motion(
                tw['pos_agent'],
                clips,
                'agent_delta_px',
                [('still_lt_2', 0.0, 2.0), ('mid_2_15', 2.0, 15.0), ('ge_15', 15.0, None)],
            )
        }
    cube = load_columns(PATHS['cube'], ['episode_idx', 'privileged_block_0_pos'])
    cu_starts, cu_ends = episode_bounds(cube['episode_idx'].astype(np.int64))
    cu_split = split_ids(np.flatnonzero((cu_ends - cu_starts) >= SPAN), N_TRAIN, N_VAL, N_TEST, SEED)
    cu_manifest = make_manifest('cube', cu_starts, cu_ends, cu_split, SEED)
    coverage['cube'] = {}
    for sampler in ('s0', 's1'):
        clips = flatten_clips(cu_manifest, sampler)
        coverage['cube'][sampler] = {
            'all': vector_motion(
                cube['privileged_block_0_pos'],
                clips,
                'block_delta_m',
                [('still_lt_0.005', 0.0, 0.005), ('mid_0.005_0.02', 0.005, 0.02), ('ge_0.02', 0.02, None)],
            )
        }
    bundle = {
        'seed': SEED,
        'frameskip': FRAMESKIP,
        'span': SPAN,
        'clips_per_episode': CLIPS_PER_EP,
        'counts': {'train': N_TRAIN, 'val': N_VAL, 'test': N_TEST},
        'sampling': {
            's0': 'uniform random start, four clips per episode, no state',
            's1': 'one clip in each valid-start quartile, no state',
        },
        'forbidden_in_sampling_and_loss': ['object_state', 'mask', 'reward', 'contact', 'event_label'],
        'challenge_pool': str(POOL),
        'pusht': manifest,
        'tworoom': tw_manifest,
        'cube': cu_manifest,
    }
    json_dump(OUT / 'manifest.json', bundle)
    json_dump(OUT / 'sampling_coverage.json', coverage)
    log('manifest written')

    items = probe_indices(starts, ends, split)
    groups = {'train': [], 'val': [], 'test': []}
    for i, item in enumerate(items):
        groups[item['group']].append(i)
    overlap = set(items[i]['episode'] for i in groups['train']) & set(items[i]['episode'] for i in groups['test'])
    if overlap:
        raise RuntimeError(f'readout leakage {len(overlap)}')
    rows = []
    for item in items:
        rows.extend([item['row'], item['row'] + FRAMESKIP])
    log(f'readout frames {len(rows)} episodes train/val/test '
        f"{len({items[i]['episode'] for i in groups['train']})}/"
        f"{len({items[i]['episode'] for i in groups['val']})}/"
        f"{len({items[i]['episode'] for i in groups['test']})}")
    images = fetch_images(PATHS['pusht'], rows)
    device = torch.device('cuda')
    encoder = FrozenDINOv2.from_pretrained(str(WEIGHTS)).to(device)
    tokens = encode_images(encoder, images, device)
    del encoder
    now = tokens[0::2]
    fut = tokens[1::2]
    y0 = np.stack([state[item['row']] for item in items]).astype(np.float32)
    y1 = np.stack([state[item['row'] + FRAMESKIP] for item in items]).astype(np.float32)
    # Centroid check uses a spread of held-out frames, then the pixels can go.
    test_pos = np.asarray(groups['test'][:40])
    centroid = rgb_centroid_check(images[test_pos * 2], y0[test_pos])
    del images
    json_dump(OUT / 'rgb_centroid.json', centroid)
    log(f"rgb centroid block median {centroid['block']['median_px']} agent {centroid['agent']['median_px']}")

    window = window_report(now[groups['test']], fut[groups['test']], y0[groups['test']], y1[groups['test']])
    json_dump(OUT / 'feature_window.json', window)
    log(
        'window spearman block/global '
        f"{window['block_window']['spearman_block']} / {window['global_mean']['spearman_block']}"
    )

    feat_now = pool4(now).to(device)
    feat_fut = pool4(fut).to(device)
    readout = {
        'n': {name: len(ids) for name, ids in groups.items()},
        'episodes': {
            name: len({items[i]['episode'] for i in ids}) for name, ids in groups.items()
        },
        'frame_rule': 'eight uniform-in-time frames per episode, not max-motion',
        'episode_overlap_train_test': 0,
    }
    specs = {
        'block_xy': (slice(2, 4), 'xy', [2, 8, 15, 30]),
        'agent_xy': (slice(0, 2), 'xy', [5, 15, 40, 80]),
        'block_angle': (slice(4, 5), 'angle', [5, 15, 30]),
    }
    for key, (sl, kind, edges) in specs.items():
        if kind == 'angle':
            t0 = np.stack([np.sin(y0[:, 4]), np.cos(y0[:, 4])], axis=1).astype(np.float32)
            t1 = np.stack([np.sin(y1[:, 4]), np.cos(y1[:, 4])], axis=1).astype(np.float32)
        else:
            t0 = y0[:, sl].astype(np.float32)
            t1 = y1[:, sl].astype(np.float32)
        # One probe sees both frames of a pair. The two frames stay in the same split.
        n_pairs = len(items)
        both_groups = {name: list(ids) + [i + n_pairs for i in ids] for name, ids in groups.items()}
        feat_both = torch.cat([feat_now, feat_fut], dim=0).to(device)
        tok_both = torch.cat([now, fut], dim=0)
        target_both = np.concatenate([t0, t1], axis=0)
        target_both_t = torch.from_numpy(target_both).to(device)
        motion = np.linalg.norm(t1 - t0, axis=1) if kind == 'xy' else wrapped_deg(y0[:, 4], y1[:, 4])
        log(
            f'fit {key} test bins '
            f'still {(motion[groups["test"]] < (2 if kind == "xy" else 5)).sum()} '
            f'small {((motion[groups["test"]] >= (2 if kind == "xy" else 5)) & (motion[groups["test"]] < 15)).sum()}'
        )
        ridge = select_ridge(feat_both, target_both_t, both_groups, kind)
        pca = select_pca(feat_both, target_both_t, both_groups, kind)
        tiny = train_tiny(tok_both, torch.from_numpy(target_both), both_groups, kind)
        del feat_both
        methods = {}
        for method, fit in (('pool4_ridge', ridge), ('pca_ridge', pca), ('tiny_spatial', tiny)):
            half = len(fit['pred']['test']) // 2
            fit_now = {'pred': {'test': fit['pred']['test'][:half]}, 'train': fit['train'], 'val': fit['val'], 'test': fit['test']}
            fit_fut = {'pred': {'test': fit['pred']['test'][half:]}, 'train': fit['train'], 'val': fit['val'], 'test': fit['test']}
            test_i = np.asarray(groups['test'])
            methods[method] = pack_method(
                method, fit_now, fit_fut, t0[test_i], t1[test_i], kind, edges
            )
            methods[method]['hyperparams'] = {
                k: fit[k] for k in ('alpha', 'k', 'val', 'steps_ran') if k in fit
            }
            methods[method]['distinguish_small'] = distinguishable(methods[method], kind)
        readout[key] = {'kind': kind, 'methods': methods}
        best = min(methods, key=lambda name: methods[name]['test_absolute_mae'])
        log(
            f"{key} best {best} test {methods[best]['test_absolute_mae']:.3f} "
            f"persist {methods[best]['true_persistence_mae']:.3f} "
            f"small {methods[best]['distinguish_small']}"
        )
    # Drop nothing huge: predictions were numpy inside methods and are small enough.
    json_dump(OUT / 'readout.json', readout)
    save_plots(readout, coverage)
    config = {
        'experiment': 'g2_label_free_2x2',
        'status': 'prepared_not_trained',
        'seed': SEED,
        'frameskip': FRAMESKIP,
        'span': SPAN,
        'history': 1,
        'clips_per_episode': CLIPS_PER_EP,
        'episodes_per_env': {'train': N_TRAIN, 'val': N_VAL, 'test': N_TEST},
        'factors': {
            'sampling': {
                's0': 'uniform random clip starts',
                's1': 'one clip in each quartile of valid starts',
            },
            'loss': {
                'mse': 'uniform mean squared error over predicted DINO tokens',
                'l1': 'uniform mean absolute error over predicted DINO tokens',
            },
        },
        'same_rule_all_envs': True,
        'shared_episode_split_across_factors': True,
        'forbidden_in_sampling_and_loss': ['object_state', 'mask', 'reward', 'contact', 'event_label'],
        'state_use': 'evaluation and this diagnostic only',
        'evaluation': {
            'do_not_judge_small_pusht_block_motion_by_state_readout': True,
            'state_readout_floor': 'about 11 PushT units and 9 degrees on held-out episodes',
            'state_readout_usable_when': 'block translation above about 15, or agent translation above about 15',
            'object_dynamics_metric': (
                'error of predicted DINO tokens versus the true future tokens inside a '
                '3x3 window around block_xy * 224/512, compared with copying the current tokens'
            ),
            'rgb_centroid': (
                'pixel-only diagnostic; median 18px from the block state point, so it is '
                'not the state coordinate and cannot be applied to DINO tokens'
            ),
        },
        'challenge_evaluation': str(POOL),
        'manifest': str(OUT / 'manifest.json'),
        'model': 'existing frozen DINOv2 spatial dynamics, no architecture change in this step',
    }
    json_dump(OUT / 'config.json', config)
    (OUT / 'STATUS').write_text('DONE\n')
    log('C1_FOLLOWUP_DONE')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        (OUT / 'STATUS').write_text('FAILED\n')
        raise
