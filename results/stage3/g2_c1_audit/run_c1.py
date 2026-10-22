"""G2 C1: data coverage, readout reliability, mask, matched-goal set.

No world-model training. Does not write into G0 or G1 directories.
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

from stable_worldmodel.wm.spatial.data import candidate_starts  # noqa: E402
from stable_worldmodel.wm.spatial.model import FrozenDINOv2  # noqa: E402

OUT = ROOT / 'results/stage3/g2_c1_audit'
G1_EPISODES = ROOT / 'results/stage3/g1_spatial/overfit500/episodes.json'
WEIGHTS = ROOT / 'results/stage3/g0_dinov2/weights'
PATHS = {
    'pusht': '/workspace/datasets/lewm-pusht/train.lance',
    'tworoom': '/workspace/datasets/lewm-tworooms/train.lance',
    'cube': '/workspace/datasets/lewm-cube/train.lance',
}
SEED = 3072
FRAMESKIP = 5
SPAN = 10
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# World-unit thresholds. PushT lives in the 512 pygame frame.
# These are descriptive bins, not contact labels.
DA_MOVE = 15.0
DA_STILL = 5.0
DB_MOVE = 5.0
DB_STILL = 2.0
DANG_MOVE = 10.0
DANG_STILL = 5.0
# Center distance. Agent circle radius is 0.375 * scale 40 = 15.
# The T (scale 30) extends roughly 60-75 from its body origin, so a
# center distance is only a proximity proxy.
NEAR_BINS = (40.0, 60.0, 80.0)
AGENT_RADIUS = 15.0
BLOCK_SCALE = 30.0


def log(message: str):
    print(message, flush=True)


def episode_bounds_from_ids(episode_idx: np.ndarray):
    change = np.flatnonzero(np.diff(episode_idx)) + 1
    starts = np.concatenate([[0], change]).astype(np.int64)
    ends = np.concatenate([change, [len(episode_idx)]]).astype(np.int64)
    return starts, ends


def load_columns(path: str, columns: list[str], batch_size: int = 200_000):
    import lance

    dataset = lance.dataset(path)
    total = dataset.count_rows()
    parts = {name: [] for name in columns}
    seen = 0
    bar = tqdm(total=total, desc=f'load {Path(path).parent.name}', unit='row', mininterval=0.5)
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        n = batch.num_rows
        for name in columns:
            column = batch.column(name)
            if str(column.type).startswith('fixed_size_list') or 'list' in str(column.type):
                width = column.type.list_size
                flat = column.flatten().to_numpy(zero_copy_only=False)
                parts[name].append(np.asarray(flat, dtype=np.float32).reshape(n, width))
            else:
                parts[name].append(np.asarray(column.to_numpy(zero_copy_only=False)))
        seen += n
        bar.update(n)
    bar.close()
    out = {}
    for name in columns:
        chunks = parts[name]
        out[name] = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
    return out


def wrapped_deg(a0, a1):
    delta = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
    return np.abs(np.degrees(delta))


def transition_views(state, starts, ends, k=FRAMESKIP):
    """Pair row t with t+k inside the same episode."""
    n = len(state)
    valid = np.zeros(n - k, dtype=bool)
    for s, e in zip(starts, ends):
        if e - s > k:
            valid[s : e - k] = True
    idx = np.flatnonzero(valid)
    s0 = state[idx]
    s1 = state[idx + k]
    return idx, s0, s1


def pusht_kinematics(s0, s1):
    da = np.linalg.norm(s1[:, 0:2] - s0[:, 0:2], axis=1)
    db = np.linalg.norm(s1[:, 2:4] - s0[:, 2:4], axis=1)
    dang = wrapped_deg(s0[:, 4], s1[:, 4])
    dist0 = np.linalg.norm(s0[:, 0:2] - s0[:, 2:4], axis=1)
    dist1 = np.linalg.norm(s1[:, 0:2] - s1[:, 2:4], axis=1)
    return da, db, dang, dist0, dist1


def rate(mask):
    return float(np.mean(mask)) if len(mask) else float('nan')


def class_masks(da, db, dang, dist0, dist1):
    static = (da < DA_STILL) & (db < DB_STILL) & (dang < DANG_STILL)
    agent_only = (da >= DA_MOVE) & (db < DB_STILL) & (dang < DANG_STILL)
    block_translate = db >= DB_MOVE
    block_rotate = dang >= DANG_MOVE
    both = (da >= DA_MOVE) & (db >= DB_MOVE)
    near = {}
    approach = {}
    separate = {}
    stay = {}
    for thr in NEAR_BINS:
        near[thr] = (dist0 <= thr) | (dist1 <= thr)
        approach[thr] = (dist0 > thr + 20) & (dist1 <= thr)
        separate[thr] = (dist0 <= thr) & (dist1 > thr + 20)
        stay[thr] = (dist0 <= thr) & (dist1 <= thr)
    return {
        'static': static,
        'agent_only': agent_only,
        'block_translate': block_translate,
        'block_rotate': block_rotate,
        'agent_and_block': both,
        'near': near,
        'approach': approach,
        'separate': separate,
        'stay_near': stay,
    }


def quantile_dict(values):
    qs = [0, 1, 5, 25, 50, 75, 90, 95, 99, 100]
    return {f'p{q}': float(np.percentile(values, q)) for q in qs}


def summarize_pusht(da, db, dang, dist0, dist1):
    masks = class_masks(da, db, dang, dist0, dist1)
    summary = {
        'n': int(len(da)),
        'agent_delta': quantile_dict(da),
        'block_delta': quantile_dict(db),
        'angle_delta_deg': quantile_dict(dang),
        'center_distance': quantile_dict(dist0),
        'rates': {
            'static': rate(masks['static']),
            'agent_only': rate(masks['agent_only']),
            'block_translate': rate(masks['block_translate']),
            'block_rotate': rate(masks['block_rotate']),
            'agent_and_block': rate(masks['agent_and_block']),
        },
        'proximity_proxy': {},
    }
    for thr in NEAR_BINS:
        summary['proximity_proxy'][str(thr)] = {
            'either_frame_within': rate(masks['near'][thr]),
            'approach': rate(masks['approach'][thr]),
            'separate': rate(masks['separate'][thr]),
            'stay_near': rate(masks['stay_near'][thr]),
            'note': 'center-distance proxy, not a contact label',
        }
    # Threshold sensitivity so the bias claim is not one cutoff.
    sensitivity = []
    for db_thr in (1.0, 2.0, 5.0, 10.0):
        for da_thr in (5.0, 15.0, 30.0):
            sensitivity.append(
                {
                    'da_thr': da_thr,
                    'db_thr': db_thr,
                    'agent_only': rate((da >= da_thr) & (db < db_thr) & (dang < DANG_STILL)),
                    'block_translate': rate(db >= db_thr),
                }
            )
    summary['threshold_sensitivity'] = sensitivity
    return summary, masks


def g1_bias(starts, ends, state, episodes_doc):
    rows = []
    chosen_da, chosen_db = [], []
    best_db, best_da_at_block = [], []
    rank_of_chosen_by_db = []
    missed_block = 0
    matched_rule = 0
    for split in ('train', 'test'):
        for ep, start, motion in zip(
            episodes_doc['pusht'][split]['episode'],
            episodes_doc['pusht'][split]['start'],
            episodes_doc['pusht'][split]['motion'],
        ):
            length = int(ends[ep] - starts[ep])
            cands = candidate_starts(length, SPAN, k=16)
            base = int(starts[ep])
            das, dbs, dangs, scores = [], [], [], []
            for step in cands.tolist():
                s0 = state[base + int(step)]
                s1 = state[base + int(step) + FRAMESKIP]
                da = float(np.linalg.norm(s1[:2] - s0[:2]))
                db = float(np.linalg.norm(s1[2:4] - s0[2:4]))
                dang = float(wrapped_deg(np.array([s0[4]]), np.array([s1[4]]))[0])
                das.append(da)
                dbs.append(db)
                dangs.append(dang)
                scores.append(float(np.hypot(da, db)))
            das = np.asarray(das)
            dbs = np.asarray(dbs)
            scores = np.asarray(scores)
            pick = int(np.argmax(scores))
            if int(cands[pick]) == int(start):
                matched_rule += 1
            # Actual stored clip, even if the candidate grid differs.
            s0 = state[base + int(start)]
            s1 = state[base + int(start) + FRAMESKIP]
            da = float(np.linalg.norm(s1[:2] - s0[:2]))
            db = float(np.linalg.norm(s1[2:4] - s0[2:4]))
            dang = float(wrapped_deg(np.array([s0[4]]), np.array([s1[4]]))[0])
            chosen_da.append(da)
            chosen_db.append(db)
            j = int(np.argmax(dbs))
            best_db.append(float(dbs[j]))
            best_da_at_block.append(float(das[j]))
            # 1 = largest block motion among the 16 candidates.
            rank = int(1 + np.sum(dbs > db + 1e-6))
            rank_of_chosen_by_db.append(rank)
            if db < DB_MOVE and np.any(dbs >= DB_MOVE):
                missed_block += 1
            rows.append(
                {
                    'split': split,
                    'episode': int(ep),
                    'start': int(start),
                    'recorded_motion': float(motion),
                    'agent_delta': da,
                    'block_delta': db,
                    'angle_delta_deg': dang,
                    'best_block_delta_among_candidates': float(dbs[j]),
                    'block_rank_among_16': rank,
                }
            )
    chosen_da = np.asarray(chosen_da)
    chosen_db = np.asarray(chosen_db)
    return {
        'n': len(rows),
        'rule_match_fraction': matched_rule / max(len(rows), 1),
        'median_agent_delta': float(np.median(chosen_da)),
        'median_block_delta': float(np.median(chosen_db)),
        'median_best_block_delta_on_same_grid': float(np.median(best_db)),
        'fraction_block_below_5': float(np.mean(chosen_db < DB_MOVE)),
        'fraction_agent_only': float(
            np.mean(
                (chosen_da >= DA_MOVE)
                & (chosen_db < DB_STILL)
                & (np.array([r['angle_delta_deg'] for r in rows]) < DANG_STILL)
            )
        ),
        'fraction_block_translate': float(np.mean(chosen_db >= DB_MOVE)),
        'fraction_where_a_candidate_moved_the_block_and_was_not_picked': missed_block
        / max(len(rows), 1),
        'median_block_rank_among_16': float(np.median(rank_of_chosen_by_db)),
        'clips': rows,
    }


def split_ids(episodes, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(episodes))
    n = len(episodes)
    n_train = int(round(n * 0.7))
    n_val = int(round(n * 0.15))
    order = np.asarray(episodes)[perm]
    return {
        'train': order[:n_train].astype(int).tolist(),
        'val': order[n_train : n_train + n_val].astype(int).tolist(),
        'test': order[n_train + n_val :].astype(int).tolist(),
    }


def fill_strata(records_by_episode, order, quotas):
    """records_by_episode[ep][stratum] = dict or None. Rare strata are filled first."""
    used = set()
    pool = {name: [] for name in quotas}
    for name in quotas:
        for ep in order:
            if ep in used:
                continue
            rec = records_by_episode[ep].get(name)
            if rec is None:
                continue
            if len(pool[name]) >= quotas[name]:
                break
            pool[name].append(rec)
            used.add(ep)
    return pool


def pusht_episode_options(starts, ends, state):
    out = {}
    k = FRAMESKIP
    for ep, (s, e) in enumerate(zip(starts, ends)):
        if e - s <= k:
            continue
        s0 = state[s : e - k]
        s1 = state[s + k : e]
        da, db, dang, dist0, dist1 = pusht_kinematics(s0, s1)
        steps = np.arange(e - s - k)
        def take(mask, score):
            if not np.any(mask):
                return None
            j = int(np.argmax(np.where(mask, score, -1e9)))
            return {
                'episode': int(ep),
                'start': int(steps[j]),
                'agent_delta': float(da[j]),
                'block_delta': float(db[j]),
                'angle_delta_deg': float(dang[j]),
                'center_distance': float(dist0[j]),
            }
        out[ep] = {
            'block_rotate': take(dang >= DANG_MOVE, dang),
            'block_translate': take(db >= DB_MOVE, db),
            'proximity_proxy': take((dist0 <= 60) & (dist1 <= 60), -dist0),
            'agent_only': take(
                (da >= DA_MOVE) & (db < DB_STILL) & (dang < DANG_STILL), da
            ),
            'static': take(
                (da < DA_STILL) & (db < DB_STILL) & (dang < DANG_STILL), -da - db
            ),
        }
    return out


def other_episode_options(starts, ends, values, kind):
    """values is (N, D) positions. kind is 'tworoom' or 'cube'."""
    out = {}
    k = FRAMESKIP
    move_thr = 8.0 if kind == 'tworoom' else 0.02
    still_thr = 2.0 if kind == 'tworoom' else 0.005
    for ep, (s, e) in enumerate(zip(starts, ends)):
        if e - s <= k:
            continue
        delta = np.linalg.norm(values[s + k : e] - values[s : e - k], axis=1)
        steps = np.arange(e - s - k)
        def take(mask, score):
            if not np.any(mask):
                return None
            j = int(np.argmax(np.where(mask, score, -1e9)))
            return {
                'episode': int(ep),
                'start': int(steps[j]),
                'motion': float(delta[j]),
            }
        out[ep] = {
            'moving': take(delta >= move_thr, delta),
            'static': take(delta < still_thr, -delta),
        }
    return out


def pool_from_options(options, quotas, seed):
    episodes = np.array(sorted(options), dtype=np.int64)
    # Fill rare strata first, then split the selected episodes.
    order = np.random.default_rng(seed).permutation(episodes)
    # Priority is the insertion order of quotas.
    pool = fill_strata(options, order.tolist(), quotas)
    selected = sorted({rec['episode'] for rows in pool.values() for rec in rows})
    splits = split_ids(selected, seed)
    split_of = {}
    for name, ids in splits.items():
        for ep in ids:
            split_of[ep] = name
    for rows in pool.values():
        for rec in rows:
            rec['split'] = split_of[rec['episode']]
    counts = {name: len(rows) for name, rows in pool.items()}
    by_split = {
        split: {name: sum(r['split'] == split for r in rows) for name, rows in pool.items()}
        for split in ('train', 'val', 'test')
    }
    return {'clips': pool, 'counts': counts, 'counts_by_split': by_split, 'n_episodes': len(selected)}


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
    for start in tqdm(range(0, len(rows), 256), desc='decode', leave=False):
        chunk = [int(r) for r in rows[start : start + 256]]
        table = dataset.take(chunk, columns=['pixels'])
        blobs = table['pixels'].to_pylist()
        for i, blob in enumerate(blobs):
            images[start + i] = decode_image(blob)
    return np.stack(images)


@torch.no_grad()
def encode_images(encoder, images, device, batch=64):
    parts = []
    for start in tqdm(range(0, len(images), batch), desc='dino', leave=False):
        pixels = preprocess(images[start : start + batch], device)
        tokens = encoder.encode(pixels)[:, 0].float().cpu()
        if tokens.shape[1:] != (256, 384):
            raise RuntimeError(tuple(tokens.shape))
        parts.append(tokens)
    return torch.cat(parts, dim=0)


def pool4(tokens):
    n, p, d = tokens.shape
    side = int(round(p**0.5))
    grid = tokens.reshape(n, side, side, d).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, (4, 4))
    return pooled.permute(0, 2, 3, 1).reshape(n, -1)


def dual_ridge(x_fit, y_fit, x_eval, alpha):
    x_mu = x_fit.mean(0, keepdim=True)
    x_sd = x_fit.std(0, keepdim=True).clamp_min(1e-6)
    y_mu = y_fit.mean(0, keepdim=True)
    y_sd = y_fit.std(0, keepdim=True).clamp_min(1e-6)
    xt = (x_fit - x_mu) / x_sd
    yt = (y_fit - y_mu) / y_sd
    xe = (x_eval - x_mu) / x_sd
    gram = xt @ xt.T
    coef = torch.linalg.solve(gram + float(alpha) * torch.eye(len(xt), device=xt.device), yt)
    pred = xe @ xt.T @ coef
    return pred * y_sd + y_mu


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def angle_mae(pred_sc, true_sc):
    pred = np.arctan2(pred_sc[:, 0], pred_sc[:, 1])
    true = np.arctan2(true_sc[:, 0], true_sc[:, 1])
    err = np.arctan2(np.sin(pred - true), np.cos(pred - true))
    return float(np.degrees(np.abs(err)).mean())


def fit_ridge_group(features, targets, groups, kind):
    """features/targets are torch (N, ...). groups is a dict of index arrays."""
    device = features.device
    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
    fit_i = torch.as_tensor(groups['train'], device=device)
    val_i = torch.as_tensor(groups['val'], device=device)
    best_alpha, best_val = None, 1e9
    y_val = targets[val_i].float().cpu().numpy()
    for alpha in alphas:
        pred = dual_ridge(features[fit_i], targets[fit_i], features[val_i], alpha)
        err = mae(pred.float().cpu().numpy(), y_val)
        if kind == 'angle':
            err = angle_mae(pred.float().cpu().numpy(), y_val)
        if err < best_val:
            best_val = err
            best_alpha = alpha
    out = {'alpha': best_alpha, 'val_select': best_val}
    for name, index in groups.items():
        pred = dual_ridge(
            features[fit_i], targets[fit_i], features[torch.as_tensor(index, device=device)], best_alpha
        ).float().cpu().numpy()
        true = targets[torch.as_tensor(index, device=device)].float().cpu().numpy()
        out[name] = {
            'mae': angle_mae(pred, true) if kind == 'angle' else mae(pred, true),
            'pred': pred,
            'true': true,
        }
    # A deliberately strong ridge, not chosen to minimize train error.
    strong = {}
    for name, index in groups.items():
        pred = dual_ridge(
            features[fit_i],
            targets[fit_i],
            features[torch.as_tensor(index, device=device)],
            100.0,
        ).float().cpu().numpy()
        true = targets[torch.as_tensor(index, device=device)].float().cpu().numpy()
        strong[name] = angle_mae(pred, true) if kind == 'angle' else mae(pred, true)
        if name == 'test':
            strong['pred'] = pred
            strong['true'] = true
    out['ridge_alpha_100'] = strong
    return out


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
        # tokens (B, 256, 384)
        x = tokens.reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
        return self.net(x)


def train_tiny(tokens, targets, groups, kind, steps=80):
    device = torch.device('cuda')
    model = TinySpatialReadout(targets.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_i = np.asarray(groups['train'])
    val_i = np.asarray(groups['val'])
    y = targets.to(device)
    x = tokens.to(device)
    best_state, best_val, bad = None, 1e9, 0
    history = []
    for step in range(steps):
        model.train()
        take = train_i[np.random.default_rng(SEED + step).choice(len(train_i), size=min(64, len(train_i)), replace=False)]
        pred = model(x[torch.as_tensor(take, device=device)])
        loss = (pred - y[torch.as_tensor(take, device=device)]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(x[torch.as_tensor(val_i, device=device)]).float().cpu().numpy()
            val_true = targets[val_i].numpy()
            err = angle_mae(val_pred, val_true) if kind == 'angle' else mae(val_pred, val_true)
            history.append({'step': step, 'val': err, 'batch_loss': float(loss.detach())})
            if err < best_val - 1e-5:
                best_val = err
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    result = {'best_val': best_val, 'history': history}
    with torch.no_grad():
        for name, index in groups.items():
            pred = model(x[torch.as_tensor(index, device=device)]).float().cpu().numpy()
            true = targets[np.asarray(index)].numpy()
            result[name] = {
                'mae': angle_mae(pred, true) if kind == 'angle' else mae(pred, true),
                'pred': pred,
                'true': true,
            }
    return result


def delta_report(pred_now, pred_fut, true_now, true_fut, kind):
    if kind == 'angle':
        def ang(sc):
            return np.arctan2(sc[:, 0], sc[:, 1])
        true_d = np.arctan2(np.sin(ang(true_fut) - ang(true_now)), np.cos(ang(true_fut) - ang(true_now)))
        pred_d = np.arctan2(np.sin(ang(pred_fut) - ang(pred_now)), np.cos(ang(pred_fut) - ang(pred_now)))
        err = np.degrees(np.abs(np.arctan2(np.sin(pred_d - true_d), np.cos(pred_d - true_d))))
        persist = np.degrees(np.abs(true_d))
        return {
            'delta_mae': float(err.mean()),
            'true_motion_mae': float(persist.mean()),
            'fraction_delta_closer_than_zero': float(np.mean(err < persist)),
        }
    true_d = true_fut - true_now
    pred_d = pred_fut - pred_now
    err = np.linalg.norm(pred_d - true_d, axis=1)
    motion = np.linalg.norm(true_d, axis=1)
    if np.std(motion) < 1e-8 or np.std(err) < 1e-8:
        spear = float('nan')
    else:
        spear = float(np.corrcoef(np.argsort(np.argsort(motion)), np.argsort(np.argsort(np.linalg.norm(pred_d, axis=1))))[0, 1])
    return {
        'delta_mae': float(err.mean()),
        'true_motion_mae': float(motion.mean()),
        'fraction_delta_closer_than_zero': float(np.mean(err < motion)),
        'spearman_pred_vs_true_motion': spear,
    }


def binned_delta(pred_now, pred_fut, true_now, true_fut, bins):
    motion = np.linalg.norm(true_fut - true_now, axis=1)
    pred_d = pred_fut - pred_now
    true_d = true_fut - true_now
    err = np.linalg.norm(pred_d - true_d, axis=1)
    rows = []
    edges = [0.0, *bins, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (motion >= lo) & (motion < hi)
        if int(mask.sum()) < 5:
            continue
        rows.append(
            {
                'lo': lo,
                'hi': None if hi > 1e8 else hi,
                'n': int(mask.sum()),
                'mean_motion': float(motion[mask].mean()),
                'delta_mae': float(err[mask].mean()),
                'zero_delta_mae': float(motion[mask].mean()),
                'beats_zero': bool(err[mask].mean() < motion[mask].mean()),
            }
        )
    return rows


def reliable(binned, spearman):
    """A readout can judge small motion only if its delta beats predicting zero there."""
    small = [row for row in binned if row['lo'] >= 2 and row.get('hi') is not None and row['hi'] <= 8]
    if not small:
        return False
    return all(row['beats_zero'] for row in small) and (
        spearman == spearman and spearman >= 0.5
    )


def sample_pairs(pool, stratum, split, n_episodes, rng):
    rows = [r for r in pool['clips'][stratum] if r['split'] == split]
    if len(rows) > n_episodes:
        take = rng.choice(len(rows), size=n_episodes, replace=False)
        rows = [rows[int(i)] for i in take]
    return rows[:n_episodes]


def collect_readout_indices(starts, pool, spec, rng):
    """spec: list of (stratum, split, n_episodes). One clip per episode already."""
    chosen = []
    seen = set()
    for stratum, split, n_ep in spec:
        for rec in sample_pairs(pool, stratum, split, n_ep, rng):
            ep = rec['episode']
            if ep in seen:
                continue
            seen.add(ep)
            row = int(starts[ep] + rec['start'])
            chosen.append({**rec, 'row': row, 'row_next': row + FRAMESKIP, 'stratum': stratum})
    return chosen


def run_readout(name, tokens_now, tokens_fut, y_now, y_fut, groups, kind, bins):
    features = pool4(tokens_now).cuda()
    features_f = pool4(tokens_fut).cuda()
    targets = torch.from_numpy(y_now).cuda()
    targets_f = torch.from_numpy(y_fut).cuda()
    ridge = fit_ridge_group(features, targets, groups, kind)
    tiny = train_tiny(tokens_now, torch.from_numpy(y_now), groups, kind)
    # Delta uses the test pairs only.
    test = np.asarray(groups['test'])
    report = {
        'g0_pool4_ridge': {k: ridge[k]['mae'] if isinstance(ridge[k], dict) and 'mae' in ridge[k] else ridge[k] for k in ('alpha', 'train', 'val', 'test')},
        'ridge_alpha_100_test': ridge['ridge_alpha_100']['test'],
        'tiny_spatial': {k: tiny[k]['mae'] for k in ('train', 'val', 'test')},
    }
    # Drop bulky preds from the mae summary; keep test preds for delta.
    for label, pack, feat_f in (
        ('g0_pool4_ridge', ridge, features_f),
        ('tiny_spatial', tiny, None),
    ):
        pred_now = pack['test']['pred']
        true_now = pack['test']['true']
        true_fut = y_fut[test]
        if label == 'g0_pool4_ridge':
            pred_fut = dual_ridge(
                features[torch.as_tensor(groups['train'], device=features.device)],
                targets[torch.as_tensor(groups['train'], device=features.device)],
                feat_f[torch.as_tensor(test, device=features.device)],
                ridge['alpha'],
            ).float().cpu().numpy()
        else:
            # Recompute future preds with the trained head by calling train result's model
            # is not returned. Fit is enough if we store fut preds inside train_tiny.
            pred_fut = pack['test_future']['pred']
        report[label]['absolute_test'] = pack['test']['mae']
        report[label]['feature_persistence'] = (
            angle_mae(pred_now, true_fut) if kind == 'angle' else mae(pred_now, true_fut)
        )
        report[label]['true_state_persistence'] = (
            angle_mae(true_now, true_fut) if kind == 'angle' else mae(true_now, true_fut)
        )
        report[label]['ceiling_on_future_frame'] = (
            angle_mae(pred_fut, true_fut) if kind == 'angle' else mae(pred_fut, true_fut)
        )
        report[label]['delta'] = delta_report(pred_now, pred_fut, true_now, true_fut, kind)
        if kind == 'xy' and bins is not None:
            report[label]['delta_bins'] = binned_delta(pred_now, pred_fut, true_now, true_fut, bins)
            report[label]['small_motion_reliable'] = reliable(
                report[label]['delta_bins'], report[label]['delta'].get('spearman_pred_vs_true_motion', float('nan'))
            )
    return report


def attach_future(pack, tokens_fut, y_fut, groups, kind):
    """Train the tiny head once more is wasteful. Predict future with a second forward stored here."""
    return pack


def train_tiny_both(tokens_now, tokens_fut, targets_now, groups, kind):
    trained = train_tiny(tokens_now, torch.from_numpy(targets_now), groups, kind)
    # Replay the best head. train_tiny already dropped the module, so predict
    # the future frame by a second call that shares weights. Easiest: extend
    # train_tiny. This wrapper re-runs; callers should use train_tiny_paired.
    return trained


def train_tiny_paired(tokens_now, tokens_fut, y_now, groups, kind):
    device = torch.device('cuda')
    targets = torch.from_numpy(np.ascontiguousarray(y_now))
    model = TinySpatialReadout(targets.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_i = np.asarray(groups['train'])
    val_i = np.asarray(groups['val'])
    x = tokens_now.to(device)
    y = targets.to(device)
    best_state, best_val, bad = None, 1e9, 0
    for step in range(80):
        model.train()
        take = train_i[
            np.random.default_rng(SEED + step).choice(len(train_i), size=min(64, len(train_i)), replace=False)
        ]
        pred = model(x[torch.as_tensor(take, device=device)])
        loss = (pred - y[torch.as_tensor(take, device=device)]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == 79:
            model.eval()
            with torch.no_grad():
                val_pred = model(x[torch.as_tensor(val_i, device=device)]).float().cpu().numpy()
            err = angle_mae(val_pred, targets[val_i].numpy()) if kind == 'angle' else mae(val_pred, targets[val_i].numpy())
            if err < best_val - 1e-5:
                best_val = err
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= 8 and step > 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    result = {}
    with torch.no_grad():
        fut = tokens_fut.to(device)
        for name, index in groups.items():
            idx = torch.as_tensor(index, device=device)
            pred = model(x[idx]).float().cpu().numpy()
            pred_f = model(fut[idx]).float().cpu().numpy()
            true = targets[np.asarray(index)].numpy()
            result[name] = {
                'mae': angle_mae(pred, true) if kind == 'angle' else mae(pred, true),
                'pred': pred,
            }
            result[name + '_future'] = {'pred': pred_f}
    result['test_future'] = result['test_future']
    return result


def readout_one(tokens_now, tokens_fut, y_now, y_fut, groups, kind, bins):
    features = pool4(tokens_now).cuda()
    features_f = pool4(tokens_fut).cuda()
    targets = torch.from_numpy(np.ascontiguousarray(y_now)).cuda()
    ridge = fit_ridge_group(features, targets, groups, kind)
    tiny = train_tiny_paired(tokens_now, tokens_fut, y_now, groups, kind)
    test = np.asarray(groups['test'])
    fit = torch.as_tensor(groups['train'], device=features.device)
    pred_fut = dual_ridge(features[fit], targets[fit], features_f[torch.as_tensor(test, device=features.device)], ridge['alpha'])
    pred_fut = pred_fut.float().cpu().numpy()

    def pack(name, pred_now, pred_future, true_now, true_future):
        body = {
            'absolute_mae': angle_mae(pred_now, true_now) if kind == 'angle' else mae(pred_now, true_now),
            'ceiling_mae': angle_mae(pred_future, true_future) if kind == 'angle' else mae(pred_future, true_future),
            'feature_persistence_mae': angle_mae(pred_now, true_future) if kind == 'angle' else mae(pred_now, true_future),
            'true_state_persistence_mae': angle_mae(true_now, true_future) if kind == 'angle' else mae(true_now, true_future),
            'delta': delta_report(pred_now, pred_future, true_now, true_future, kind),
        }
        if kind == 'xy':
            body['delta_bins'] = binned_delta(pred_now, pred_future, true_now, true_future, bins)
            body['small_motion_reliable'] = reliable(body['delta_bins'], body['delta'].get('spearman_pred_vs_true_motion', float('nan')))
        return body

    true_now = y_now[test]
    true_fut = y_fut[test]
    return {
        'ridge_alpha': ridge['alpha'],
        'ridge_mae': {split: ridge[split]['mae'] for split in ('train', 'val', 'test')},
        'ridge_alpha_100_mae': {split: ridge['ridge_alpha_100'][split] for split in ('train', 'val', 'test')},
        'tiny_mae': {split: tiny[split]['mae'] for split in ('train', 'val', 'test')},
        'ridge_test': pack('ridge', ridge['test']['pred'], pred_fut, true_now, true_fut),
        'tiny_test': pack('tiny', tiny['test']['pred'], tiny['test_future']['pred'], true_now, true_fut),
        'train_val_gap_ridge': ridge['train']['mae'] - ridge['val']['mae'],
    }


def tee_vertices(scale=BLOCK_SCALE):
    length = 4.0
    bar = np.array(
        [[-length * scale / 2, scale], [length * scale / 2, scale], [length * scale / 2, 0], [-length * scale / 2, 0]],
        dtype=np.float32,
    )
    stem = np.array(
        [[-scale / 2, scale], [-scale / 2, length * scale], [scale / 2, length * scale], [scale / 2, scale]],
        dtype=np.float32,
    )
    cog = (bar.mean(0) + stem.mean(0)) / 2
    return bar, stem, cog


def raster_mask(pos, angle, y_sign, rot_sign, size=224):
    import cv2

    bar, stem, cog = tee_vertices()
    canvas = np.zeros((size, size), np.uint8)
    scale = size / 512.0
    for poly in (bar, stem):
        local = poly.copy()
        local[:, 1] *= y_sign
        c = cog.copy()
        c[1] *= y_sign
        rel = local - c
        theta = rot_sign * angle
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        world = rel @ rot.T + np.asarray(pos, dtype=np.float32)
        pix = np.round(world * scale).astype(np.int32)
        cv2.fillPoly(canvas, [pix.reshape(-1, 1, 2)], 1)
    agent = np.zeros_like(canvas)
    center = np.round(np.asarray(pos, dtype=np.float32) * 0).astype(int)  # placeholder replaced by caller
    return canvas, agent, center


def draw_agent(pos, size=224):
    import cv2

    canvas = np.zeros((size, size), np.uint8)
    center = tuple(np.round(np.asarray(pos, dtype=np.float32) * size / 512.0).astype(int))
    radius = max(1, int(round(AGENT_RADIUS * size / 512.0)))
    cv2.circle(canvas, center, radius, 1, thickness=-1)
    return canvas


def block_raster(pos, angle, y_sign, rot_sign, size=224):
    import cv2

    bar, stem, cog = tee_vertices()
    canvas = np.zeros((size, size), np.uint8)
    scale = size / 512.0
    theta = rot_sign * float(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    for poly in (bar, stem):
        local = poly.astype(np.float32).copy()
        local[:, 1] *= y_sign
        c = cog.astype(np.float32).copy()
        c[1] *= y_sign
        world = (local - c) @ rot.T + np.asarray(pos, dtype=np.float32)
        pix = np.round(world * scale).astype(np.int32)
        cv2.fillPoly(canvas, [pix.reshape(-1, 1, 2)], 1)
    return canvas


PALETTE = np.array(
    [
        [255, 255, 255],  # background
        [144, 238, 144],  # LightGreen goal
        [65, 105, 225],  # RoyalBlue agent
        [119, 136, 153],  # LightSlateGray block
    ],
    dtype=np.float32,
)
PALETTE_NAMES = ['background', 'goal', 'agent', 'block']


def rgb_labels(image):
    flat = image.reshape(-1, 3).astype(np.float32)
    dist = ((flat[:, None, :] - PALETTE[None, :, :]) ** 2).sum(-1)
    label = dist.argmin(axis=1).reshape(image.shape[:2])
    return label


def iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float('nan')
    return float(inter / union)


def token_fraction(mask):
    # 224 = 16 * 14
    cell = mask.astype(np.float32).reshape(16, 14, 16, 14).mean(axis=(1, 3))
    return cell


def mask_audit(images, states):
    # Pick the geometric convention with the best IoU against the gray class
    # on a handful of frames where the agent is far from the block.
    dist = np.linalg.norm(states[:, 0:2] - states[:, 2:4], axis=1)
    order = np.argsort(-dist)
    probe_idx = order[:8]
    conventions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    scores = []
    for y_sign, rot_sign in conventions:
        vals = []
        for i in probe_idx:
            geom = block_raster(states[i, 2:4], states[i, 4], y_sign, rot_sign)
            rgb = rgb_labels(images[i]) == 3
            vals.append(iou(geom, rgb))
        scores.append({'y_sign': y_sign, 'rot_sign': rot_sign, 'iou': float(np.nanmean(vals))})
    best = max(scores, key=lambda row: row['iou'] if row['iou'] == row['iou'] else -1)
    per = []
    for i in range(len(images)):
        geom = block_raster(states[i, 2:4], states[i, 4], best['y_sign'], best['rot_sign'])
        agent = draw_agent(states[i, 0:2])
        visible = np.logical_and(geom > 0, agent == 0)
        rgb = rgb_labels(images[i])
        rgb_block = rgb == 3
        rgb_agent = rgb == 2
        rgb_goal = rgb == 1
        frac = token_fraction(visible)
        token = frac >= 0.15
        covered = token.repeat(14, axis=0).repeat(14, axis=1)
        pixels = visible.sum()
        lost = np.logical_and(visible, ~covered).sum()
        occluded = np.logical_and(geom > 0, agent > 0).sum()
        per.append(
            {
                'iou_visible': iou(visible, rgb_block),
                'iou_raw_geom': iou(geom, rgb_block),
                'agent_overlap_pixels': int(occluded),
                'goal_pixels': int(rgb_goal.sum()),
                'rgb_agent_pixels': int(rgb_agent.sum()),
                'rgb_block_pixels': int(rgb_block.sum()),
                'geom_pixels': int(geom.sum()),
                'token_count': int(token.sum()),
                'pixel_fraction_lost_at_0.15': float(lost / max(pixels, 1)),
                'near': bool(dist[i] <= 60),
            }
        )
    far = [row for row in per if not row['near']]
    near = [row for row in per if row['near']]
    def avg(rows, key):
        if not rows:
            return None
        return float(np.mean([row[key] for row in rows]))
    summary = {
        'convention_search': scores,
        'chosen_convention': best,
        'n': len(per),
        'iou_visible_mean': avg(per, 'iou_visible'),
        'iou_visible_far': avg(far, 'iou_visible'),
        'iou_visible_near': avg(near, 'iou_visible'),
        'token_count_mean': avg(per, 'token_count'),
        'pixel_fraction_lost_mean': avg(per, 'pixel_fraction_lost_at_0.15'),
        'usable_for_loss': bool(best['iou'] >= 0.55 and avg(far, 'iou_visible') is not None and avg(far, 'iou_visible') >= 0.55),
    }
    return summary, best, per


def save_mask_figure(images, states, convention, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Six diverse frames: far, near, and the rest in order.
    dist = np.linalg.norm(states[:, 0:2] - states[:, 2:4], axis=1)
    far = np.argsort(-dist)[:3]
    near = np.argsort(dist)[:3]
    pick = list(dict.fromkeys([*far.tolist(), *near.tolist()]))[:6]
    fig, axes = plt.subplots(len(pick), 4, figsize=(8, 2.2 * len(pick)))
    for row, i in enumerate(pick):
        geom = block_raster(states[i, 2:4], states[i, 4], convention['y_sign'], convention['rot_sign'])
        agent = draw_agent(states[i, 0:2])
        visible = np.logical_and(geom > 0, agent == 0)
        rgb = rgb_labels(images[i])
        frac = token_fraction(visible)
        axes[row, 0].imshow(images[i])
        axes[row, 1].imshow(visible, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].imshow(rgb == 3, cmap='gray', vmin=0, vmax=1)
        axes[row, 3].imshow(frac, cmap='magma', vmin=0, vmax=1)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            axes[row, 0].set_title('RGB')
            axes[row, 1].set_title('geometry, agent removed')
            axes[row, 2].set_title('RGB gray class')
            axes[row, 3].set_title('16x16 coverage')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_coverage_figure(da, db, g1, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    take = rng.choice(len(da), size=min(150_000, len(da)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hexbin(da[take], db[take], gridsize=40, bins='log', cmap='Blues', mincnt=1)
    axes[0].scatter(
        [r['agent_delta'] for r in g1['clips']],
        [r['block_delta'] for r in g1['clips']],
        s=12,
        c='crimson',
        label='G1 clips',
    )
    axes[0].set_xlabel('agent move')
    axes[0].set_ylabel('block move')
    axes[0].set_xlim(0, 200)
    axes[0].set_ylim(0, 80)
    axes[0].legend(loc='upper right')
    labels = ['static', 'agent_only', 'block_translate', 'block_rotate']
    # rates filled by caller via the title only; bars are passed in g1 summary separately
    axes[1].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_rate_figure(full_rates, g1_rates, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keys = list(full_rates)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, [full_rates[k] for k in keys], width=0.36, label='all transitions')
    ax.bar(x + 0.18, [g1_rates[k] for k in keys], width=0.36, label='G1 clips')
    ax.set_xticks(x, keys, rotation=20, ha='right')
    ax.set_ylabel('fraction')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def json_dump(path, payload):
    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (np.floating, float)):
            number = float(value)
            return None if not np.isfinite(number) else number
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    path.write_text(json.dumps(convert(payload), indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'STATUS').write_text('RUNNING\n')
    log('C1 start')
    pusht = load_columns(PATHS['pusht'], ['episode_idx', 'state'])
    episode_idx = pusht['episode_idx'].astype(np.int64)
    state = pusht['state']
    starts, ends = episode_bounds_from_ids(episode_idx)
    log(f'pusht episodes {len(starts)} rows {len(state)}')
    idx, s0, s1 = transition_views(state, starts, ends)
    da, db, dang, dist0, dist1 = pusht_kinematics(s0, s1)
    summary, masks = summarize_pusht(da, db, dang, dist0, dist1)
    g1_doc = json.loads(G1_EPISODES.read_text())
    g1 = g1_bias(starts, ends, state, g1_doc)
    g1_rates = {
        'static': float(np.mean([
            r['agent_delta'] < DA_STILL and r['block_delta'] < DB_STILL and r['angle_delta_deg'] < DANG_STILL
            for r in g1['clips']
        ])),
        'agent_only': g1['fraction_agent_only'],
        'block_translate': g1['fraction_block_translate'],
        'block_rotate': float(np.mean([r['angle_delta_deg'] >= DANG_MOVE for r in g1['clips']])),
    }
    summary['g1'] = {k: v for k, v in g1.items() if k != 'clips'}
    summary['g1_rates'] = g1_rates
    summary['population_rates'] = summary['rates']
    json_dump(OUT / 'pusht_coverage.json', summary)
    json_dump(OUT / 'g1_clips_classified.json', g1['clips'])
    save_coverage_figure(da, db, g1, OUT / 'pusht_motion_scatter.png')
    save_rate_figure(summary['rates'], {
        'static': g1_rates['static'],
        'agent_only': g1_rates['agent_only'],
        'block_translate': g1_rates['block_translate'],
        'block_rotate': g1_rates['block_rotate'],
        'agent_and_block': float('nan'),
    }, OUT / 'pusht_class_rates.png')
    # The rate figure used nan for a key the population has. Rewrite cleanly.
    save_rate_figure(
        {k: summary['rates'][k] for k in ('static', 'agent_only', 'block_translate', 'block_rotate')},
        {k: g1_rates[k] for k in ('static', 'agent_only', 'block_translate', 'block_rotate')},
        OUT / 'pusht_class_rates.png',
    )
    log('building candidate pools')
    options = pusht_episode_options(starts, ends, state)
    quotas = {
        'block_rotate': 400,
        'block_translate': 400,
        'proximity_proxy': 300,
        'agent_only': 300,
        'static': 200,
    }
    pusht_pool = pool_from_options(options, quotas, SEED)
    # Free the big kinematic arrays before the other datasets.
    del idx, s0, s1, da, db, dang, dist0, dist1, masks, options
    tw = load_columns(PATHS['tworoom'], ['episode_idx', 'pos_agent', 'pos_target'])
    tw_starts, tw_ends = episode_bounds_from_ids(tw['episode_idx'].astype(np.int64))
    tw_opt = other_episode_options(tw_starts, tw_ends, tw['pos_agent'], 'tworoom')
    tw_pool = pool_from_options(tw_opt, {'moving': 400, 'static': 200}, SEED)
    cube = load_columns(
        PATHS['cube'],
        ['episode_idx', 'privileged_block_0_pos', 'proprio_effector_pos', 'privileged_target_block_pos', 'success'],
    )
    cu_starts, cu_ends = episode_bounds_from_ids(cube['episode_idx'].astype(np.int64))
    cu_opt = other_episode_options(cu_starts, cu_ends, cube['privileged_block_0_pos'], 'cube')
    cu_pool = pool_from_options(cu_opt, {'moving': 400, 'static': 200}, SEED)
    manifest = {
        'seed': SEED,
        'frameskip': FRAMESKIP,
        'span': SPAN,
        'note': (
            'Episode-disjoint within each environment. Strata are filled '
            'rarest-first and an episode is used once. proximity_proxy is '
            'center distance <= 60, not a contact label. Do not train from '
            'this file until the split field is respected.'
        ),
        'pusht_thresholds': {
            'agent_move': DA_MOVE,
            'block_move': DB_MOVE,
            'angle_deg': DANG_MOVE,
            'near': 60,
        },
        'pusht': {k: pusht_pool[k] for k in ('counts', 'counts_by_split', 'n_episodes', 'clips')},
        'tworoom': {k: tw_pool[k] for k in ('counts', 'counts_by_split', 'n_episodes', 'clips')},
        'cube': {k: cu_pool[k] for k in ('counts', 'counts_by_split', 'n_episodes', 'clips')},
    }
    json_dump(OUT / 'candidate_pool.json', manifest)
    log(f"pools pusht {pusht_pool['counts']} tworoom {tw_pool['counts']} cube {cu_pool['counts']}")

    # Readout sample. Keep it small and stratified, episode-disjoint by construction.
    rng = np.random.default_rng(SEED)
    device = torch.device('cuda')
    encoder = FrozenDINOv2.from_pretrained(str(WEIGHTS)).to(device)
    readout = {}
    specs = {
        'pusht': [
            ('block_translate', 'train', 40),
            ('block_rotate', 'train', 20),
            ('agent_only', 'train', 20),
            ('block_translate', 'val', 12),
            ('block_rotate', 'val', 8),
            ('block_translate', 'test', 24),
            ('block_rotate', 'test', 12),
            ('agent_only', 'test', 12),
        ],
    }
    pusht_items = collect_readout_indices(starts, pusht_pool, specs['pusht'], rng)
    groups = {'train': [], 'val': [], 'test': []}
    for i, item in enumerate(pusht_items):
        groups[item['split']].append(i)
    for split, ids in groups.items():
        if len(ids) < 8:
            raise RuntimeError(f'pusht readout split {split} has {len(ids)} clips')
    overlap = set(pusht_items[i]['episode'] for i in groups['train']) & set(
        pusht_items[i]['episode'] for i in groups['val']
    ) | set(pusht_items[i]['episode'] for i in groups['train']) & set(
        pusht_items[i]['episode'] for i in groups['test']
    ) | set(pusht_items[i]['episode'] for i in groups['val']) & set(
        pusht_items[i]['episode'] for i in groups['test']
    )
    if overlap:
        raise RuntimeError(f'readout episode leakage {overlap}')
    rows = []
    for item in pusht_items:
        rows.extend([item['row'], item['row_next']])
    images = fetch_images(PATHS['pusht'], rows)
    tokens = encode_images(encoder, images, device)
    now = tokens[0::2]
    fut = tokens[1::2]
    y_now = np.stack([state[item['row']] for item in pusht_items])
    y_fut = np.stack([state[item['row_next']] for item in pusht_items])
    pusht_readout = {}
    for key, sl, kind, bins in (
        ('block_xy', slice(2, 4), 'xy', [2, 5, 8, 15, 30]),
        ('agent_xy', slice(0, 2), 'xy', [5, 15, 40, 80]),
        ('block_angle', slice(4, 5), 'angle', None),
    ):
        if kind == 'angle':
            ang0 = y_now[:, 4]
            ang1 = y_fut[:, 4]
            target0 = np.stack([np.sin(ang0), np.cos(ang0)], axis=1).astype(np.float32)
            target1 = np.stack([np.sin(ang1), np.cos(ang1)], axis=1).astype(np.float32)
        else:
            target0 = y_now[:, sl].astype(np.float32)
            target1 = y_fut[:, sl].astype(np.float32)
        pusht_readout[key] = readout_one(now, fut, target0, target1, groups, kind, bins)
    readout['pusht'] = {
        'n': {split: len(ids) for split, ids in groups.items()},
        'episode_overlap': [],
        'targets': pusht_readout,
    }
    # Keep a few images for the mask audit before releasing the array.
    mask_pick = []
    for want in ('block_translate', 'block_rotate', 'agent_only', 'proximity_proxy'):
        hits = [i for i, item in enumerate(pusht_items) if item['stratum'] == want]
        mask_pick.extend(hits[:8])
    mask_pick = list(dict.fromkeys(mask_pick))[:36]
    mask_images = images[np.array(mask_pick) * 2]
    mask_states = y_now[mask_pick]
    del images, tokens, now, fut
    mask_summary, convention, _per = mask_audit(mask_images, mask_states)
    save_mask_figure(mask_images, mask_states, convention, OUT / 'mask_examples.png')
    json_dump(OUT / 'mask_audit.json', mask_summary)
    log(f"mask iou far {mask_summary['iou_visible_far']} usable {mask_summary['usable_for_loss']}")

    def env_readout(env, starts_e, pool, values_now_fn, columns_y):
        items = []
        spec = [('moving', 'train', 36), ('static', 'train', 12), ('moving', 'val', 12), ('moving', 'test', 24), ('static', 'test', 8)]
        items = collect_readout_indices(starts_e, pool, spec, rng)
        groups_e = {'train': [], 'val': [], 'test': []}
        for i, item in enumerate(items):
            groups_e[item['split']].append(i)
        eps = {split: {items[i]['episode'] for i in ids} for split, ids in groups_e.items()}
        if eps['train'] & eps['val'] or eps['train'] & eps['test'] or eps['val'] & eps['test']:
            raise RuntimeError(f'{env} readout episodes overlap')
        row_list = []
        for item in items:
            row_list.extend([item['row'], item['row_next']])
        imgs = fetch_images(PATHS[env], row_list)
        tok = encode_images(encoder, imgs, device)
        return items, groups_e, tok[0::2], tok[1::2]

    tw_items, tw_groups, tw_now, tw_fut = env_readout('tworoom', tw_starts, tw_pool, None, None)
    # positions from the already loaded column via row index
    tw_y0 = np.stack([tw['pos_agent'][item['row']] for item in tw_items]).astype(np.float32)
    tw_y1 = np.stack([tw['pos_agent'][item['row_next']] for item in tw_items]).astype(np.float32)
    readout['tworoom'] = {
        'n': {split: len(ids) for split, ids in tw_groups.items()},
        'agent_xy': readout_one(tw_now, tw_fut, tw_y0, tw_y1, tw_groups, 'xy', [2, 8, 20, 40]),
    }
    cu_items, cu_groups, cu_now, cu_fut = env_readout('cube', cu_starts, cu_pool, None, None)
    cu_read = {}
    for key, arr in (
        ('block_xyz', cube['privileged_block_0_pos']),
        ('effector_xyz', cube['proprio_effector_pos']),
    ):
        y0 = np.stack([arr[item['row']] for item in cu_items]).astype(np.float32)
        y1 = np.stack([arr[item['row_next']] for item in cu_items]).astype(np.float32)
        cu_read[key] = readout_one(cu_now, cu_fut, y0, y1, cu_groups, 'xy', [0.005, 0.02, 0.05, 0.1])
    readout['cube'] = {'n': {split: len(ids) for split, ids in cu_groups.items()}, **cu_read}
    json_dump(OUT / 'readout.json', readout)
    log('readout written')

    # Matched-goal pairs from pool test episodes only.
    def goal_pairs(starts_e, ends_e, episodes, k=25, limit=100):
        pairs = []
        for ep in episodes:
            length = int(ends_e[ep] - starts_e[ep])
            if length <= k + 1:
                continue
            goal_step = length // 2
            ctx_step = goal_step - k
            if ctx_step < 0:
                continue
            base = int(starts_e[ep])
            pairs.append({'episode': int(ep), 'context_row': base + ctx_step, 'goal_row': base + goal_step, 'offset': k})
            if len(pairs) >= limit:
                break
        return pairs

    test_eps = [ep for ep, recs in ((r['episode'], r) for rows in pusht_pool['clips'].values() for r in rows) if False]
    pusht_test = sorted({r['episode'] for rows in pusht_pool['clips'].values() for r in rows if r['split'] == 'test'})
    tw_test = sorted({r['episode'] for rows in tw_pool['clips'].values() for r in rows if r['split'] == 'test'})
    cu_test = sorted({r['episode'] for rows in cu_pool['clips'].values() for r in rows if r['split'] == 'test'})
    goal = {
        'definition': (
            'The planner yaml uses a later dataset frame as the goal image '
            '(goal_offset_steps 25). These pairs use offset 25 inside a '
            'held-out episode. They are not simulator rollouts. Cube success '
            'remains ||block-target|| <= 0.04 and does not include yaw. Cube '
            'pairs use the moving-stratum clip as context and the frame 25 '
            'steps later as the goal; the episode midpoint is already a '
            'success and does not move. The green PushT marker is a different '
            'target from the future frame.'
        ),
        'pusht': [],
        'tworoom': [],
        'cube': [],
    }
    green = np.array([226.93271166, 283.85171816], dtype=np.float64)
    for pair in goal_pairs(starts, ends, pusht_test):
        ctx = state[pair['context_row']]
        fut = state[pair['goal_row']]
        pair.update(
            {
                'block_distance_to_future_frame': float(np.linalg.norm(ctx[2:4] - fut[2:4])),
                'agent_distance_to_future_frame': float(np.linalg.norm(ctx[0:2] - fut[0:2])),
                'future_block_distance_to_green_marker': float(np.linalg.norm(fut[2:4] - green)),
                'context_block_distance_to_green_marker': float(np.linalg.norm(ctx[2:4] - green)),
                'future_frame_within_20_of_green': bool(np.linalg.norm(fut[2:4] - green) < 20),
            }
        )
        goal['pusht'].append(pair)
    for pair in goal_pairs(tw_starts, tw_ends, tw_test):
        ctx = tw['pos_agent'][pair['context_row']]
        fut = tw['pos_agent'][pair['goal_row']]
        target = tw['pos_target'][pair['goal_row']]
        pair.update(
            {
                'agent_distance_to_future_frame': float(np.linalg.norm(ctx - fut)),
                'context_distance_to_target': float(np.linalg.norm(ctx - target)),
                'future_distance_to_target': float(np.linalg.norm(fut - target)),
            }
        )
        goal['tworoom'].append(pair)
    cu_moving_test = sorted(
        (r for r in cu_pool['clips']['moving'] if r['split'] == 'test'),
        key=lambda r: r['episode'],
    )
    for rec in cu_moving_test:
        base = int(cu_starts[rec['episode']])
        ctx_row = base + int(rec['start'])
        goal_row = ctx_row + 25
        if goal_row >= int(cu_ends[rec['episode']]):
            continue
        b0 = cube['privileged_block_0_pos'][ctx_row]
        b1 = cube['privileged_block_0_pos'][goal_row]
        target = cube['privileged_target_block_pos'][goal_row]
        s0 = float(np.reshape(cube['success'][ctx_row], -1)[0])
        s1 = float(np.reshape(cube['success'][goal_row], -1)[0])
        d0 = float(np.linalg.norm(b0 - target))
        d1 = float(np.linalg.norm(b1 - target))
        goal['cube'].append(
            {
                'episode': int(rec['episode']),
                'context_row': int(ctx_row),
                'goal_row': int(goal_row),
                'offset': 25,
                'index': 'moving_clip',
                'context_block_to_target': d0,
                'goal_frame_block_to_target': d1,
                'block_distance_between_frames': float(np.linalg.norm(b0 - b1)),
                'context_success_column': s0,
                'goal_success_column': s1,
                'context_success_by_0.04': bool(d0 <= 0.04),
                'goal_success_by_0.04': bool(d1 <= 0.04),
            }
        )
    def mean_key(rows, key):
        if not rows:
            return None
        return float(np.mean([row[key] for row in rows]))
    goal['summary'] = {
        'pusht_n': len(goal['pusht']),
        'pusht_future_frame_near_green_fraction': mean_key(goal['pusht'], 'future_frame_within_20_of_green'),
        'pusht_median_block_distance_to_future': float(np.median([r['block_distance_to_future_frame'] for r in goal['pusht']])) if goal['pusht'] else None,
        'cube_n': len(goal['cube']),
        'cube_index': 'moving_clip_plus_25',
        'cube_goal_frame_success_fraction': mean_key(goal['cube'], 'goal_success_by_0.04'),
        'cube_context_success_fraction': mean_key(goal['cube'], 'context_success_by_0.04'),
        'cube_median_block_move': float(np.median([r['block_distance_between_frames'] for r in goal['cube']])) if goal['cube'] else None,
        'cube_success_column_matches_0.04': float(np.mean([
            (row['goal_success_column'] > 0.5) == row['goal_success_by_0.04'] for row in goal['cube']
        ])) if goal['cube'] else None,
        'tworoom_n': len(goal['tworoom']),
        'simulator_rollout': 'not run',
    }
    json_dump(OUT / 'matched_goal_set.json', goal)
    config = {
        'seed': SEED,
        'frameskip': FRAMESKIP,
        'pusht_bins': {
            'agent_move': DA_MOVE,
            'agent_still': DA_STILL,
            'block_move': DB_MOVE,
            'block_still': DB_STILL,
            'angle_move_deg': DANG_MOVE,
            'angle_still_deg': DANG_STILL,
            'near_bins': list(NEAR_BINS),
            'agent_radius': AGENT_RADIUS,
            'block_scale': BLOCK_SCALE,
        },
        'green_marker_xy_from_g0': green.tolist(),
        'g1_episodes': str(G1_EPISODES),
        'weights': str(WEIGHTS),
    }
    json_dump(OUT / 'config.json', config)
    (OUT / 'STATUS').write_text('DONE\n')
    log('C1_DONE')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        if OUT.exists():
            (OUT / 'STATUS').write_text('FAILED\n')
        raise
