"""G0: does frozen DINOv2 keep the state PushT, TwoRoom, and Cube need?

Diagnostic only. Does not train a world model and does not write checkpoints.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image
from transformers import Dinov2Model

ROOT = Path('/workspace/stable-worldmodel')
OUT = ROOT / 'results/stage3/g0_dinov2'
WEIGHTS = OUT / 'weights'
sys.path.insert(0, str(ROOT))

PATHS = {
    'pusht': '/workspace/datasets/lewm-pusht/train.lance',
    'tworoom': '/workspace/datasets/lewm-tworooms/train.lance',
    'cube': '/workspace/datasets/lewm-cube/train.lance',
}
LEWM_CKPT = ROOT / 'results/stage2a/rtx5090_stage2a/checkpoints/resume.ckpt'
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
N_EP = 100
FRAMES = 32
MIN_LEN = 48
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def r2_score(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
        pred = pred[:, None]
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    out = np.ones(y.shape[1])
    ok = ss_tot > 1e-12
    out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return out


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 5 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float('nan')

    def rank(x):
        order = np.argsort(x, kind='mergesort')
        r = np.empty(x.size, dtype=np.float64)
        r[order] = np.arange(x.size)
        return r

    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def decode_image(blob) -> np.ndarray:
    raw = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    with Image.open(io.BytesIO(raw)) as img:
        arr = np.asarray(img.convert('RGB'), dtype=np.uint8)
    return arr


def episode_bounds(dataset):
    ep = dataset.to_table(columns=['episode_idx']).column(0).to_numpy()
    change = np.flatnonzero(np.diff(ep)) + 1
    starts = np.concatenate([[0], change]).astype(np.int64)
    ends = np.concatenate([change, [len(ep)]]).astype(np.int64)
    return starts, ends


def choose_episodes(starts, ends):
    lengths = ends - starts
    long = np.flatnonzero(lengths >= MIN_LEN)
    pick = np.linspace(0, len(long) - 1, N_EP, dtype=int)
    chosen = long[pick]
    # unique, preserve order
    seen = set()
    out = []
    for ep in chosen.tolist():
        if ep not in seen:
            seen.add(ep)
            out.append(ep)
    return np.array(out, dtype=np.int64)


def sample_rows(starts, ends, episodes):
    rows = []
    ep_ids = []
    steps = []
    for ep in episodes:
        length = int(ends[ep] - starts[ep])
        idx = np.unique(np.round(np.linspace(0, length - 1, FRAMES)).astype(int))
        base = int(starts[ep])
        for s in idx.tolist():
            rows.append(base + s)
            ep_ids.append(int(ep))
            steps.append(int(s))
    return np.array(rows), np.array(ep_ids), np.array(steps)


def take_columns(dataset, rows, columns, batch=256):
    parts = []
    for start in range(0, len(rows), batch):
        part = dataset.take(rows[start : start + batch].tolist(), columns=columns)
        parts.append({name: part[name].to_pylist() for name in columns})
    merged = {name: [] for name in columns}
    for part in parts:
        for name in columns:
            merged[name].extend(part[name])
    return merged


def as_float_array(values):
    return np.asarray(values, dtype=np.float64)


def preprocess(images_u8, size):
    x = torch.from_numpy(images_u8).permute(0, 3, 1, 2).float().div(255.0)
    if size != 224:
        x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False, antialias=True)
        x = x.clamp(0, 1)
    x = (x - MEAN) / STD
    return x


@torch.no_grad()
def encode_dino(model, images_u8, size, batch=32):
    cls_out = []
    tok_out = []
    for start in range(0, len(images_u8), batch):
        pixels = preprocess(images_u8[start : start + batch], size).cuda()
        hidden = model(pixel_values=pixels).last_hidden_state
        cls_out.append(hidden[:, 0].float().cpu())
        tok_out.append(hidden[:, 1:].float().cpu())
    return torch.cat(cls_out), torch.cat(tok_out)


@torch.no_grad()
def encode_lewm(world, images_u8, batch=32):
    cls_out = []
    proj_out = []
    for start in range(0, len(images_u8), batch):
        pixels = preprocess(images_u8[start : start + batch], 224).cuda()
        hidden = world.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state
        cls = hidden[:, 0]
        proj = world.projector(cls)
        cls_out.append(cls.float().cpu())
        proj_out.append(proj.float().cpu())
    return torch.cat(cls_out), torch.cat(proj_out)


def random_features(images_u8, grid, dim, seed=3072):
    x = torch.from_numpy(images_u8).permute(0, 3, 1, 2).float().div(255.0)
    small = F.interpolate(x, size=(grid, grid), mode='area')
    flat = small.permute(0, 2, 3, 1).reshape(len(images_u8), -1)
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(flat.shape[1], dim, generator=g) / (flat.shape[1] ** 0.5)
    return flat @ weight


def pool_tokens(tokens, grid):
    # tokens (N, P, D) -> (N, grid*grid*D)
    n, p, d = tokens.shape
    side = int(round(p ** 0.5))
    if side * side != p:
        raise RuntimeError(f'patch count {p} is not square')
    image = tokens.reshape(n, side, side, d).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(image, (grid, grid))
    return pooled.permute(0, 2, 3, 1).reshape(n, -1)


def dual_ridge(x_train, y_train, x_eval, alpha):
    x_mu = x_train.mean(dim=0, keepdim=True)
    x_sd = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mu = y_train.mean(dim=0, keepdim=True)
    y_sd = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    xt = (x_train - x_mu) / x_sd
    yt = (y_train - y_mu) / y_sd
    xe = (x_eval - x_mu) / x_sd
    gram = xt @ xt.T
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coef = torch.linalg.solve(gram + float(alpha) * eye, yt)
    pred = xe @ xt.T @ coef
    return pred * y_sd + y_mu


def select_and_fit(features, targets, train_eps, inner_eps, episodes):
    """features/targets are torch CPU float tensors. episodes is numpy."""
    fit_mask = np.isin(episodes, train_eps) & ~np.isin(episodes, inner_eps)
    val_mask = np.isin(episodes, inner_eps)
    all_train = np.isin(episodes, train_eps)
    test_mask = ~all_train
    device = 'cuda'
    x = features.to(device)
    y = targets.to(device)
    fit_i = torch.as_tensor(np.flatnonzero(fit_mask), device=device)
    val_i = torch.as_tensor(np.flatnonzero(val_mask), device=device)
    tr_i = torch.as_tensor(np.flatnonzero(all_train), device=device)
    te_i = torch.as_tensor(np.flatnonzero(test_mask), device=device)
    best_alpha, best_score = None, -1e9
    y_val = y[val_i].float().cpu().numpy()
    for alpha in ALPHAS:
        pred = dual_ridge(x[fit_i], y[fit_i], x[val_i], alpha).float().cpu().numpy()
        score = float(np.nanmean(r2_score(y_val, pred)))
        if score > best_score:
            best_score = score
            best_alpha = alpha
    pred_te = dual_ridge(x[tr_i], y[tr_i], x[te_i], best_alpha).float().cpu().numpy()
    pred_tr = dual_ridge(x[tr_i], y[tr_i], x[tr_i], best_alpha).float().cpu().numpy()
    y_te = y[te_i].float().cpu().numpy()
    y_tr = y[tr_i].float().cpu().numpy()
    return {
        'alpha': best_alpha,
        'inner_val_r2': best_score,
        'train_r2': r2_score(y_tr, pred_tr).tolist(),
        'test_r2': r2_score(y_te, pred_te).tolist(),
        'test_mae': np.abs(y_te - pred_te).mean(axis=0).tolist(),
        'y_te': y_te,
        'pred_te': pred_te,
    }


def feature_table(tokens, cls, extra):
    # returns dict of CPU float tensors
    out = {
        'cls': cls.float(),
        'patch_mean': tokens.mean(dim=1).float(),
        'spatial_pool4': pool_tokens(tokens.float(), 4),
        'spatial_pool8': pool_tokens(tokens.float(), 8),
    }
    out.update(extra)
    return out


def target_specs(env, labels):
    specs = {}
    if env == 'pusht':
        state = labels['state']
        specs['agent_xy'] = state[:, 0:2]
        specs['block_xy'] = state[:, 2:4]
        ang = state[:, 4]
        specs['block_angle_sincos'] = np.stack([np.sin(ang), np.cos(ang)], axis=1)
    elif env == 'tworoom':
        specs['agent_xy'] = labels['pos_agent']
        specs['target_xy'] = labels['pos_target']
        obs = labels['observation']
        specs['door_xy'] = obs[:, 4:6]
        agent_side = labels['pos_agent'][:, 0] < 112
        target_side = labels['pos_target'][:, 0] < 112
        specs['same_side'] = (agent_side == target_side).astype(np.float64)[:, None]
        specs['distance'] = labels['distance']
    elif env == 'cube':
        specs['block_pos'] = labels['block_pos']
        yaw = labels['block_yaw'][:, 0]
        specs['block_yaw_sincos'] = np.stack([np.sin(yaw), np.cos(yaw)], axis=1)
        specs['effector_pos'] = labels['effector_pos']
        specs['gripper'] = labels['gripper']
        specs['target_pos'] = labels['target_pos']
        specs['pos_error'] = np.linalg.norm(
            labels['block_pos'] - labels['target_pos'], axis=1, keepdims=True
        )
        dyaw = np.arctan2(
            np.sin(labels['block_yaw'][:, 0] - labels['target_yaw'][:, 0]),
            np.cos(labels['block_yaw'][:, 0] - labels['target_yaw'][:, 0]),
        )
        specs['yaw_error'] = np.abs(dyaw)[:, None]
    return {k: torch.from_numpy(v.astype(np.float32)) for k, v in specs.items()}


def pack_fit(fit, names):
    packed = {
        'alpha': fit['alpha'],
        'inner_val_r2': fit['inner_val_r2'],
        'train_r2_mean': float(np.nanmean(fit['train_r2'])),
        'test_r2_mean': float(np.nanmean(fit['test_r2'])),
        'test_r2': fit['test_r2'],
        'test_mae': fit['test_mae'],
    }
    if 'angle' in names or names == ['sin', 'cos']:
        yt, yp = fit['y_te'], fit['pred_te']
        true = np.arctan2(yt[:, 0], yt[:, 1])
        pred = np.arctan2(yp[:, 0], yp[:, 1])
        err = np.arctan2(np.sin(true - pred), np.cos(true - pred))
        packed['test_angle_mae_deg'] = float(np.degrees(np.abs(err).mean()))
    if fit['y_te'].shape[1] >= 2 and 'angle' not in names:
        packed['test_l2_mae'] = float(
            np.linalg.norm(fit['y_te'] - fit['pred_te'], axis=1).mean()
        )
    return packed


def sensitivity(images, tokens, episodes, steps, grid_name):
    side = int(round(tokens.shape[1] ** 0.5))
    tok = tokens.float().reshape(len(tokens), side, side, -1).numpy()
    within = []
    across = []
    # within: neighboring samples in an episode, 4 to 12 raw steps apart
    order = np.argsort(episodes * 10_000_000 + steps)
    for a, b in zip(order[:-1], order[1:]):
        if episodes[a] != episodes[b]:
            continue
        gap = abs(int(steps[b] - steps[a]))
        if 4 <= gap <= 12:
            within.append((int(a), int(b)))
        if len(within) >= 80:
            break
    rng = np.random.default_rng(0)
    ep_to_idx = {}
    for i, ep in enumerate(episodes.tolist()):
        ep_to_idx.setdefault(ep, []).append(i)
    ep_list = list(ep_to_idx)
    while len(across) < 80 and len(ep_list) >= 2:
        e1, e2 = rng.choice(len(ep_list), size=2, replace=False)
        i1 = int(rng.choice(ep_to_idx[ep_list[e1]]))
        i2 = int(rng.choice(ep_to_idx[ep_list[e2]]))
        across.append((i1, i2))

    def pair_stats(pairs):
        ratios, corrs, max_over_mean, cls_l2, mean_l2, max_l2 = [], [], [], [], [], []
        example = None
        for n, (i, j) in enumerate(pairs):
            img_d = np.abs(images[i].astype(np.float32) - images[j].astype(np.float32)).mean(axis=2)
            img_t = torch.from_numpy(img_d)[None, None]
            patch = F.adaptive_avg_pool2d(img_t, (side, side))[0, 0].numpy()
            delta = np.linalg.norm(tok[i] - tok[j], axis=-1)
            changed = patch > 8.0
            if changed.any() and (~changed).any():
                ratios.append(float(delta[changed].mean() / max(delta[~changed].mean(), 1e-8)))
            if np.std(patch) > 1e-4 and np.std(delta) > 1e-6:
                corrs.append(spearman(patch.reshape(-1), delta.reshape(-1)))
            max_over_mean.append(float(delta.max() / max(delta.mean(), 1e-8)))
            mean_l2.append(float(delta.mean()))
            max_l2.append(float(delta.max()))
            if example is None and changed.sum() >= 3:
                example = (i, j, patch, delta)
        def summ(v):
            v = np.asarray(v, dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return None
            return {'median': float(np.median(v)), 'mean': float(v.mean()), 'n': int(v.size)}
        return {
            'n_pairs': len(pairs),
            'token_l2_on_changed_over_background': summ(ratios),
            'spatial_spearman_pixel_vs_token': summ(corrs),
            'max_over_mean_token_l2': summ(max_over_mean),
            'mean_token_l2': summ(mean_l2),
            'max_token_l2': summ(max_l2),
            'example': example,
        }, example

    w, wex = pair_stats(within)
    a, aex = pair_stats(across)
    return {'grid': grid_name, 'within_episode': w, 'across_episode': a}, wex


def save_example(path, images, example, title_side):
    if example is None:
        return
    i, j, patch, delta = example
    side = patch.shape[0]

    def heat(grid, base):
        g = grid.astype(np.float64)
        g = (g - g.min()) / max(g.max() - g.min(), 1e-8)
        color = np.zeros((side, side, 3), dtype=np.uint8)
        color[..., 0] = (g * 255).astype(np.uint8)
        color[..., 2] = ((1 - g) * 255).astype(np.uint8)
        im = Image.fromarray(color).resize((224, 224), Image.NEAREST)
        frame = Image.fromarray(base).convert('RGB').resize((224, 224))
        return Image.blend(frame, im, 0.55)

    a = Image.fromarray(images[i])
    b = Image.fromarray(images[j])
    diff = np.abs(images[i].astype(np.int16) - images[j].astype(np.int16)).mean(axis=2)
    diff = (np.clip(diff / max(diff.max(), 1), 0, 1) * 255).astype(np.uint8)
    d = Image.fromarray(diff).convert('RGB')
    canvas = Image.new('RGB', (224 * 5, 224), (0, 0, 0))
    panels = [a, b, d, heat(patch, images[i]), heat(delta, images[i])]
    for c, panel in enumerate(panels):
        canvas.paste(panel.resize((224, 224)), (c * 224, 0))
    canvas.save(path)


def goal_ranking(feat, progress, episodes, steps, test_eps, patches=False):
    rhos_best, rhos_first, rhos_step = [], [], []
    for ep in test_eps:
        idx = np.flatnonzero(episodes == ep)
        if idx.size < 8:
            continue
        prog = progress[idx]
        g = idx[int(np.argmin(prog))]
        first = idx[int(np.argmin(steps[idx]))]
        others = idx[idx != g]

        def dist(ref):
            if patches:
                d = feat[others] - feat[ref]
                return np.linalg.norm(d.reshape(len(others), -1), axis=1)
            return np.linalg.norm(feat[others] - feat[ref], axis=1)

        rhos_best.append(spearman(dist(g), prog[idx != g] if False else progress[others]))
        # progress[others] aligns with dist(g) which used `others`
        rhos_first.append(spearman(dist(first), progress[others]))
        rhos_step.append(spearman(np.abs(steps[others] - steps[g]), progress[others]))
    def summ(v):
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return {
            'median': float(np.median(v)),
            'mean': float(v.mean()),
            'frac_above_0.5': float(np.mean(v > 0.5)),
            'n': int(v.size),
        }
    return {
        'to_best_frame': summ(rhos_best),
        'to_first_frame': summ(rhos_first),
        'step_distance': summ(rhos_step),
    }


def load_lewm():
    cfg_dir = str(ROOT / 'scripts/train/config')
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name='lewm_generalist',
            overrides=[
                'data.envs.pusht.name=' + PATHS['pusht'],
                'data.envs.tworoom.name=' + PATHS['tworoom'],
                'data.envs.cube.name=' + PATHS['cube'],
            ],
        )
    torch.manual_seed(3072)
    world = instantiate(cfg.model).cuda().eval()
    obj = torch.load(LEWM_CKPT, map_location='cpu', weights_only=False)
    sd = {k[len('model.'):]: v for k, v in obj['state_dict'].items() if k.startswith('model.')}
    missing, unexpected = world.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'LeWM load missing={missing[:8]} unexpected={unexpected[:8]}')
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)
    return world


def env_labels(name, table):
    if name == 'pusht':
        return {'state': as_float_array(table['state'])}
    if name == 'tworoom':
        agent = as_float_array(table['pos_agent'])
        target = as_float_array(table['pos_target'])
        obs = as_float_array(table['observation'])
        dist = as_float_array(table['distance_to_target'])
        eucl = np.linalg.norm(agent - target, axis=1)
        return {
            'pos_agent': agent,
            'pos_target': target,
            'observation': obs,
            'distance': dist,
            'eucl_vs_recorded_mae': float(np.abs(eucl - dist[:, 0]).mean()),
            'obs_matches_agent': float(np.abs(obs[:, 0:2] - agent).max()),
            'obs_matches_target': float(np.abs(obs[:, 2:4] - target).max()),
        }
    block = as_float_array(table['privileged_block_0_pos'])
    yaw = as_float_array(table['privileged_block_0_yaw'])
    target = as_float_array(table['privileged_target_block_pos'])
    tyaw = as_float_array(table['privileged_target_block_yaw'])
    eff = as_float_array(table['proprio_effector_pos'])
    grip = as_float_array(table['proprio_gripper_opening'])
    success = as_float_array(table['success'])
    return {
        'block_pos': block,
        'block_yaw': yaw,
        'target_pos': target,
        'target_yaw': tyaw,
        'effector_pos': eff,
        'gripper': grip,
        'success': success,
    }


def main():
    import lance

    OUT.mkdir(parents=True, exist_ok=True)
    print('loading dinov2', flush=True)
    dino = Dinov2Model.from_pretrained(WEIGHTS, local_files_only=True).cuda().eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    print('loading lewm epoch1', flush=True)
    world = load_lewm()
    summary = {
        'setup': {
            'model': 'facebook/dinov2-small',
            'revision': 'ed25f3a31f01632728cabb09d1542f84ab7b0056',
            'architecture': 'ViT-S/14 hidden 384 layers 12 patch 14 no registers',
            'weights_sha256': sha256(WEIGHTS / 'model.safetensors'),
            'preprocess_224': 'native 224 uint8 / 255, ImageNet mean/std, no resize. Official HF processor would upsample shortest edge to 256 then center-crop 224; that resample is skipped because frames are already 224.',
            'preprocess_196': 'bicubic antialias resize 224 to 196, then the same ImageNet norm. 196/14=14 patches.',
            'position_embedding': 'config image_size 518, interpolated to the input grid inside Dinov2Embeddings',
            'frozen': 'eval + no_grad + requires_grad False',
            'lewm': str(LEWM_CKPT),
            'lewm_note': 'epoch-1 generalist, CLS and projector, eval mode, 224 ImageNet norm',
            'frames_per_env': N_EP * FRAMES,
            'split': '100 episodes, length>=48, 32 frames each, every 4th episode held out',
            'probe': 'column-standardized dual ridge, alpha chosen on an inner episode split, refit on all train episodes',
            'spatial_readout': 'linear ridge on 4x4 and 8x8 average-pooled patch tokens',
            'baselines': 'train-mean constant, fixed random projection of area-pooled RGB grid',
        },
        'envs': {},
    }
    for name, path in PATHS.items():
        print('\n==', name, flush=True)
        dataset = lance.dataset(path)
        starts, ends = episode_bounds(dataset)
        episodes = choose_episodes(starts, ends)
        rows, ep_ids, steps = sample_rows(starts, ends, episodes)
        columns = {
            'pusht': ['pixels', 'state', 'step_idx'],
            'tworoom': ['pixels', 'pos_agent', 'pos_target', 'observation', 'distance_to_target'],
            'cube': [
                'pixels', 'privileged_block_0_pos', 'privileged_block_0_yaw',
                'privileged_target_block_pos', 'privileged_target_block_yaw',
                'proprio_effector_pos', 'proprio_gripper_opening', 'success',
            ],
        }[name]
        table = take_columns(dataset, rows, columns)
        images = np.stack([decode_image(b) for b in table['pixels']])
        if images.shape[1:] != (224, 224, 3):
            raise RuntimeError(f'{name} image shape {images.shape}')
        labels = env_labels(name, table)
        print(name, 'frames', len(images), 'episodes', len(np.unique(ep_ids)), flush=True)
        cls224, tok224 = encode_dino(dino, images, 224)
        cls196, tok196 = encode_dino(dino, images, 196)
        lewm_cls, lewm_proj = encode_lewm(world, images)
        print(
            name,
            'tokens', tuple(tok224.shape), tuple(tok196.shape),
            'lewm', tuple(lewm_cls.shape),
            flush=True,
        )
        rand224 = random_features(images, 16, 384)
        rand196 = random_features(images, 14, 384)
        test_eps = episodes[::4]
        train_eps = np.setdiff1d(episodes, test_eps)
        inner_eps = train_eps[::5]
        specs = target_specs(name, labels)
        groups = {
            'dino224': feature_table(tok224, cls224, {
                'random_proj': rand224,
                'constant': torch.ones(len(images), 1),
                'lewm_cls': lewm_cls,
                'lewm_projector': lewm_proj,
            }),
            'dino196': feature_table(tok196, cls196, {
                'random_proj': rand196,
                'constant': torch.ones(len(images), 1),
            }),
        }
        probes = {}
        for res, feats in groups.items():
            probes[res] = {}
            for feat_name, feat in feats.items():
                probes[res][feat_name] = {}
                for target_name, target in specs.items():
                    fit = select_and_fit(feat, target, train_eps, inner_eps, ep_ids)
                    names = ['sin', 'cos'] if 'sincos' in target_name else []
                    probes[res][feat_name][target_name] = pack_fit(fit, names)
                    print(
                        f'  {res} {feat_name} {target_name} testR2 {probes[res][feat_name][target_name]["test_r2_mean"]:.3f}',
                        flush=True,
                    )
        sens224, ex224 = sensitivity(images, tok224, ep_ids, steps, 16)
        sens196, _ = sensitivity(images, tok196, ep_ids, steps, 14)
        # LeWM has no patch tokens; compare CLS distance on the same within pairs later via probe.
        save_example(OUT / f'sensitivity_{name}_224.png', images, ex224, 16)
        goals = {}
        if name == 'tworoom':
            progress = labels['distance'][:, 0]
            goals['recorded_distance'] = {
                'cls_224': goal_ranking(cls224.numpy(), progress, ep_ids, steps, test_eps),
                'patch_mean_224': goal_ranking(tok224.numpy(), progress, ep_ids, steps, test_eps, patches=True),
                'cls_196': goal_ranking(cls196.numpy(), progress, ep_ids, steps, test_eps),
                'patch_mean_196': goal_ranking(tok196.numpy(), progress, ep_ids, steps, test_eps, patches=True),
                'lewm_cls': goal_ranking(lewm_cls.numpy(), progress, ep_ids, steps, test_eps),
            }
            goals['label_checks'] = {
                'eucl_vs_recorded_mae': labels['eucl_vs_recorded_mae'],
                'obs_matches_agent_maxabs': labels['obs_matches_agent'],
                'obs_matches_target_maxabs': labels['obs_matches_target'],
                'note': 'distance_to_target matches Euclidean agent-target distance. It is not a wall-aware path length. Door position is observation[4:6].',
            }
        elif name == 'cube':
            progress = np.linalg.norm(labels['block_pos'] - labels['target_pos'], axis=1)
            goals['block_to_target_pos'] = {
                'cls_224': goal_ranking(cls224.numpy(), progress, ep_ids, steps, test_eps),
                'patch_mean_224': goal_ranking(tok224.numpy(), progress, ep_ids, steps, test_eps, patches=True),
                'patch_mean_196': goal_ranking(tok196.numpy(), progress, ep_ids, steps, test_eps, patches=True),
                'lewm_cls': goal_ranking(lewm_cls.numpy(), progress, ep_ids, steps, test_eps),
            }
            goals['success_rate_frames'] = float(labels['success'][:, 0].mean())
            goals['episodes_with_success'] = int(sum(
                labels['success'][ep_ids == ep, 0].max() > 0.5 for ep in np.unique(ep_ids)
            ))
        else:
            # No goal pose column. Report terminal block dispersion, not a fabricated goal.
            last_rows = (ends[episodes] - 1).astype(np.int64)
            last = dataset.take(last_rows.tolist(), columns=['state'])
            last_state = as_float_array(last['state'].to_pylist())
            goals['unavailable'] = {
                'reason': 'train.lance has state (agent xy, block xy, angle, agent vel) and no goal pose.',
                'terminal_block_xy_std': last_state[:, 2:4].std(axis=0).tolist(),
                'terminal_block_angle_std': float(last_state[:, 4].std()),
                'n_episode_ends': int(len(last_state)),
            }
        label_scale = {}
        for key, value in specs.items():
            arr = value.numpy()
            label_scale[key] = {
                'std': arr.std(axis=0).tolist(),
                'mean': arr.mean(axis=0).tolist(),
            }
        summary['envs'][name] = {
            'n_frames': int(len(images)),
            'n_episodes': int(len(np.unique(ep_ids))),
            'n_test_episodes': int(len(test_eps)),
            'n_train_episodes': int(len(train_eps)),
            'label_scale': label_scale,
            'probes': probes,
            'sensitivity': {'dino224': sens224, 'dino196': sens196},
            'goal': goals,
        }
        # drop bulky example arrays before json
        for block in summary['envs'][name]['sensitivity'].values():
            for kind in ('within_episode', 'across_episode'):
                block[kind].pop('example', None)
        with open(OUT / 'metrics.json', 'w') as f:
            json.dump(summary, f)
        del images, tok224, tok196, cls224, cls196
        torch.cuda.empty_cache()
        print('saved', name, flush=True)
    with open(OUT / 'metrics.json', 'w') as f:
        json.dump(summary, f)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
