"""G1: can one frozen-DINO spatial dynamics fit three environments?

One-step overfit only. At most 500 optimizer updates on 96 train episodes
per environment, with 32 episode-disjoint held-out episodes. Uniform patch
MSE is the training loss, not a planning objective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.pop('HF_ENDPOINT', None)

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from stable_worldmodel.wm.spatial.data import (  # noqa: E402
    RAW_ACTION_DIM,
    apply_action_normalizer,
    candidate_starts,
    fit_action_normalizer,
    pack_blocked_action,
    select_episode_ids,
    split_episodes,
)
from stable_worldmodel.wm.spatial.metrics import (  # noqa: E402
    moving_patch_errors,
    prediction_mse,
)
from stable_worldmodel.wm.spatial.model import (  # noqa: E402
    ENV_NAMES,
    FrozenDINOv2,
    SpatialDynamics,
)

OUT = ROOT / 'results/stage3/g1_spatial/overfit500'
WEIGHTS = ROOT / 'results/stage3/g0_dinov2/weights'
WEIGHT_SHA = 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'
PATHS = {
    'pusht': '/workspace/datasets/lewm-pusht/train.lance',
    'tworoom': '/workspace/datasets/lewm-tworooms/train.lance',
    'cube': '/workspace/datasets/lewm-cube/train.lance',
}
LABELS = {
    'pusht': ['state'],
    'tworoom': ['pos_agent'],
    'cube': ['privileged_block_0_pos', 'proprio_effector_pos'],
}
MOTION_KEY = {
    'pusht': 'state',
    'tworoom': 'pos_agent',
    'cube': 'privileged_block_0_pos',
}
ENVS = ENV_NAMES
SEED = 3072
N_TRAIN = 96
N_TEST = 32
BATCH = 16
STEPS = 500
LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
FRAMESKIP = 5
NUM_STEPS = 2
SPAN = NUM_STEPS * FRAMESKIP
EVAL_AT = (0, 100, 250, 500)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def log(path: Path, message: str):
    line = message.rstrip()
    print(line, flush=True)
    with open(path, 'a', buffering=1) as handle:
        handle.write(line + '\n')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def param_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for param in module.parameters():
        array = np.ascontiguousarray(param.detach().float().cpu().numpy())
        digest.update(array.tobytes())
    return digest.hexdigest()


def episode_bounds(dataset):
    episode = dataset.to_table(columns=['episode_idx']).column(0).to_numpy()
    change = np.flatnonzero(np.diff(episode)) + 1
    starts = np.concatenate([[0], change]).astype(np.int64)
    ends = np.concatenate([change, [len(episode)]]).astype(np.int64)
    return starts, ends


def take_columns(dataset, rows, columns, batch=512):
    rows = [int(r) for r in rows]
    merged = {name: [None] * len(rows) for name in columns}
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        table = dataset.take(chunk, columns=columns)
        for name in columns:
            values = table[name].to_pylist()
            merged[name][start : start + len(values)] = values
    return merged


def decode_image(blob) -> np.ndarray:
    raw = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        array = np.asarray(img.convert('RGB'))
    if array.shape != (224, 224, 3):
        raise RuntimeError(f'expected 224x224 RGB, got {array.shape}')
    return array


def vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def motion_vector(env: str, value: np.ndarray) -> np.ndarray:
    if env == 'pusht':
        return value[:4]
    return value


def label_matrix(env: str, labels: dict[str, np.ndarray]) -> np.ndarray:
    if env == 'pusht':
        return labels['state'][:, :5]
    if env == 'tworoom':
        return labels['pos_agent']
    return np.concatenate(
        [
            labels['privileged_block_0_pos'],
            labels['proprio_effector_pos'],
        ],
        axis=1,
    )


GROUPS = {
    'pusht': (
        ('agent_xy', slice(0, 2), 'l2'),
        ('block_xy', slice(2, 4), 'l2'),
        ('block_angle', slice(4, 5), 'angle'),
    ),
    'tworoom': (('agent_xy', slice(0, 2), 'l2'),),
    'cube': (
        ('block_xyz', slice(0, 3), 'l2'),
        ('effector_xyz', slice(3, 6), 'l2'),
    ),
}


def choose_clips(dataset, starts, ends, episodes, env: str):
    """One clip per episode, at the candidate with the largest state move."""
    key = MOTION_KEY[env]
    plans = []
    wanted = []
    for ep in episodes.tolist():
        length = int(ends[ep] - starts[ep])
        options = candidate_starts(length, SPAN, k=16)
        base = int(starts[ep])
        plans.append((int(ep), base, options))
        for step in options.tolist():
            wanted.append(base + int(step))
            wanted.append(base + int(step) + FRAMESKIP)
    unique = sorted(set(wanted))
    fetched = take_columns(dataset, unique, [key])
    values = {
        row: vector(item) for row, item in zip(unique, fetched[key])
    }
    chosen = []
    for ep, base, options in plans:
        best_step = int(options[0])
        best_motion = -1.0
        for step in options.tolist():
            ctx = values[base + int(step)]
            tgt = values[base + int(step) + FRAMESKIP]
            delta = float(
                np.linalg.norm(motion_vector(env, tgt) - motion_vector(env, ctx))
            )
            if delta > best_motion:
                best_motion = delta
                best_step = int(step)
        chosen.append(
            {
                'episode': ep,
                'start': best_step,
                'motion': best_motion,
                'ctx_row': base + best_step,
                'tgt_row': base + best_step + FRAMESKIP,
            }
        )
    return chosen


def load_split(dataset, clips, env: str):
    rows = []
    for clip in clips:
        rows.append(clip['ctx_row'])
        rows.append(clip['tgt_row'])
        rows.extend(range(clip['ctx_row'], clip['tgt_row']))
    columns = ['pixels', 'action', *LABELS[env]]
    unique = sorted(set(rows))
    fetched = take_columns(dataset, unique, columns)
    by_row = {
        row: {name: fetched[name][i] for name in columns}
        for i, row in enumerate(unique)
    }
    raw_dim = RAW_ACTION_DIM[env]
    images_ctx = []
    images_tgt = []
    actions = []
    labels_ctx = {name: [] for name in LABELS[env]}
    labels_tgt = {name: [] for name in LABELS[env]}
    for clip in clips:
        ctx_row = clip['ctx_row']
        tgt_row = clip['tgt_row']
        action_rows = list(range(ctx_row, tgt_row))
        if tgt_row in action_rows or action_rows[-1] != tgt_row - 1:
            raise RuntimeError('action block overlaps the target frame')
        raw = np.stack([vector(by_row[row]['action']) for row in action_rows])
        if raw.shape != (FRAMESKIP, raw_dim):
            raise RuntimeError(f'{env} action shape {raw.shape}, expected {(FRAMESKIP, raw_dim)}')
        blocked = pack_blocked_action(raw, num_steps=1, frameskip=FRAMESKIP)[0]
        images_ctx.append(decode_image(by_row[ctx_row]['pixels']))
        images_tgt.append(decode_image(by_row[tgt_row]['pixels']))
        actions.append(blocked)
        for name in LABELS[env]:
            labels_ctx[name].append(vector(by_row[ctx_row][name]))
            labels_tgt[name].append(vector(by_row[tgt_row][name]))
    return {
        'images_ctx': np.stack(images_ctx),
        'images_tgt': np.stack(images_tgt),
        'action': np.stack(actions).astype(np.float32),
        'labels_ctx': {k: np.stack(v) for k, v in labels_ctx.items()},
        'labels_tgt': {k: np.stack(v) for k, v in labels_tgt.items()},
        'episode': np.array([c['episode'] for c in clips], dtype=np.int64),
        'start': np.array([c['start'] for c in clips], dtype=np.int64),
        'motion': np.array([c['motion'] for c in clips], dtype=np.float32),
    }


def preprocess(images_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images_u8))
    x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div(255)
    mean = torch.tensor(MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def encode_images(encoder, images_u8, device, batch=32) -> torch.Tensor:
    parts = []
    for start in range(0, len(images_u8), batch):
        pixels = preprocess(images_u8[start : start + batch], device)
        tokens = encoder.encode(pixels)
        if tokens.shape[1:] != (1, 256, 384):
            raise RuntimeError(f'unexpected DINO tokens {tuple(tokens.shape)}')
        parts.append(tokens[:, 0].float().cpu())
    return torch.cat(parts, dim=0)


def pool4(tokens: torch.Tensor) -> np.ndarray:
    n, patches, dim = tokens.shape
    side = int(round(patches**0.5))
    grid = tokens.reshape(n, side, side, dim).permute(0, 3, 1, 2).float()
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (4, 4))
    return pooled.permute(0, 2, 3, 1).reshape(n, -1).numpy()


def fit_probe(x, y, seed=SEED):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    n_val = max(8, len(x) // 4)
    val, train = perm[:n_val], perm[n_val:]

    def solve(rows, alpha, x_mean, x_std, y_mean, y_std):
        xn = (x[rows] - x_mean) / x_std
        yn = (y[rows] - y_mean) / y_std
        gram = xn @ xn.T
        coef = np.linalg.solve(gram + alpha * np.eye(len(rows)), yn)
        weight = xn.T @ coef
        return weight

    def stats(rows):
        x_mean = x[rows].mean(0)
        x_std = np.maximum(x[rows].std(0), 1e-6)
        y_mean = y[rows].mean(0)
        y_std = np.maximum(y[rows].std(0), 1e-6)
        return x_mean, x_std, y_mean, y_std

    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
    x_mean, x_std, y_mean, y_std = stats(train)
    best_alpha, best_err = None, None
    for alpha in alphas:
        weight = solve(train, alpha, x_mean, x_std, y_mean, y_std)
        pred = ((x[val] - x_mean) / x_std) @ weight * y_std + y_mean
        err = float(np.mean((pred - y[val]) ** 2))
        if best_err is None or err < best_err:
            best_alpha, best_err = alpha, err
    x_mean, x_std, y_mean, y_std = stats(np.arange(len(x)))
    weight = solve(np.arange(len(x)), best_alpha, x_mean, x_std, y_mean, y_std)
    return {
        'alpha': float(best_alpha),
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'weight': weight,
    }


def apply_probe(probe, features) -> np.ndarray:
    xn = (np.asarray(features, dtype=np.float64) - probe['x_mean']) / probe['x_std']
    return xn @ probe['weight'] * probe['y_std'] + probe['y_mean']


def group_error(pred, true, sl, kind: str) -> float:
    if kind == 'l2':
        return float(np.linalg.norm(pred[:, sl] - true[:, sl], axis=1).mean())
    delta = np.arctan2(
        np.sin(pred[:, sl] - true[:, sl]), np.cos(pred[:, sl] - true[:, sl])
    )
    return float(np.degrees(np.abs(delta)).mean())


def grad_norm(params) -> float:
    total = None
    for param in params:
        if param.grad is None:
            continue
        value = param.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(torch.sqrt(total))


def gpu_status():
    try:
        text = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used',
                '--format=csv,noheader,nounits',
            ],
            text=True,
        ).strip().splitlines()[0]
        util, mem = [part.strip() for part in text.split(',')]
        return int(util), float(mem) / 1024
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return -1, float('nan')


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


@torch.no_grad()
def predict_tokens(model, ctx, actions, env_index, use_action, batch=32):
    device = next(model.parameters()).device
    parts = []
    env = torch.full((ctx.shape[0],), env_index, dtype=torch.long, device=device)
    for start in range(0, ctx.shape[0], batch):
        pred = model(
            ctx[start : start + batch].to(device)[:, None],
            actions[start : start + batch].to(device)[:, None],
            env[start : start + batch],
            use_action=use_action,
        )
        parts.append(pred[:, 0].float().cpu())
    return torch.cat(parts, dim=0)


def evaluate_split(model, split, env, env_index, probe, shuffle_perm):
    ctx = split['ctx']
    tgt = split['tgt']
    act = split['act']
    correct = predict_tokens(model, ctx, act, env_index, True)
    shuffled = predict_tokens(model, ctx, act[shuffle_perm], env_index, True)
    no_action = predict_tokens(model, ctx, act, env_index, False)
    y_tgt = label_matrix(env, split['labels_tgt'])
    y_ctx = label_matrix(env, split['labels_ctx'])
    hat = {
        'correct': apply_probe(probe, pool4(correct)),
        'persistence': apply_probe(probe, pool4(ctx)),
        'ceiling': apply_probe(probe, pool4(tgt)),
        'shuffle': apply_probe(probe, pool4(shuffled)),
        'no_action': apply_probe(probe, pool4(no_action)),
    }
    state = {}
    for name, sl, kind in GROUPS[env]:
        state[name] = {
            'correct': group_error(hat['correct'], y_tgt, sl, kind),
            'persistence': group_error(hat['persistence'], y_tgt, sl, kind),
            'ceiling': group_error(hat['ceiling'], y_tgt, sl, kind),
            'shuffle': group_error(hat['shuffle'], y_tgt, sl, kind),
            'no_action': group_error(hat['no_action'], y_tgt, sl, kind),
            'same_frame_readout': group_error(hat['persistence'], y_ctx, sl, kind),
            'true_motion': group_error(y_ctx, y_tgt, sl, kind),
        }
    return {
        'mse_correct': float((correct - tgt).pow(2).mean()),
        'mse_persistence': float((ctx - tgt).pow(2).mean()),
        'mse_shuffle': float((shuffled - tgt).pow(2).mean()),
        'mse_no_action': float((no_action - tgt).pow(2).mean()),
        'moving_correct': moving_patch_errors(correct, tgt, ctx),
        'moving_shuffle': moving_patch_errors(shuffled, tgt, ctx),
        'moving_no_action': moving_patch_errors(no_action, tgt, ctx),
        'state': state,
        'mean_state_motion': float(split['motion'].mean()),
    }


def save_checkpoint(path, model, opt, step, normalizer):
    torch.save(
        {
            'step': step,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'normalizer': {
                env: {
                    'mean': torch.from_numpy(stats['mean']),
                    'std': torch.from_numpy(stats['std']),
                }
                for env, stats in normalizer.items()
            },
        },
        path,
    )


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None):
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args(argv)
    OUT = args.output_dir.resolve()
    if OUT.exists():
        raise FileExistsError(f'refusing to overwrite G1 output directory {OUT}')
    if (OUT / 'metrics.csv').exists():
        raise SystemExit(f'{OUT} already has a run. Refusing to overwrite it.')
    OUT.mkdir(parents=True, exist_ok=False)
    log_path = OUT / 'train.log'
    (OUT / 'STATUS').write_text('RUNNING\n')
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import lance

    weight_file = WEIGHTS / 'model.safetensors'
    file_sha = sha256_file(weight_file)
    if file_sha != WEIGHT_SHA:
        raise RuntimeError(f'DINOv2 weight sha mismatch: {file_sha}')

    config = {
        'seed': SEED,
        'steps': STEPS,
        'n_train': N_TRAIN,
        'n_test': N_TEST,
        'batch_per_env': BATCH,
        'lr': LR,
        'weight_decay': WD,
        'grad_clip': GRAD_CLIP,
        'frameskip': FRAMESKIP,
        'history': 1,
        'grid': 16,
        'token_dim': 384,
        'hidden': 256,
        'depth': 6,
        'heads': 8,
        'mlp_ratio': 4,
        'dropout': 0.0,
        'image_size': 224,
        'action_raw_dim': RAW_ACTION_DIM,
        'action_blocked_dim': {k: v * FRAMESKIP for k, v in RAW_ACTION_DIM.items()},
        'action_norm': 'z-score fit on the 96 train clips of each env only',
        'objective': 'one-step uniform spatial MSE. Not a planning cost.',
        'design': (
            'Factorized spatial attention plus causal temporal attention, '
            'instead of one flat sequence. Depth 6 and hidden 256 are unchanged. '
            'History is 1 for this one-step test; history 2 is covered by the '
            'causal-mask unit tests. Actions use a nonzero MLP, not AdaLN-zero. '
            'Clips are spread across each dataset and, within an episode, taken '
            'at the candidate step with the largest control-relevant state move.'
        ),
        'weights_sha256': file_sha,
    }
    (OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    log(log_path, f'G1 start. output={OUT}')

    data = {}
    normalizer = {}
    for env in ENVS:
        t0 = time.perf_counter()
        log(log_path, f'indexing {env}')
        dataset = lance.dataset(PATHS[env])
        starts, ends = episode_bounds(dataset)
        lengths = ends - starts
        chosen = select_episode_ids(
            lengths, N_TRAIN + N_TEST, min_length=SPAN, seed=SEED
        )
        train_eps, test_eps = split_episodes(chosen, N_TRAIN, N_TEST, SEED)
        if set(train_eps.tolist()) & set(test_eps.tolist()):
            raise RuntimeError(f'{env} train/test episodes overlap')
        train_clips = choose_clips(dataset, starts, ends, train_eps, env)
        test_clips = choose_clips(dataset, starts, ends, test_eps, env)
        train = load_split(dataset, train_clips, env)
        test = load_split(dataset, test_clips, env)
        mean, std = fit_action_normalizer(train['action'])
        if mean.shape != (RAW_ACTION_DIM[env] * FRAMESKIP,):
            raise RuntimeError(f'{env} normalizer dim {mean.shape}')
        train['action'] = apply_action_normalizer(train['action'], mean, std)
        test['action'] = apply_action_normalizer(test['action'], mean, std)
        normalizer[env] = {'mean': mean, 'std': std}
        data[env] = {'train': train, 'test': test}
        log(
            log_path,
            f'loaded {env}: train {len(train_eps)} test {len(test_eps)} '
            f'motion {train["motion"].mean():.4f}/{test["motion"].mean():.4f} '
            f'in {time.perf_counter() - t0:.1f}s',
        )
        del dataset

    (OUT / 'episodes.json').write_text(
        json.dumps(
            {
                env: {
                    split: {
                        'episode': data[env][split]['episode'].tolist(),
                        'start': data[env][split]['start'].tolist(),
                        'motion': data[env][split]['motion'].tolist(),
                    }
                    for split in ('train', 'test')
                }
                for env in ENVS
            }
        )
        + '\n'
    )
    (OUT / 'action_norm.json').write_text(
        json.dumps(
            {
                env: {
                    'mean': normalizer[env]['mean'].tolist(),
                    'std': normalizer[env]['std'].tolist(),
                }
                for env in ENVS
            }
        )
        + '\n'
    )

    log(log_path, 'loading frozen DINOv2-S/14')
    torch.cuda.reset_peak_memory_stats()
    encoder = FrozenDINOv2.from_pretrained(str(WEIGHTS)).to(device)
    if any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError('DINOv2 has a trainable parameter')
    encoder_hash = param_hash(encoder)
    for env in ENVS:
        for split in ('train', 'test'):
            part = data[env][split]
            part['ctx'] = encode_images(encoder, part['images_ctx'], device)
            part['tgt'] = encode_images(encoder, part['images_tgt'], device)
            part['act'] = torch.from_numpy(np.ascontiguousarray(part['action']))
            if split == 'train':
                part['timed_ctx'] = np.ascontiguousarray(part['images_ctx'][:BATCH])
                part['timed_tgt'] = np.ascontiguousarray(part['images_tgt'][:BATCH])
            del part['images_ctx']
            del part['images_tgt']
    encode_peak = torch.cuda.max_memory_allocated() / (1024**3)
    if param_hash(encoder) != encoder_hash:
        raise RuntimeError('encoder weights changed during the cache encode')
    log(log_path, f'cached tokens. encode peak {encode_peak:.2f} GB')

    set_seed(SEED)
    model = SpatialDynamics(
        token_dim=384,
        hidden=256,
        depth=6,
        heads=8,
        mlp_ratio=4.0,
        grid=16,
        dropout=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    example = {}
    model.train()
    for env in ENVS:
        ctx = data[env]['train']['ctx'][:2].to(device)
        act = data[env]['train']['act'][:2].to(device)
        eid = torch.full((2,), ENVS.index(env), dtype=torch.long, device=device)
        pred = model(ctx[:, None], act[:, None], eid)
        example[env] = {
            'context': list(ctx[:, None].shape),
            'action': list(act[:, None].shape),
            'env_id': list(eid.shape),
            'prediction': list(pred.shape),
            'target': list(data[env]['train']['tgt'][:2, None].shape),
        }
        if pred.shape != (2, 1, 256, 384):
            raise RuntimeError(f'{env} prediction shape {tuple(pred.shape)}')
    config['parameters'] = int(n_params)
    config['shapes'] = example
    config['encode_peak_gb'] = encode_peak
    (OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    log(log_path, f'dynamics parameters {n_params}')

    probes = {}
    for env in ENVS:
        train = data[env]['train']
        features = np.concatenate(
            [pool4(train['ctx']), pool4(train['tgt'])], axis=0
        )
        labels = np.concatenate(
            [
                label_matrix(env, train['labels_ctx']),
                label_matrix(env, train['labels_tgt']),
            ],
            axis=0,
        )
        probes[env] = fit_probe(features, labels)
        log(log_path, f'{env} readout alpha {probes[env]["alpha"]}')

    rng = np.random.default_rng(SEED)
    batches = []
    for _ in range(STEPS):
        batches.append(
            {env: rng.choice(N_TRAIN, size=BATCH, replace=False).tolist() for env in ENVS}
        )
    (OUT / 'batches.json').write_text(json.dumps(batches) + '\n')
    shuffle_perm = {
        env: {
            split: np.random.default_rng(SEED + 17).permutation(
                len(data[env][split]['ctx'])
            )
            for split in ('train', 'test')
        }
        for env in ENVS
    }

    def run_eval(step_done: int):
        model.eval()
        report = {'step': step_done}
        for env in ENVS:
            report[env] = {}
            for split in ('train', 'test'):
                report[env][split] = evaluate_split(
                    model,
                    data[env][split],
                    env,
                    ENVS.index(env),
                    probes[env],
                    shuffle_perm[env][split],
                )
        model.train()
        return report

    evals = []
    full_stats = {}
    csv_path = OUT / 'metrics.csv'
    fields = [
        'step',
        'total_loss',
        'pusht_loss',
        'tworoom_loss',
        'cube_loss',
        'pusht_grad',
        'tworoom_grad',
        'cube_grad',
        'joint_grad',
        'lr',
        'step_time_s',
        'peak_vram_gb',
        'gpu_util',
        'gpu_mem_gb',
    ]
    with open(csv_path, 'w', newline='', buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        evals.append(run_eval(0))
        (OUT / 'eval.json').write_text(json.dumps(jsonable(evals), indent=2) + '\n')
        log(log_path, 'eval at step 0 written')

        # One uncached step. It is not an optimizer update.
        torch.cuda.reset_peak_memory_stats()
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        full_losses = []
        for env in ENVS:
            pix_ctx = preprocess(data[env]['train']['timed_ctx'], device)
            pix_tgt = preprocess(data[env]['train']['timed_tgt'], device)
            with torch.no_grad():
                ctx = encoder.encode(pix_ctx)
                tgt = encoder.encode(pix_tgt)
            act = data[env]['train']['act'][:BATCH].to(device)
            eid = torch.full((BATCH,), ENVS.index(env), device=device, dtype=torch.long)
            full_losses.append(prediction_mse(model(ctx, act[:, None], eid), tgt))
        torch.stack(full_losses).mean().backward()
        torch.cuda.synchronize()
        full_stats['seconds'] = time.perf_counter() - t0
        full_stats['peak_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        opt.zero_grad(set_to_none=True)
        for env in ENVS:
            del data[env]['train']['timed_ctx']
            del data[env]['train']['timed_tgt']
        log(
            log_path,
            f'full encode+backward {full_stats["seconds"]:.3f}s, '
            f'peak {full_stats["peak_gb"]:.2f} GB (no optimizer step)',
        )

        from tqdm import trange

        torch.cuda.reset_peak_memory_stats()
        bar = trange(STEPS, desc='g1', dynamic_ncols=True, mininterval=0.3, file=sys.stderr)
        for step_idx, batch in zip(bar, batches):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            losses = {}
            for env, index in batch.items():
                take = torch.as_tensor(index, dtype=torch.long)
                ctx = data[env]['train']['ctx'].index_select(0, take).to(device)
                tgt = data[env]['train']['tgt'].index_select(0, take).to(device)
                act = data[env]['train']['act'].index_select(0, take).to(device)
                eid = torch.full((BATCH,), ENVS.index(env), device=device, dtype=torch.long)
                losses[env] = prediction_mse(
                    model(ctx[:, None], act[:, None], eid), tgt[:, None]
                )
            torch.cuda.synchronize()
            forward_s = time.perf_counter() - t0
            env_grads = {}
            for env, loss in losses.items():
                opt.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)
                env_grads[env] = grad_norm(params)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            total = torch.stack([losses[env] for env in ENVS]).mean()
            total.backward()
            joint = grad_norm(params)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = forward_s + (time.perf_counter() - t1)
            done = step_idx + 1
            values = {env: float(losses[env].detach()) for env in ENVS}
            if not all(np.isfinite(v) for v in values.values()):
                raise RuntimeError(f'non-finite loss at step {done}: {values}')
            mean_loss = float(np.mean(list(values.values())))
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            util, mem = gpu_status() if done % 10 == 0 or done == STEPS else (-1, float('nan'))
            lr = opt.param_groups[0]['lr']
            row = {
                'step': done,
                'total_loss': mean_loss,
                'pusht_loss': values['pusht'],
                'tworoom_loss': values['tworoom'],
                'cube_loss': values['cube'],
                'pusht_grad': env_grads['pusht'],
                'tworoom_grad': env_grads['tworoom'],
                'cube_grad': env_grads['cube'],
                'joint_grad': joint,
                'lr': lr,
                'step_time_s': dt,
                'peak_vram_gb': peak,
                'gpu_util': util,
                'gpu_mem_gb': mem,
            }
            writer.writerow(row)
            handle.flush()
            bar.set_postfix(
                loss=f'{mean_loss:.4f}',
                pusht=f'{values["pusht"]:.4f}',
                tworoom=f'{values["tworoom"]:.4f}',
                cube=f'{values["cube"]:.4f}',
                lr=f'{lr:.2e}',
                vram=f'{peak:.2f}GB',
                gpu=f'{util}%' if util >= 0 else '-',
            )
            if done in EVAL_AT:
                evals.append(run_eval(done))
                (OUT / 'eval.json').write_text(json.dumps(jsonable(evals), indent=2) + '\n')
                save_checkpoint(OUT / f'checkpoint_step{done}.pt', model, opt, done, normalizer)
                log(log_path, f'eval and checkpoint at step {done}')
        bar.close()

    train_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    if param_hash(encoder) != encoder_hash:
        raise RuntimeError('encoder weights changed during training')
    trainable = sum(p.requires_grad for p in encoder.parameters())
    if trainable:
        raise RuntimeError('encoder became trainable')

    # Resume check: reload the step-500 checkpoint and compare one batch.
    fresh = SpatialDynamics().to(device)
    blob = torch.load(OUT / 'checkpoint_step500.pt', map_location=device, weights_only=True)
    fresh.load_state_dict(blob['model'])
    fresh.eval()
    model.eval()
    with torch.no_grad():
        take = torch.as_tensor(batches[-1]['pusht'], dtype=torch.long)
        ctx = data['pusht']['train']['ctx'].index_select(0, take).to(device)
        act = data['pusht']['train']['act'].index_select(0, take).to(device)
        eid = torch.zeros(BATCH, dtype=torch.long, device=device)
        a = model(ctx[:, None], act[:, None], eid)
        b = fresh(ctx[:, None], act[:, None], eid)
    if not torch.allclose(a, b, atol=1e-5):
        raise RuntimeError('reloaded checkpoint does not reproduce predictions')

    times = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=10)
    summary = {
        'steps_completed': STEPS,
        'parameters': n_params,
        'shapes': example,
        'encoder_frozen': True,
        'encoder_hash_unchanged': True,
        'checkpoint_reload_ok': True,
        'encode_peak_gb': encode_peak,
        'full_step_s': full_stats.get('seconds'),
        'full_step_peak_gb': full_stats.get('peak_gb'),
        'train_peak_gb': float(train_peak_gb),
        'step_time_note': (
            'step_time_s is the cached-token forward, joint backward and '
            'optimizer step. Per-env gradient norms are an extra backward '
            'and are not included. full_step_s also encodes both frames.'
        ),
        'step_time_s': {
            'median': float(np.median(times)),
            'mean': float(np.mean(times)),
            'first': float(times[0]),
        },
        'evals': jsonable(evals),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(open(csv_path)))
        steps = [int(r['step']) for r in rows]
        plt.figure(figsize=(7, 4))
        for env in ENVS:
            plt.plot(steps, [float(r[f'{env}_loss']) for r in rows], label=env)
        plt.xlabel('optimizer update')
        plt.ylabel('batch spatial MSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / 'loss_curve.png', dpi=120)
        plt.close()
    except Exception as exc:  # plotting is optional
        log(log_path, f'plot skipped: {exc}')
    (OUT / 'STATUS').write_text('DONE\n')
    log(log_path, f'G1_DONE exit=0 steps={STEPS} median_step={float(np.median(times)):.3f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            status = OUT / 'STATUS'
            if status.exists() and status.read_text().strip() == 'RUNNING':
                status.write_text('FAILED\n')
                with open(OUT / 'train.log', 'a', buffering=1) as handle:
                    handle.write('G1_DONE exit=1\n')
        except OSError:
            pass
        sys.exit(1)
