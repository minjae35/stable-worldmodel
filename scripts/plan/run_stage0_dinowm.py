"""DINO-WM PushT eval on the LeWM Stage 0 pairs, one shard per GPU.

The parent process only draws the tqdm bar. Each worker is a separate
process with its own ``CUDA_VISIBLE_DEVICES`` and its own log file.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = '/workspace/datasets/lewm-pusht'
LEWM_DIR = ROOT / 'results' / 'stage0' / 'eval100'
BUNDLED_PAIRS = ROOT / 'scripts' / 'plan' / 'config' / 'stage0_eval100_pairs.json'


def _pairs_path() -> Path:
    """Prefer the local eval copy, then the committed pair list."""
    local = LEWM_DIR / 'pairs.json'
    return local if local.exists() else BUNDLED_PAIRS
OUT_DIR = ROOT / 'results' / 'stage0' / 'dinowm_eval100'
SHARDS = [(0, 0, 25), (1, 25, 50), (2, 50, 75), (3, 75, 100)]
N_EVAL = 100


def _read_rows(path: Path) -> dict[int, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[int(row['pair_index'])] = row
    return rows


def _monitor() -> None:
    from tqdm import tqdm

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = []
    procs = []
    for gpu, start, end in SHARDS:
        shard = OUT_DIR / f'gpu{gpu}'
        shard.mkdir(parents=True, exist_ok=True)
        log_path = shard / 'eval.log'
        handle = log_path.open('a')
        logs.append(handle)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
        env['PYTHONUNBUFFERED'] = '1'
        env['MUJOCO_GL'] = 'egl'
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    '-u',
                    str(Path(__file__).resolve()),
                    '--worker',
                    '--gpu',
                    str(gpu),
                    '--start',
                    str(start),
                    '--end',
                    str(end),
                ],
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        )

    done = {}
    for gpu, _, _ in SHARDS:
        done.update(_read_rows(OUT_DIR / f'gpu{gpu}' / 'episodes.jsonl'))
    bar = tqdm(
        total=N_EVAL,
        initial=len(done),
        desc='DINO-WM Stage 0',
        unit='episode',
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format=(
            '{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} '
            '[{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        ),
    )
    credit = sum(float(row['wall_s']) for row in done.values())
    bar.start_t -= credit
    started = time.perf_counter()
    try:
        while True:
            rows = {}
            counts = []
            for gpu, _, _ in SHARDS:
                shard_rows = _read_rows(OUT_DIR / f'gpu{gpu}' / 'episodes.jsonl')
                rows.update(shard_rows)
                counts.append(len(shard_rows))
            if len(rows) > bar.n:
                bar.update(len(rows) - bar.n)
            n_obj = sum(bool(row['object_ever']) for row in rows.values())
            n_full = sum(bool(row['full_ever']) for row in rows.values())
            shown = max(len(rows), 1)
            bar.set_postfix_str(
                f'Object Success: {n_obj}/{len(rows)} | '
                f'Full: {n_full}/{len(rows)} | GPU: {", ".join(map(str, counts))}',
                refresh=False,
            )
            bar.refresh()
            if all(proc.poll() is not None for proc in procs):
                break
            time.sleep(2)
    finally:
        bar.close()
        for proc, handle in zip(procs, logs):
            proc.wait()
            handle.close()
    failed = [proc.returncode for proc in procs if proc.returncode]
    if failed:
        raise SystemExit(f'workers failed with codes {failed}')
    _merge(time.perf_counter() - started)


def _merge(wall_s: float) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_stage0_pusht import aggregate, write_table

    rows = {}
    gpu_times = {}
    for gpu, start, end in SHARDS:
        shard_rows = _read_rows(OUT_DIR / f'gpu{gpu}' / 'episodes.jsonl')
        overlap = set(rows) & set(shard_rows)
        if overlap:
            raise SystemExit(f'duplicate pairs {sorted(overlap)}')
        rows.update(shard_rows)
        timing_path = OUT_DIR / f'gpu{gpu}' / 'timing.json'
        gpu_times[f'gpu{gpu}'] = (
            json.loads(timing_path.read_text()) if timing_path.exists() else {}
        )
        gpu_times[f'gpu{gpu}']['n_rows'] = len(shard_rows)
        gpu_times[f'gpu{gpu}']['pair_range'] = [start, end - 1]
    missing = [i for i in range(N_EVAL) if i not in rows]
    if missing:
        raise SystemExit(f'missing pairs {missing[:10]} count={len(missing)}')
    ordered = [rows[i] for i in range(N_EVAL)]
    summary = aggregate(ordered, wall_s)
    lewm = _lewm_by_pair()
    summary['comparison'] = _compare(ordered, lewm)
    summary['gpu'] = gpu_times
    payload = {
        'complete': True,
        'primary_metric': 'object_only_success_ever',
        'summary': summary,
        'episodes': ordered,
    }
    _atomic(OUT_DIR / 'metrics.json', payload)
    write_table(OUT_DIR / 'episode_results.csv', ordered)
    _atomic(
        OUT_DIR / 'progress.json',
        {'n_completed': N_EVAL, 'n_total': N_EVAL, 'complete': True},
    )
    print(
        f'object ever {summary["object_only_success_count_ever"]}/100  '
        f'final {summary["object_only_success_count_final"]}/100  '
        f'full {summary["full_state_success_count"]}/100  '
        f'wall {wall_s:.1f}s',
        file=sys.stderr,
    )


def _lewm_by_pair() -> dict[int, dict]:
    path = LEWM_DIR / 'episode_results.csv'
    out = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out[int(row['pair_index'])] = row
    return out


def _flag(row, key):
    value = row[key]
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes'}
    return bool(value)


def _compare(dino_rows, lewm_rows) -> dict:
    def _overlap(key):
        both = lewm_only = dino_only = neither = 0
        for row in dino_rows:
            other = lewm_rows[int(row['pair_index'])]
            left = _flag(other, key)
            right = bool(row[key])
            if left and right:
                both += 1
            elif left:
                lewm_only += 1
            elif right:
                dino_only += 1
            else:
                neither += 1
        return {
            'both_success': both,
            'lewm_only': lewm_only,
            'dinowm_only': dino_only,
            'neither': neither,
        }

    return {
        'object_only_success_ever': _overlap('object_ever'),
        'object_only_success_final': _overlap('object_final'),
        'full_state_success': _overlap('full_ever'),
    }


def _atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _open_dataset(log: logging.Logger):
    """Open the local copy of the PushT Lance table LeWM used.

    Four workers used to read the Hugging Face copy at once and got
    rate-limited. The table now lives on disk. The lock only keeps the
    first column scan from running four times at the same moment.
    """
    import stable_worldmodel as swm

    lock_path = OUT_DIR / 'dataset.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(8):
        try:
            with lock_path.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    return swm.data.load_dataset(
                        DATASET_NAME,
                        keys_to_cache=['action', 'proprio', 'state'],
                    )
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        except Exception as exc:
            text = f'{type(exc).__name__}: {exc}'
            if 'RateLimited' not in text and '429' not in text:
                raise
            last = exc
            wait = 40 * (attempt + 1)
            log.warning('dataset open retry %s in %ss', attempt + 1, wait)
            time.sleep(wait)
    raise RuntimeError('could not open the PushT Lance table') from last


def _worker(gpu: int, start: int, end: int) -> None:
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import warnings

    warnings.filterwarnings('ignore', module='gymnasium')
    logging.getLogger('gymnasium').setLevel(logging.ERROR)
    logging.getLogger('stable_worldmodel.planning.solver.cem').setLevel(logging.INFO)
    logging.getLogger('stable_worldmodel.planning.solver.cem').propagate = False

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import torch
    from torchvision.transforms import v2 as transforms
    import stable_worldmodel as swm

    from dinowm_adapter import CKPT_DIR, DinoWMAdapter, build_cost
    from run_stage0_pusht import (
        CALLABLES,
        EVAL_BUDGET,
        GOAL_OFFSET,
        PLAN,
        SEED,
        _action_report,
        _append_jsonl,
        _json_dump,
        summarize_episode,
    )

    shard = OUT_DIR / f'gpu{gpu}'
    shard.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f'dinowm.gpu{gpu}')
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.FileHandler(shard / 'eval.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(handler)
    logging.getLogger('stable_worldmodel.planning.solver.cem').addHandler(handler)

    pairs = json.loads(_pairs_path().read_text())['pairs']
    shard_pairs = [pair for pair in pairs if start <= int(pair['pair_index']) < end]
    done = _read_rows(shard / 'episodes.jsonl')
    norm = json.loads((CKPT_DIR / 'swm_h5_norm_stats.json').read_text())
    scaler = swm.data.ZScoreScaler(
        mean=np.asarray(norm['action_mean'], dtype=np.float64).reshape(1, -1),
        std=np.asarray(norm['action_std'], dtype=np.float64).reshape(1, -1),
    )
    transform = {
        key: transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        for key in ('pixels', 'goal')
    }
    log.info('loading DINO-WM on cuda (visible gpu %s)', gpu)
    model = DinoWMAdapter().to('cuda').eval()
    weight = model.proprio_encoder.patch_embed.weight.detach()
    log.info(
        'proprio kernel max %.6f norm %.6f',
        float(weight.abs().max()),
        float(weight.norm()),
    )
    policy = swm.policy.WorldModelPolicy(
        solver=swm.planning.CEMSolver(
            cost=build_cost(model),
            batch_size=1,
            num_samples=300,
            var_scale=1.0,
            n_steps=30,
            topk=30,
            device='cuda',
            seed=SEED,
        ),
        config=swm.PlanConfig(**PLAN),
        process={'action': scaler},
        transform=transform,
    )
    dataset = _open_dataset(log)
    world = swm.World(
        env_name='swm/PushT-v1',
        num_envs=1,
        max_episode_steps=EVAL_BUDGET * 2,
        image_shape=(224, 224),
    )
    world.set_policy(policy)
    video_dir = shard / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    shared_videos = OUT_DIR / 'videos'
    shared_videos.mkdir(parents=True, exist_ok=True)
    _json_dump(
        shard / 'config.json',
        {
            'checkpoint': str(CKPT_DIR),
            'architecture': 'gaoyuezhou/dino_wm VWorldModel',
            'dataset': DATASET_NAME,
            'encoder': 'dinov2_vits14 frozen, images resized 224 -> 196',
            'pixels': 'default_transform: [0, 1] then Normalize(0.5)',
            'rgb_only': True,
            'proprio': (
                'noprop arm. proprio_key is null. Train and plan feed a '
                'constant zero of dim 1. The 10-d slot stays in the latent. '
                'PushT state is not an input, and the proprio term is not in the cost.'
            ),
            'action_normalization': {
                'mean': norm['action_mean'],
                'std': norm['action_std'],
                'source': 'swm_h5_norm_stats.json',
            },
            'cost': 'last-frame visual patch MSE',
            'plan_config': PLAN,
            'cem_seed': '42 + pair_index',
            'pairs': 'results/stage0/eval100/pairs.json',
            'gpu': gpu,
            'pair_start': start,
            'pair_end': end,
        },
    )
    if gpu == 0 and not (OUT_DIR / 'pairs.json').exists():
        _json_dump(OUT_DIR / 'pairs.json', json.loads(_pairs_path().read_text()))
        _json_dump(OUT_DIR / 'config.json', json.loads((shard / 'config.json').read_text()))
    started = time.perf_counter()
    try:
        for pair in shard_pairs:
            idx = int(pair['pair_index'])
            if idx in done:
                continue
            policy.reset_episode_state()
            policy.solver.torch_gen.manual_seed(SEED + idx)
            model._cached_pixels = None
            tick = time.perf_counter()
            metrics = None
            for attempt in range(4):
                try:
                    metrics = world.evaluate(
                        dataset=dataset,
                        episodes_idx=[int(pair['episode_idx'])],
                        start_steps=[int(pair['start_step'])],
                        goal_offset=GOAL_OFFSET,
                        eval_budget=EVAL_BUDGET,
                        callables=CALLABLES,
                        video=video_dir,
                        record_trace=True,
                    )
                    break
                except Exception as exc:
                    text = f'{type(exc).__name__}: {exc}'
                    retryable = 'RateLimited' in text or 'LanceError' in text
                    if not retryable or attempt == 3:
                        raise
                    policy.reset_episode_state()
                    policy.solver.torch_gen.manual_seed(SEED + idx)
                    model._cached_pixels = None
                    wait = 20 * (attempt + 1)
                    log.warning('pair %s retry %s after %ss: %s', idx, attempt + 1, wait, text[:300])
                    time.sleep(wait)
            wall = time.perf_counter() - tick
            actions = _action_report(policy)
            if actions['out_of_bounds'] or actions['n_finite'] == 0:
                raise RuntimeError(f'pair {idx} action check failed: {actions}')
            src = video_dir / 'env_0.mp4'
            dst = shared_videos / (
                f'pair_{idx:03d}_ep{int(pair["episode_idx"])}_s{int(pair["start_step"])}.mp4'
            )
            if src.exists():
                src.replace(dst)
            saved = dict(pair)
            saved['video'] = str(dst) if dst.exists() else None
            row = summarize_episode(
                saved,
                metrics['traces'][0],
                bool(np.asarray(metrics['episode_successes'])[0]),
                wall,
                list(policy.solve_times),
                list(policy.solve_batches),
            )
            row['gpu'] = gpu
            if row['criterion_mismatch']:
                raise RuntimeError(f'pair {idx} criterion mismatch')
            done[idx] = row
            _append_jsonl(shard / 'episodes.jsonl', row)
            log.info(
                'pair %s object_ever=%s object_final=%s full=%s wall=%.2fs replans=%s',
                idx,
                int(row['object_ever']),
                int(row['object_final']),
                int(row['full_ever']),
                wall,
                row['n_replans'],
            )
    finally:
        world.close()
        _json_dump(
            shard / 'timing.json',
            {
                'wall_s': time.perf_counter() - started,
                'n_completed': len(done),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=25)
    args = parser.parse_args()
    if args.worker:
        try:
            _worker(args.gpu, args.start, args.end)
        except Exception:
            logging.getLogger(f'dinowm.gpu{args.gpu}').error(
                'worker failed\n%s', traceback.format_exc()
            )
            raise
        return
    _monitor()


if __name__ == '__main__':
    main()
