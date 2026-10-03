"""Closed-loop CEM evaluation of a checkpoint on fixed (start, goal) pairs."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import env_path, pairs_dir
from rdwm.exp1.data import EnvIndex, split_episodes
from rdwm.exp1.model import RDWM
from rdwm.exp1.planning import (
    make_pairs,
    open_episode_reader,
    open_eval_session,
    pairs_path,
    run_pair,
)
from rdwm.exp1.progress import Progress
from rdwm.exp1.util import file_lock, write_json


def load_model(cfg: dict, ckpt_path: Path, device) -> RDWM:
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    meta = state['meta']
    envs = meta['envs']
    run_meta = json.loads((ckpt_path.parent / 'meta.json').read_text())
    variant = run_meta['variant']
    encoder_name = run_meta.get('encoder', meta.get('encoder', 'vit'))
    saved_model = run_meta.get('config', {}).get('model', {})
    cfg = copy.deepcopy(cfg)
    for key in (
        'dyn_depth',
        'dyn_heads',
        'dyn_head_dim',
        'dyn_ffn',
        'dyn_dropout',
        'action_conditioning',
        'action_emb_dim',
    ):
        if key in saved_model:
            cfg['model'][key] = saved_model[key]
    dims = {e: int(cfg['envs'][e]['action_dim']) for e in envs}
    visual = None
    if encoder_name == 'frozen_dinov2_small':
        from rdwm.exp1.dino_encoder import FrozenDINOv2Encoder

        visual = FrozenDINOv2Encoder()
    elif encoder_name != 'vit':
        raise RuntimeError(f'{ckpt_path}: unknown encoder {encoder_name}')
    model = RDWM(cfg, variant, envs, dims, encoder=visual)
    model.load_state_dict(state['model'])
    for env in envs:
        if not bool(model.action_norm[env].set):
            raise RuntimeError(f'{ckpt_path}: action stats for {env} were never set')
    return model.to(device).eval()


def get_pairs(cfg: dict, env: str, split: str, n: int, session, reader) -> list[dict]:
    """Fixed pair list for (env, split); built once with the simulator."""
    path = pairs_path(pairs_dir(cfg), split, env, n)
    with file_lock(path.with_suffix('.lock')):
        if not path.exists():
            index = EnvIndex(env, env_path(cfg, env), cfg['paths']['cache_root'])
            episodes = split_episodes(index.num_episodes, cfg['data']['split'])[split]
            payload = make_pairs(
                session, reader, episodes, index.lengths, n, cfg, f'{env}/{split}'
            )
            write_json(path, payload)
    return json.loads(path.read_text())['pairs']


def evaluate(
    cfg: dict,
    ckpt_path: Path,
    split: str,
    pair_list_size: int,
    use_pairs: int,
    envs: list[str] | None = None,
) -> dict:
    device = torch.device('cuda')
    model = load_model(cfg, ckpt_path, device)
    envs = envs or model.envs
    out_dir = ckpt_path.parent / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'ckpt': str(ckpt_path), 'split': split, 'per_env': {}}
    for env in envs:
        session = open_eval_session(env)
        if session.action_dim != cfg['envs'][env]['action_dim']:
            raise RuntimeError(f'{env}: runtime action dim {session.action_dim}')
        reader = open_episode_reader(env_path(cfg, env))
        pairs = get_pairs(cfg, env, split, pair_list_size, session, reader)[:use_pairs]
        results, cache = [], {}
        t0 = time.time()
        label = {'pusht': 'PushT'}.get(env, env)
        step = ckpt_path.stem.replace('ckpt_', '')
        bar = Progress(
            len(pairs),
            0,
            f'CEM eval @{step} | {label}',
            out_dir / f'{ckpt_path.stem}_{split}_{env}_progress.log',
        )
        bar.set_postfix('waiting for the first pair')
        bar.update(0)
        for i, pair in enumerate(pairs):
            ep = int(pair['episode'])
            if ep not in cache:
                cache = {ep: reader.load_episode(ep)}
            res = run_pair(model, session, cache[ep], pair, cfg, device, seed=1000 + i)
            results.append(res)
            dist = res.get('goal_pos_final')
            dist_s = f'{dist:.1f}' if isinstance(dist, float) else '-'
            bar.set_postfix(
                f"pair {i + 1} success={res.get('success')} "
                f"goal_dist={dist_s} steps={res.get('steps')}"
            )
            bar.update(i + 1)
        bar.close()
        session.close()
        done = [r for r in results if not r.get('skipped_start_success')]
        summary['per_env'][env] = {
            'pairs': len(results),
            'evaluated': len(done),
            'success_rate': float(np.mean([r['success'] for r in done])) if done else None,
            'mean_replans': float(np.mean([r['replans'] for r in done])) if done else None,
            'cost_first_iter': float(np.mean([r['elite_cost_first_iter_mean'] for r in done])),
            'cost_last_iter': float(np.mean([r['elite_cost_last_iter_mean'] for r in done])),
            'sec_per_pair': (time.time() - t0) / max(1, len(results)),
            **_distance_summary(done),
        }
        write_json(out_dir / f'{ckpt_path.stem}_{split}_{env}.json', {'results': results})
    rates = [v['success_rate'] for v in summary['per_env'].values() if v['success_rate'] is not None]
    summary['macro_success'] = float(np.mean(rates)) if rates else None
    summary['worst_env_success'] = float(np.min(rates)) if rates else None
    write_json(out_dir / f'{ckpt_path.stem}_{split}_summary.json', summary)
    return summary


def _distance_summary(rows: list[dict]) -> dict:
    vals = [r['goal_pos_initial'] for r in rows if r.get('goal_pos_initial') is not None]
    if not vals:
        return {}
    final = [r['goal_pos_final'] for r in rows]
    red = [r['goal_pos_reduction_pct'] for r in rows if r.get('goal_pos_reduction_pct') is not None]
    return {
        'goal_pos_initial': float(np.mean(vals)),
        'goal_pos_final': float(np.mean(final)),
        'goal_pos_reduction_pct': float(np.mean(red)) if red else None,
    }
