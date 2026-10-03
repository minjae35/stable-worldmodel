"""CEM horizon sweep on the fixed PushT 5k checkpoint.

Only the planning horizon changes. Receding execution stays 1 block, and
candidates, elites, iterations, pairs, and seeds stay as in the eval protocol.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.data import preprocess
from rdwm.exp1.evaluate import get_pairs, load_model
from rdwm.exp1.plan_diag import _latent_mse, _spearman
from rdwm.exp1.planning import (
    CEMPlanner,
    _encode_frame,
    _pusht_goal_distance,
    open_episode_reader,
    open_eval_session,
)
from rdwm.exp1.progress import Progress
from rdwm.envs.sessions import _row, dataset_frame


def _cfg_horizon(cfg: dict, horizon: int) -> dict:
    out = copy.deepcopy(cfg)
    out['cem']['horizon'] = int(horizon)
    out['cem']['receding_horizon'] = 1
    return out


@torch.no_grad()
def _predict_plan(model, z_hist, a_hist, plan, z_goal, env: str, bf16: bool):
    """Roll the chosen mean plan. Returns per-step latents [K,N,D] and terminal cost."""
    acts = plan[None]
    if a_hist.shape[1] == 0:
        ah = acts.new_zeros((1, 0, plan.shape[1], plan.shape[-1]))
    else:
        ah = a_hist
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        preds = model.rollout(z_hist, ah, acts, env)
    preds = preds[0].float()
    cost = _latent_mse(preds[-1], z_goal)
    return preds, cost


def _closed_loop(model, session, episode, pair, cfg, device, seed: int) -> dict:
    """Same loop as run_pair, plus the model's own terminal cost and 1-block error."""
    env = session.name
    ecfg, ccfg = cfg['eval'], cfg['cem']
    block = int(cfg['model']['action_block'])
    bf16 = bool(cfg['train']['bf16']) and device.type == 'cuda'
    start, goal = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal)
    initial = _pusht_goal_distance(session)['pos']
    planner = CEMPlanner(
        model, env, cfg, session.action_low, session.action_high, device, seed
    )
    z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
    z_hist = [_encode_frame(model, session.render(), cfg, device)]
    a_hist: list[torch.Tensor] = []
    steps, success, warm = 0, False, None
    pred_costs, step_errors = [], []
    budget = int(ecfg['exec_budget'])
    while steps < budget and not success:
        h = min(len(z_hist), int(ccfg['history']))
        zh = torch.stack(z_hist[-h:], dim=1)
        if h > 1:
            ah = torch.stack(a_hist[-(h - 1):], dim=0)[None]
        else:
            ah = torch.zeros((1, 0, block, planner.low.shape[0]), device=device)
        plan, info = planner.plan(zh, ah, z_goal, warm)
        preds, term_cost = _predict_plan(model, zh, ah, plan, z_goal, env, bf16)
        pred_costs.append(term_cost)
        native = plan[:1].reshape(-1, plan.shape[-1]).detach().cpu().numpy()
        executed = 0
        for action in native:
            session.step(np.asarray(action, dtype=np.float64))
            executed += 1
            steps += 1
            if session.success():
                success = True
                break
            if steps >= budget:
                break
        live = _encode_frame(model, session.render(), cfg, device)
        if executed == len(native):
            step_errors.append(_latent_mse(preds[0], live))
        if success or steps >= budget:
            break
        z_hist.append(live)
        a_hist.append(plan[0])
        warm = plan[1:] if ccfg['warm_start'] else None
    final = _pusht_goal_distance(session)['pos']
    reduction = (initial - final) / initial * 100.0 if initial > 0 else None
    return {
        'episode': int(pair['episode']),
        'start': start,
        'goal': goal,
        'success': bool(success),
        'goal_pos_initial': initial,
        'goal_pos_final': final,
        'goal_pos_reduction_pct': reduction,
        'closer': bool(final < initial),
        'pred_terminal_cost_mean': float(np.mean(pred_costs)),
        'pred_terminal_cost_first': pred_costs[0],
        'executed_block_pred_vs_actual_mean': float(np.mean(step_errors)) if step_errors else None,
        'replans': len(pred_costs),
    }


def _open_loop_terminal(model, session, episode, pair, cfg, device, seed: int) -> dict:
    """Execute the first CEM plan for the whole horizon, then the caller restores."""
    env = session.name
    block = int(cfg['model']['action_block'])
    horizon = int(cfg['cem']['horizon'])
    bf16 = bool(cfg['train']['bf16']) and device.type == 'cuda'
    start, goal = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal)
    planner = CEMPlanner(
        model, env, cfg, session.action_low, session.action_high, device, seed
    )
    z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
    z0 = _encode_frame(model, session.render(), cfg, device)[None]
    ah = torch.zeros((1, 0, block, planner.low.shape[0]), device=device)
    plan, info = planner.plan(z0, ah, z_goal, None)
    preds, term_cost = _predict_plan(model, z0, ah, plan, z_goal, env, bf16)
    for action in plan.reshape(-1, plan.shape[-1]).detach().cpu().numpy():
        session.step(np.asarray(action, dtype=np.float64))
    live = _encode_frame(model, session.render(), cfg, device)
    return {
        'elite_terminal_cost': info['elite_cost_last'],
        'mean_plan_terminal_cost': term_cost,
        'pred_vs_actual_terminal': _latent_mse(preds[-1], live),
        'goal_pos_after_open_loop': _pusht_goal_distance(session)['pos'],
        'native_steps': int(horizon * block),
    }


def _encode_frames(model, frames: list[np.ndarray], cfg, device) -> torch.Tensor:
    batch = torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            for frame in frames
        ]
    )
    pixels = preprocess(batch[None].to(device), cfg)
    with torch.autocast(
        'cuda', dtype=torch.bfloat16, enabled=bool(cfg['train']['bf16'])
    ):
        return model.encode(pixels).float()[0]


def _trajectory_alignment(model, pairs, cache, cfg, device) -> dict:
    """Dataset observations along the held-out trajectories that reach the goal."""
    curves = []
    all_lat, all_dist = [], []
    pre_lat, pre_dist = [], []
    for p_i, pair in enumerate(pairs):
        ep, start, goal = int(pair['episode']), int(pair['start']), int(pair['goal'])
        episode = cache[ep]
        idxs = list(range(start, goal + 1))
        frames = [dataset_frame(episode, t) for t in idxs]
        latents = _encode_frames(model, frames, cfg, device)
        z_goal = latents[-1]
        goal_state = _row(episode, goal, 'state')
        rows = []
        for j, t in enumerate(idxs):
            dist = float(np.linalg.norm(_row(episode, t, 'state')[:4] - goal_state[:4]))
            lat = _latent_mse(latents[j], z_goal)
            rows.append({'step': t - start, 'goal_pos': dist, 'latent_mse': lat})
            all_lat.append(lat)
            all_dist.append(dist)
            if t < goal:
                pre_lat.append(lat)
                pre_dist.append(dist)
        curves.append(
            {
                'pair': p_i,
                'episode': ep,
                'start': start,
                'spearman_all': _spearman([r['latent_mse'] for r in rows], [r['goal_pos'] for r in rows]),
                'spearman_before_goal': _spearman(
                    [r['latent_mse'] for r in rows[:-1]], [r['goal_pos'] for r in rows[:-1]]
                ),
                'latent_at_start': rows[0]['latent_mse'],
                'latent_at_goal': rows[-1]['latent_mse'],
                'goal_pos_at_start': rows[0]['goal_pos'],
                'by_block': [rows[k * 5] for k in range(0, 6)],
                'points': rows,
            }
        )
    return {
        'n_trajectories': len(curves),
        'pooled_spearman': _spearman(all_lat, all_dist),
        'pooled_spearman_before_goal': _spearman(pre_lat, pre_dist),
        'mean_latent_at_start': float(np.mean([c['latent_at_start'] for c in curves])),
        'mean_latent_at_goal': float(np.mean([c['latent_at_goal'] for c in curves])),
        'trajectories': curves,
    }


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def diagnose(cfg: dict, ckpt: Path, out: Path, device) -> dict:
    torch.set_grad_enabled(False)
    model = load_model(cfg, ckpt, device)
    session = open_eval_session('pusht')
    reader = open_episode_reader(env_path(cfg, 'pusht'))
    pairs = get_pairs(cfg, 'pusht', 'val', cfg['eval']['pilot_val_pairs'], session, reader)[:10]
    cache: dict[int, dict] = {}

    def episode_of(ep: int) -> dict:
        if ep not in cache:
            cache.clear()
            cache[ep] = reader.load_episode(ep)
        return cache[ep]

    horizons = (1, 2, 3, 5)
    log = out.parent / 'horizon_diag_progress.log'
    bar = Progress(len(horizons) * len(pairs), 0, 'CEM horizon | PushT', log)
    per_h = []
    done = 0
    for horizon in horizons:
        hcfg = _cfg_horizon(cfg, horizon)
        rows, terminals = [], []
        for i, pair in enumerate(pairs):
            episode = episode_of(int(pair['episode']))
            seed = 1000 + i
            terminals.append(_open_loop_terminal(model, session, episode, pair, hcfg, device, seed))
            row = _closed_loop(model, session, episode, pair, hcfg, device, seed)
            row['open_loop_terminal'] = terminals[-1]
            rows.append(row)
            done += 1
            bar.desc = f'CEM H={horizon} | PushT'
            bar.set_postfix(
                f"pair {i + 1}/10 success={row['success']} "
                f"dist {row['goal_pos_initial']:.0f}->{row['goal_pos_final']:.0f}"
            )
            bar.update(done)
        per_h.append(
            {
                'horizon_blocks': horizon,
                'receding_blocks': 1,
                'seed0': 1000,
                'success_count': int(sum(r['success'] for r in rows)),
                'success_rate': float(np.mean([r['success'] for r in rows])),
                'goal_pos_initial': _mean(rows, 'goal_pos_initial'),
                'goal_pos_final': _mean(rows, 'goal_pos_final'),
                'goal_pos_reduction_pct': _mean(rows, 'goal_pos_reduction_pct'),
                'fraction_closer': float(np.mean([r['closer'] for r in rows])),
                'pred_terminal_cost': _mean(rows, 'pred_terminal_cost_mean'),
                'pred_vs_actual_terminal': _mean(terminals, 'pred_vs_actual_terminal'),
                'executed_block_pred_vs_actual': _mean(rows, 'executed_block_pred_vs_actual_mean'),
                'pairs': rows,
            }
        )
    bar.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'ckpt': str(ckpt), 'horizons': per_h}, indent=2))

    bar = Progress(len(pairs), 0, 'goal latent alignment | PushT', log)
    curves = []
    all_lat, all_dist, pre_lat, pre_dist = [], [], [], []
    for i, pair in enumerate(pairs):
        episode_of(int(pair['episode']))
        one = _trajectory_alignment(model, [pair], cache, cfg, device)['trajectories'][0]
        curves.append(one)
        for j, point in enumerate(one['points']):
            all_lat.append(point['latent_mse'])
            all_dist.append(point['goal_pos'])
            if j < len(one['points']) - 1:
                pre_lat.append(point['latent_mse'])
                pre_dist.append(point['goal_pos'])
        bar.set_postfix(
            f"trajectory {i + 1}/10 spearman {one['spearman_all']:.2f}"
        )
        bar.update(i + 1)
    bar.close()
    alignment = {
        'n_trajectories': len(curves),
        'pooled_spearman': _spearman(all_lat, all_dist),
        'pooled_spearman_before_goal': _spearman(pre_lat, pre_dist),
        'mean_latent_at_start': float(np.mean([c['latent_at_start'] for c in curves])),
        'mean_latent_at_goal': float(np.mean([c['latent_at_goal'] for c in curves])),
        'trajectories': curves,
    }

    report = {
        'ckpt': str(ckpt),
        'note': (
            'Horizon sweep only. 5k PushT specialist, one cosine. '
            'Same 10 val pairs, seed 1000+pair, CEM 300/30/30, receding 1 block.'
        ),
        'horizons': per_h,
        'latent_alignment': alignment,
    }
    out.write_text(json.dumps(report, indent=2))
    _print(report)
    session.close()
    return report


def _print(report: dict) -> None:
    print('--- CEM horizon sweep (receding 1 block, 10 pairs) ---', flush=True)
    for row in report['horizons']:
        print(
            f"H={row['horizon_blocks']}: success {row['success_count']}/10 | "
            f"dist {row['goal_pos_initial']:.1f}->{row['goal_pos_final']:.1f} | "
            f"reduction {row['goal_pos_reduction_pct']:.1f}% | "
            f"closer {row['fraction_closer']:.0%} | "
            f"pred_cost {row['pred_terminal_cost']:.3f} | "
            f"terminal_err {row['pred_vs_actual_terminal']:.3f} | "
            f"exec_err {row['executed_block_pred_vs_actual']:.3f}",
            flush=True,
        )
    al = report['latent_alignment']
    print('--- held-out trajectories, latent MSE vs goal distance ---', flush=True)
    print(
        f"pooled spearman {al['pooled_spearman']:.3f} | "
        f"before goal frame {al['pooled_spearman_before_goal']:.3f} | "
        f"latent start {al['mean_latent_at_start']:.3f} goal {al['mean_latent_at_goal']:.3f}",
        flush=True,
    )
    for curve in al['trajectories']:
        blocks = ' '.join(f"{p['latent_mse']:.3f}" for p in curve['by_block'])
        print(
            f"pair {curve['pair']}: spearman {curve['spearman_all']:.2f} "
            f"before_goal {curve['spearman_before_goal']:.2f} latent[{blocks}]",
            flush=True,
        )
