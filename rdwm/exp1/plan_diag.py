"""Why PushT CEM latent cost falls while simulator goal distance rises.

Diagnostic only. The model, loss, and CEM hyperparameters are unchanged.
The checkpoint under test is the 5k PushT specialist: one cosine over 5,000
updates, and the same goal-distance failure is already present there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.data import EnvIndex, preprocess
from rdwm.exp1.evaluate import get_pairs, load_model
from rdwm.exp1.planning import (
    CEMPlanner,
    _encode_frame,
    _pusht_goal_distance,
    open_episode_reader,
    open_eval_session,
)
from rdwm.exp1.progress import Progress
from rdwm.envs.sessions import _row, dataset_frame


def _corr(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return _corr(rx, ry)


def _native(index: EnvIndex, episode: int, start: int, n: int) -> np.ndarray:
    off = int(index.offsets[episode]) + int(start)
    return np.asarray(index.actions[off : off + n], dtype=np.float32)


def _blocks(native: np.ndarray, block: int) -> np.ndarray:
    return native.reshape(-1, block, native.shape[-1])


def _step_all(session, native: np.ndarray) -> None:
    for action in native:
        session.step(np.asarray(action, dtype=np.float64))


def _latent_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).square().mean())


@torch.no_grad()
def _rollout_terminal(model, z_start, actions: torch.Tensor, env: str, bf16: bool):
    """actions [B, K, 5, d] -> terminal latent [B, N, D]."""
    b = actions.shape[0]
    z = z_start.expand(b, *z_start.shape[1:])
    ah = actions.new_zeros((b, 0, actions.shape[2], actions.shape[-1]))
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        preds = model.rollout(z, ah, actions, env)
    return preds[:, -1].float()


@torch.no_grad()
def _rollout_all(model, z_start, actions: torch.Tensor, env: str, bf16: bool):
    """actions [1, K, 5, d] -> latents [K, N, D]."""
    ah = actions.new_zeros((1, 0, actions.shape[2], actions.shape[-1]))
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        preds = model.rollout(z_start, ah, actions, env)
    return preds[0].float()


def _sample_candidates(planner: CEMPlanner, n: int) -> torch.Tensor:
    shape = (planner.horizon, planner.block, planner.low.shape[0])
    mean = planner.center.expand(shape)
    std = planner.init_std.expand(shape)
    noise = torch.randn((n, *shape), generator=planner.gen, device=planner.device)
    return planner.clamp(noise * std + mean)


def _frame_pair(episode, session, index: int, cfg, device, model):
    data = dataset_frame(episode, index)
    live = session.render()
    pix = float(np.mean(np.abs(data.astype(np.float32) - live.astype(np.float32))))
    td = torch.from_numpy(np.ascontiguousarray(data)).permute(2, 0, 1)
    tl = torch.from_numpy(np.ascontiguousarray(live)).permute(2, 0, 1)
    both = preprocess(torch.stack([td, tl])[None].to(device), cfg)[0]
    pre = float((both[0] - both[1]).abs().mean())
    z_data = _encode_frame(model, data, cfg, device)
    z_live = _encode_frame(model, live, cfg, device)
    return {
        'pixel_mae': pix,
        'preprocessed_mae': pre,
        'latent_mse': _latent_mse(z_data, z_live),
        'z_data': z_data,
        'z_live': z_live,
    }


def diagnose(cfg: dict, ckpt: Path, out: Path, n_candidates: int, n_pairs: int, device) -> dict:
    env = 'pusht'
    torch.set_grad_enabled(False)
    model = load_model(cfg, ckpt, device)
    bf16 = bool(cfg['train']['bf16'])
    block = int(cfg['model']['action_block'])
    horizon = int(cfg['cem']['horizon'])
    session = open_eval_session(env)
    reader = open_episode_reader(env_path(cfg, env))
    index = EnvIndex(env, env_path(cfg, env), cfg['paths']['cache_root'])
    pairs = get_pairs(cfg, env, 'val', cfg['eval']['pilot_val_pairs'], session, reader)[:n_pairs]
    cache: dict[int, dict] = {}
    gpu = '0'
    log = out.parent / 'plan_diag_progress.log'

    def episode_of(ep: int) -> dict:
        if ep not in cache:
            cache.clear()
            cache[ep] = reader.load_episode(ep)
        return cache[ep]

    report: dict = {
        'ckpt': str(ckpt),
        'note': (
            '5k PushT specialist, one cosine over 5000 updates. '
            'Diagnostic of the cost-vs-distance split. Not an official 10k result.'
        ),
        'n_pairs': len(pairs),
        'n_candidates': n_candidates,
        'horizon_blocks': horizon,
        'action_block': block,
    }

    # --- 1. latent cost vs true goal distance ---
    bar = Progress(len(pairs) * n_candidates, 0, f'PushT A plan-diag | GPU {gpu}', log)
    bar.set_postfix('latent cost vs goal distance')
    points = []
    per_pair = []
    seen = 0
    for p_i, pair in enumerate(pairs):
        ep, start, goal = int(pair['episode']), int(pair['start']), int(pair['goal'])
        episode = episode_of(ep)
        session.restore(episode, start)
        session.set_goal(episode, goal)
        z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
        z_start = _encode_frame(model, session.render(), cfg, device)[None]
        planner = CEMPlanner(
            model, env, cfg, session.action_low, session.action_high, device, seed=1000 + p_i
        )
        cand = _sample_candidates(planner, n_candidates)
        terminal = _rollout_terminal(model, z_start, cand, env, bf16)
        pred_cost = (terminal - z_goal).square().mean(dim=(-1, -2)).detach().cpu().numpy()
        row_cost, row_dist, row_actual = [], [], []
        for j in range(n_candidates):
            session.restore(episode, start)
            session.set_goal(episode, goal)
            _step_all(session, cand[j].reshape(-1, cand.shape[-1]).cpu().numpy())
            dist = _pusht_goal_distance(session)['pos']
            actual = _latent_mse(_encode_frame(model, session.render(), cfg, device), z_goal)
            row_cost.append(float(pred_cost[j]))
            row_dist.append(dist)
            row_actual.append(actual)
            points.append(
                {'pair': p_i, 'pred_cost': float(pred_cost[j]), 'goal_pos': dist, 'actual_latent_cost': actual}
            )
            seen += 1
            if seen % 4 == 0 or j + 1 == n_candidates:
                bar.set_postfix(
                    f'correlation pair {p_i + 1}/{len(pairs)} | '
                    f'pred_cost={pred_cost[j]:.3f} | goal_dist={dist:.1f}'
                )
                bar.update(seen)
        per_pair.append(
            {
                'pair': p_i,
                'episode': ep,
                'start': start,
                'pearson_pred_vs_dist': _corr(row_cost, row_dist),
                'spearman_pred_vs_dist': _spearman(row_cost, row_dist),
                'pearson_actual_latent_vs_dist': _corr(row_actual, row_dist),
                'spearman_actual_latent_vs_dist': _spearman(row_actual, row_dist),
                'pearson_pred_vs_actual_latent': _corr(row_cost, row_actual),
                'spearman_pred_vs_actual_latent': _spearman(row_cost, row_actual),
            }
        )
    bar.close()
    costs = [p['pred_cost'] for p in points]
    dists = [p['goal_pos'] for p in points]
    actuals = [p['actual_latent_cost'] for p in points]
    report['correlation'] = {
        'pooled': {
            'n': len(points),
            'pearson_pred_cost_vs_goal_pos': _corr(costs, dists),
            'spearman_pred_cost_vs_goal_pos': _spearman(costs, dists),
            'pearson_actual_latent_vs_goal_pos': _corr(actuals, dists),
            'spearman_actual_latent_vs_goal_pos': _spearman(actuals, dists),
            'pearson_pred_vs_actual_latent': _corr(costs, actuals),
            'spearman_pred_vs_actual_latent': _spearman(costs, actuals),
        },
        'per_pair': per_pair,
        'points': points,
    }

    # --- 2. CEM vs random vs dataset sequence ---
    bar = Progress(len(pairs) * 3, 0, f'PushT A plan-diag | GPU {gpu}', log)
    comparisons = []
    done = 0
    for p_i, pair in enumerate(pairs):
        ep, start, goal = int(pair['episode']), int(pair['start']), int(pair['goal'])
        episode = episode_of(ep)
        session.restore(episode, start)
        session.set_goal(episode, goal)
        z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
        z_start = _encode_frame(model, session.render(), cfg, device)[None]
        planner = CEMPlanner(
            model, env, cfg, session.action_low, session.action_high, device, seed=2000 + p_i
        )
        cem_plan, cem_info = planner.plan(
            z_start, z_start.new_zeros((1, 0, block, planner.low.shape[0])), z_goal
        )
        sequences = {
            'cem': cem_plan.cpu().numpy(),
            'random': _sample_candidates(planner, 1)[0].cpu().numpy(),
            'dataset': _blocks(_native(index, ep, start, horizon * block), block),
        }
        for name, seq in sequences.items():
            seq_t = torch.as_tensor(seq, dtype=torch.float32, device=device)[None]
            pred = _rollout_terminal(model, z_start, seq_t, env, bf16)[0]
            pred_cost = _latent_mse(pred, z_goal)
            session.restore(episode, start)
            session.set_goal(episode, goal)
            before = _pusht_goal_distance(session)['pos']
            _step_all(session, seq.reshape(-1, seq.shape[-1]))
            after = _pusht_goal_distance(session)['pos']
            live = _encode_frame(model, session.render(), cfg, device)
            flat = seq.reshape(-1, seq.shape[-1])
            comparisons.append(
                {
                    'pair': p_i,
                    'which': name,
                    'pred_cost': pred_cost,
                    'goal_pos_before': before,
                    'goal_pos_after': after,
                    'rollout_error': _latent_mse(pred, live),
                    'actual_latent_cost': _latent_mse(live, z_goal),
                    'action_abs_mean': float(np.abs(flat).mean()),
                    'action_abs_max': float(np.abs(flat).max()),
                    'cem_elite_cost_last': cem_info['elite_cost_last'] if name == 'cem' else None,
                }
            )
            done += 1
            bar.set_postfix(
                f'pair {p_i + 1} {name} | pred_cost={pred_cost:.3f} | goal_dist={after:.1f}'
            )
            bar.update(done)
    bar.close()
    report['exploitation'] = comparisons

    # --- 3. multi-step rollout error on the dataset action sequence ---
    bar = Progress(len(pairs), 0, f'PushT A plan-diag | GPU {gpu}', log)
    horizons = []
    for p_i, pair in enumerate(pairs):
        ep, start = int(pair['episode']), int(pair['start'])
        episode = episode_of(ep)
        native = _native(index, ep, start, horizon * block)
        session.restore(episode, start)
        z_live0 = _encode_frame(model, session.render(), cfg, device)[None]
        z_data0 = _encode_frame(model, dataset_frame(episode, start), cfg, device)[None]
        live_z, data_z = [], []
        for k in range(horizon):
            _step_all(session, native[k * block : (k + 1) * block])
            live_z.append(_encode_frame(model, session.render(), cfg, device))
            data_z.append(_encode_frame(model, dataset_frame(episode, start + (k + 1) * block), cfg, device))
        acts = torch.as_tensor(_blocks(native, block), dtype=torch.float32, device=device)[None]
        pred_live = _rollout_all(model, z_live0, acts, env, bf16)
        pred_data = _rollout_all(model, z_data0, acts, env, bf16)
        rows = []
        for k in range(horizon):
            rows.append(
                {
                    'blocks': k + 1,
                    'native_steps': (k + 1) * block,
                    'model_from_live_vs_live': _latent_mse(pred_live[k], live_z[k]),
                    'model_from_dataset_vs_dataset': _latent_mse(pred_data[k], data_z[k]),
                    'live_vs_dataset': _latent_mse(live_z[k], data_z[k]),
                }
            )
        horizons.append({'pair': p_i, 'start_live_vs_dataset': _latent_mse(z_live0[0], z_data0[0]), 'by_horizon': rows})
        last = rows[-1]
        bar.set_postfix(
            f'rollout pair {p_i + 1} | h5 live={last["model_from_live_vs_live"]:.3f} '
            f'data={last["model_from_dataset_vs_dataset"]:.3f}'
        )
        bar.update(p_i + 1)
    bar.close()
    mean_h = []
    for k in range(horizon):
        mean_h.append(
            {
                'blocks': k + 1,
                'model_from_live_vs_live': float(np.mean([h['by_horizon'][k]['model_from_live_vs_live'] for h in horizons])),
                'model_from_dataset_vs_dataset': float(np.mean([h['by_horizon'][k]['model_from_dataset_vs_dataset'] for h in horizons])),
                'live_vs_dataset': float(np.mean([h['by_horizon'][k]['live_vs_dataset'] for h in horizons])),
            }
        )
    report['rollout'] = {
        'start_live_vs_dataset': float(np.mean([h['start_live_vs_dataset'] for h in horizons])),
        'mean_by_horizon': mean_h,
        'per_pair': horizons,
    }

    # --- 4. action semantics ---
    semantics = []
    low = np.asarray(session.action_low, dtype=np.float64)
    high = np.asarray(session.action_high, dtype=np.float64)
    for p_i, pair in enumerate(pairs):
        ep, start = int(pair['episode']), int(pair['start'])
        episode = episode_of(ep)
        native = _native(index, ep, start, block)
        plan = torch.tensor(_blocks(np.concatenate([native, _native(index, ep, start + block, (horizon - 1) * block)]), block))
        executed = plan[:1].reshape(-1, plan.shape[-1]).numpy()
        session.restore(episode, start)
        caught = []

        def _capture(action, _caught=caught):
            _caught.append(np.asarray(action, dtype=np.float64).copy())
            session.step(action)

        for action in executed:
            _capture(np.asarray(action, dtype=np.float64))
        got = session.state()
        expect = _row(episode, start + block, 'state')
        normed = model.action_norm[env](torch.as_tensor(native, device=device)).cpu().numpy()
        semantics.append(
            {
                'pair': p_i,
                'executed_matches_native_max_abs': float(np.max(np.abs(np.stack(caught) - native))),
                'plan_slice_matches_first_block': bool(np.allclose(executed, plan[0].numpy())),
                'state_l2_after_5': float(np.linalg.norm(got - expect)),
                'native_outside_bounds': int(np.sum((native < low) | (native > high))),
                'native_vs_normalized_mae': float(np.mean(np.abs(native - normed))),
                'within_block_action_std': float(native.std(axis=0).mean()),
                'repeated_action': bool(np.allclose(native, native[:1])),
            }
        )
    report['action_semantics'] = {
        'relative_control': True,
        'bounds': {'low': low.tolist(), 'high': high.tolist()},
        'pairs': semantics,
        'mean_state_l2_after_5': float(np.mean([s['state_l2_after_5'] for s in semantics])),
        'any_executed_mismatch': any(s['executed_matches_native_max_abs'] > 1e-6 for s in semantics),
        'any_repeated_block': any(s['repeated_action'] for s in semantics),
        'actions_outside_bounds': int(sum(s['native_outside_bounds'] for s in semantics)),
    }

    # --- 5. goal encoding ---
    goals = []
    emb = model.env_embed[env]
    for p_i, pair in enumerate(pairs):
        ep, start, goal = int(pair['episode']), int(pair['start']), int(pair['goal'])
        episode = episode_of(ep)
        session.restore(episode, goal)
        at_goal = _frame_pair(episode, session, goal, cfg, device, model)
        session.restore(episode, start)
        at_start = _frame_pair(episode, session, start, cfg, device, model)
        z_goal = at_goal['z_data']
        old = emb.detach().clone()
        with torch.no_grad():
            emb.add_(1)
            z_shifted = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
            emb.copy_(old)
        goals.append(
            {
                'pair': p_i,
                'goal_pixel_mae': at_goal['pixel_mae'],
                'goal_latent_mse': at_goal['latent_mse'],
                'start_pixel_mae': at_start['pixel_mae'],
                'start_latent_mse': at_start['latent_mse'],
                'encode_changes_if_env_embed_changes': _latent_mse(z_goal, z_shifted) > 1e-8,
            }
        )
    report['goal_encoding'] = {
        'goal_latent_comes_from': 'dataset frame via encode(); cost is latent MSE only',
        'mean_goal_pixel_mae': float(np.mean([g['goal_pixel_mae'] for g in goals])),
        'mean_goal_latent_mse': float(np.mean([g['goal_latent_mse'] for g in goals])),
        'mean_start_pixel_mae': float(np.mean([g['start_pixel_mae'] for g in goals])),
        'mean_start_latent_mse': float(np.mean([g['start_latent_mse'] for g in goals])),
        'env_embed_leaks_into_encode': any(g['encode_changes_if_env_embed_changes'] for g in goals),
        'pairs': [{k: v for k, v in g.items()} for g in goals],
    }

    # --- 6. closed-loop CEM trace ---
    bar = Progress(len(pairs), 0, f'CEM trace | PushT', log)
    traces = []
    for p_i, pair in enumerate(pairs):
        ep, start, goal = int(pair['episode']), int(pair['start']), int(pair['goal'])
        episode = episode_of(ep)
        session.restore(episode, start)
        session.set_goal(episode, goal)
        planner = CEMPlanner(
            model, env, cfg, session.action_low, session.action_high, device, seed=3000 + p_i
        )
        z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
        z_hist = [_encode_frame(model, session.render(), cfg, device)]
        a_hist: list[torch.Tensor] = []
        steps, warm = 0, None
        replans = []
        budget = int(cfg['eval']['exec_budget'])
        while steps < budget and not session.success():
            h = min(len(z_hist), int(cfg['cem']['history']))
            zh = torch.stack(z_hist[-h:], dim=1)
            if h > 1:
                ah = torch.stack(a_hist[-(h - 1) :], dim=0)[None]
            else:
                ah = torch.zeros((1, 0, block, planner.low.shape[0]), device=device)
            before = _pusht_goal_distance(session)['pos']
            plan, info = planner.plan(zh, ah, z_goal, warm)
            block_np = plan[0].detach().cpu().numpy()
            for action in block_np:
                session.step(np.asarray(action, dtype=np.float64))
                steps += 1
                if session.success() or steps >= budget:
                    break
            after = _pusht_goal_distance(session)['pos']
            live_cost = _latent_mse(_encode_frame(model, session.render(), cfg, device), z_goal)
            replans.append(
                {
                    'replan': len(replans),
                    'steps_after': steps,
                    'pred_elite_cost': info['elite_cost_last'],
                    'pred_best_cost': info['best_cost_last'],
                    'goal_pos_before': before,
                    'goal_pos_after': after,
                    'actual_latent_cost_after': live_cost,
                    'action_abs_mean': float(np.abs(block_np).mean()),
                }
            )
            if session.success() or steps >= budget:
                break
            z_hist.append(_encode_frame(model, session.render(), cfg, device))
            a_hist.append(plan[0])
            warm = plan[1:] if cfg['cem']['warm_start'] else None
        split_at = None
        for row in replans:
            if row['goal_pos_after'] > row['goal_pos_before'] * 1.25 and row['pred_elite_cost'] <= replans[0]['pred_elite_cost']:
                split_at = row['replan']
                break
        traces.append(
            {
                'pair': p_i,
                'episode': ep,
                'start': start,
                'goal_pos_initial': replans[0]['goal_pos_before'],
                'goal_pos_final': replans[-1]['goal_pos_after'],
                'elite_cost_first': replans[0]['pred_elite_cost'],
                'elite_cost_last': replans[-1]['pred_elite_cost'],
                'first_replan_distance_up_while_cost_not': split_at,
                'replans': replans,
            }
        )
        bar.set_postfix(
            f'pair {p_i + 1}/{len(pairs)} | cost {replans[0]["pred_elite_cost"]:.3f}'
            f'->{replans[-1]["pred_elite_cost"]:.3f} | '
            f'dist {replans[0]["goal_pos_before"]:.0f}->{replans[-1]["goal_pos_after"]:.0f}'
        )
        bar.update(p_i + 1)
    bar.close()
    report['cem_trace'] = {
        'pairs': traces,
        'mean_elite_cost_first': float(np.mean([t['elite_cost_first'] for t in traces])),
        'mean_elite_cost_last': float(np.mean([t['elite_cost_last'] for t in traces])),
        'mean_goal_pos_initial': float(np.mean([t['goal_pos_initial'] for t in traces])),
        'mean_goal_pos_final': float(np.mean([t['goal_pos_final'] for t in traces])),
    }

    session.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    pooled = report['correlation']['pooled']
    print('--- correlation (pred latent cost vs goal distance) ---', flush=True)
    print(
        f"pearson {pooled['pearson_pred_cost_vs_goal_pos']:.3f} | "
        f"spearman {pooled['spearman_pred_cost_vs_goal_pos']:.3f}",
        flush=True,
    )
    print(
        f"actual latent vs distance spearman {pooled['spearman_actual_latent_vs_goal_pos']:.3f} | "
        f"pred vs actual latent spearman {pooled['spearman_pred_vs_actual_latent']:.3f}",
        flush=True,
    )
    print('--- exploitation (mean goal distance after the sequence) ---', flush=True)
    for name in ('cem', 'random', 'dataset'):
        rows = [r for r in report['exploitation'] if r['which'] == name]
        print(
            f"{name}: pred_cost {np.mean([r['pred_cost'] for r in rows]):.3f} | "
            f"goal_dist {np.mean([r['goal_pos_after'] for r in rows]):.1f} | "
            f"rollout_err {np.mean([r['rollout_error'] for r in rows]):.3f}",
            flush=True,
        )
    print('--- rollout error by horizon (blocks) ---', flush=True)
    for row in report['rollout']['mean_by_horizon']:
        print(
            f"h{row['blocks']}: live {row['model_from_live_vs_live']:.4f} | "
            f"dataset {row['model_from_dataset_vs_dataset']:.4f} | "
            f"render gap {row['live_vs_dataset']:.4f}",
            flush=True,
        )
    sem = report['action_semantics']
    print(
        f"--- actions: mismatch={sem['any_executed_mismatch']} "
        f"repeat={sem['any_repeated_block']} outside_bounds={sem['actions_outside_bounds']} "
        f"state_l2_after_5={sem['mean_state_l2_after_5']:.4f} ---",
        flush=True,
    )
    g = report['goal_encoding']
    print(
        f"--- goal encode: pixel_mae {g['mean_goal_pixel_mae']:.2f} "
        f"latent_mse {g['mean_goal_latent_mse']:.4f} "
        f"env_embed_leaks={g['env_embed_leaks_into_encode']} ---",
        flush=True,
    )
    tr = report['cem_trace']
    print(
        f"--- trace: elite cost {tr['mean_elite_cost_first']:.3f}->{tr['mean_elite_cost_last']:.3f} "
        f"goal dist {tr['mean_goal_pos_initial']:.1f}->{tr['mean_goal_pos_final']:.1f} ---",
        flush=True,
    )
