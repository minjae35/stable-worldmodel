"""LeWM PushT parity on the RD-WM PushT evaluator.

Loads a LeWM checkpoint through the existing Stable-WM loader and scores it
on the same fixed val pairs, the same one-block candidate draw, and the same
PushT success predicate. LeWM modules are not modified. The adapter only
normalizes native actions the way LeWM training does (per-dimension z-score,
then flatten the 5-step block) before calling ``LeWM.rollout``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.cost_diag import (
    _load_pairs,
    _mean_metric,
    _rank_metrics,
    _sample_n,
)
from rdwm.exp1.data import EnvIndex, preprocess
from rdwm.exp1.horizon_diag import _cfg_horizon
from rdwm.exp1.oracle_diag import _elite_update, _sample
from rdwm.exp1.oracle_pool import PushTPool
from rdwm.exp1.planning import (
    CEMPlanner,
    _pusht_goal_distance,
    open_episode_reader,
    open_eval_session,
)
from rdwm.exp1.progress import Progress
from rdwm.envs.sessions import dataset_frame

LEWM_REPO = 'quentinll/lewm-pusht'
RDWM_RANK = Path('/workspace/rdwm_runs/exp1/logs/diag_pusht/cost_rank_5k.json')
RDWM_ORACLE = Path('/workspace/rdwm_runs/exp1/logs/diag_pusht/oracle_diag_5k.json')


def load_lewm(device: torch.device, checkpoint: str | None = None):
    """Load LeWM weights. ``checkpoint`` is a weights file or the public repo id."""
    from stable_worldmodel.wm.utils import load_pretrained

    model = load_pretrained(checkpoint or LEWM_REPO)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def action_norm_from_index(cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full-table z-score, matching LeWM's column normalizer before the split."""
    index = EnvIndex('pusht', env_path(cfg, 'pusht'), cfg['paths']['cache_root'])
    acts = np.asarray(index.actions, dtype=np.float64)
    finite = np.isfinite(acts).all(axis=1)
    kept = acts[finite]
    mean = kept.mean(axis=0)
    std = np.maximum(kept.std(axis=0), 1e-8)
    span = 4 * 5
    lengths = np.asarray(index.lengths)
    n_clips = int(np.maximum(lengths - span + 1, 0).sum())
    steps_per_epoch = n_clips * 9 // 10 // 128
    meta = {
        'episodes': int(index.num_episodes),
        'finite_actions': int(finite.sum()),
        'clips_span20': n_clips,
        'lewm_updates_100_epochs_batch128_split09': int(steps_per_epoch * 100),
        'action_mean': mean.astype(np.float64).tolist(),
        'action_std': std.astype(np.float64).tolist(),
    }
    return mean.astype(np.float32), std.astype(np.float32), meta


def _frames_to_pixels(frames, cfg, device) -> torch.Tensor:
    arrays = [np.ascontiguousarray(frame) for frame in frames]
    stacked = np.stack(arrays, axis=0)
    pixels = torch.from_numpy(stacked).permute(0, 3, 1, 2)
    return preprocess(pixels[:, None].to(device), cfg)[:, 0]


@torch.no_grad()
def encode_frames(model, frames, cfg, device, chunk: int = 64) -> torch.Tensor:
    """LeWM CLS embedding after the projector. Returns [N, D] on CPU."""
    out = []
    for start in range(0, len(frames), chunk):
        pixels = _frames_to_pixels(frames[start : start + chunk], cfg, device)
        info = model.encode({'pixels': pixels[:, None]})
        out.append(info['emb'][:, 0].detach().float().cpu())
    return torch.cat(out, dim=0)


def _flatten_norm(native: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[..., 5, 2] native actions -> [..., 10] z-scored block."""
    z = (native.float() - mean) / std
    return z.reshape(*native.shape[:-2], -1)


@torch.no_grad()
def predicted_cost(
    model,
    ctx: torch.Tensor,
    past: torch.Tensor,
    cand: torch.Tensor,
    goal: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error of the last predicted CLS vs the goal CLS.

    ``cand`` is [S, 1, 5, 2] in native actions. Sum reduction ranks the same
    single vector, so the elite set matches LeWM GoalMSE.
    """
    samples = cand.shape[0]
    history = ctx.shape[0]
    emb = ctx[None, None].expand(1, samples, history, -1).contiguous()
    seq = _flatten_norm(cand, mean, std).reshape(1, samples, 1, -1)
    if history > 1:
        act_past = past[None, None].expand(1, samples, history - 1, -1).contiguous()
    else:
        act_past = seq.new_zeros(1, samples, 0, seq.shape[-1])
    info = {
        'pixels': seq.new_zeros(1, samples, history, 1),
        'emb': emb,
        'action_history': act_past,
    }
    pred = model.rollout(info, seq)['predicted_emb']
    return (pred[:, :, -1].float() - goal.float()).square().mean(dim=-1)[0]


def _plan_block(model, planner, ctx, past, goal, mean, std) -> torch.Tensor:
    shape = (1, planner.block, planner.low.shape[0])
    dist_mean = planner.center.expand(shape).clone()
    dist_std = planner.init_std.expand(shape).clone()
    for _ in range(planner.iters):
        cand = _sample(planner, dist_mean, dist_std)
        cost = predicted_cost(model, ctx, past, cand, goal, mean, std)
        dist_mean, dist_std, _ = _elite_update(cand, cost, planner.topk)
    return planner.clamp(dist_mean)


def _reference() -> dict:
    ref = {}
    if RDWM_RANK.exists():
        rank = json.loads(RDWM_RANK.read_text())
        mean = rank['costs']['mean']
        ref['rdwm_5k_rank_mean'] = {
            key: mean[key] for key in mean if key != 'pairs'
        }
    if RDWM_ORACLE.exists():
        oracle = json.loads(RDWM_ORACLE.read_text())
        ref['rdwm_5k_oracle_A'] = {
            key: oracle['A'][key]
            for key in (
                'success_count',
                'goal_pos_initial',
                'goal_pos_final',
                'goal_pos_reduction_pct',
                'closer_count',
            )
        }
    return ref


def diagnose(
    cfg: dict,
    out: Path,
    device: torch.device,
    workers: int = 32,
    n_candidates: int = 512,
    n_pairs: int = 10,
    pair_offset: int = 0,
    seed0: int = 1000,
    checkpoint: str | None = None,
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
) -> dict:
    torch.set_grad_enabled(False)
    model = load_lewm(device, checkpoint)
    if action_mean is None or action_std is None:
        mean_np, std_np, exposure = action_norm_from_index(cfg)
    else:
        mean_np = np.asarray(action_mean, dtype=np.float32)
        std_np = np.asarray(action_std, dtype=np.float32)
        exposure = {
            'source': 'training run, train-split action stats',
            'action_mean': mean_np.tolist(),
            'action_std': std_np.tolist(),
        }
    mean = torch.as_tensor(mean_np, device=device)
    std = torch.as_tensor(std_np, device=device)
    hcfg = _cfg_horizon(cfg, 1)
    session = open_eval_session('pusht')
    reader = open_episode_reader(env_path(cfg, 'pusht'))
    pairs = _load_pairs(cfg, session, reader, pair_offset, n_pairs)
    sampler = CEMPlanner(
        model, 'pusht', hcfg, session.action_low, session.action_high, device, seed0
    )
    log = out.parent / 'lewm_diag_progress.log'
    pool = PushTPool(workers)
    rank_rows = []
    plan_rows = []
    rank_total = n_pairs * n_candidates
    bar = Progress(rank_total, 0, 'LeWM rank | PushT', log, unit='cand/s')
    try:
        done = 0
        for i, pair in enumerate(pairs):
            episode = reader.load_episode(int(pair['episode']))
            start, goal_at = int(pair['start']), int(pair['goal'])
            session.restore(episode, start)
            session.set_goal(episode, goal_at)
            initial = _pusht_goal_distance(session)['pos']
            goal_frame = dataset_frame(episode, goal_at)
            z_goal = encode_frames(model, [goal_frame], cfg, device)
            cand = _sample_n(sampler, n_candidates, seed0 + pair_offset + i)
            prime = _prime_of(session, episode, start)
            bar.desc = f'LeWM rank | PushT | pair {i + 1}/{n_pairs} | stage rank'
            from rdwm.exp1.oracle_diag import _sim_outcomes

            frames, dists = _sim_outcomes(
                session, cand, render=True, pool=pool, prime=prime
            )
            z = encode_frames(model, list(frames), cfg, device)
            cost = (z.float() - z_goal.float()).square().mean(dim=-1).numpy()
            metrics = _rank_metrics(cost, dists, initial)
            metrics.update({
                'episode': int(pair['episode']),
                'start': start,
                'goal': goal_at,
                'initial_distance': initial,
            })
            rank_rows.append(metrics)
            done += len(dists)
            bar.set_postfix(
                f"pair {i + 1}/{n_pairs} | stage rank | "
                f"spearman {metrics['spearman']:.3f} | "
                f"best pct {metrics['best_distance_percentile']:.1f}"
            )
            bar.update(done)
        bar.close()

        plan_total = (
            n_pairs
            * (int(cfg['eval']['exec_budget']) // int(cfg['model']['action_block']))
            * int(cfg['cem']['n_steps'])
            * int(cfg['cem']['num_samples'])
        )
        bar = Progress(plan_total, 0, 'LeWM plan | PushT', log, unit='cand/s')
        scored = 0
        for i, pair in enumerate(pairs):
            episode = reader.load_episode(int(pair['episode']))
            planner = CEMPlanner(
                model,
                'pusht',
                hcfg,
                session.action_low,
                session.action_high,
                device,
                seed0 + i,
            )
            row, used = _closed_loop(
                model, session, episode, pair, hcfg, device, planner, mean, std, cfg
            )
            plan_rows.append(row)
            scored += used
            bar.desc = f'LeWM plan | PushT | pair {i + 1}/{n_pairs} | stage plan'
            bar.set_postfix(
                f"pair {i + 1}/{n_pairs} | stage plan | "
                f"success {int(row['success'])} | "
                f"dist {row['goal_pos_initial']:.0f}->{row['goal_pos_final']:.0f}"
            )
            bar.update(min(plan_total, scored))
        bar.close()
    finally:
        pool.close()
        session.close()

    report = {
        'checkpoint': checkpoint or LEWM_REPO,
        'note': (
            'huggingface.co/lewm/pusht does not exist. '
            'quentinll/lewm-pusht is the public checkpoint whose config matches '
            'scripts/train/config/lewm.yaml and whose card calls it the official PushT LeWM.'
        ),
        'protocol': {
            'pairs': n_pairs,
            'pair_offset': pair_offset,
            'candidates': n_candidates,
            'seed0': seed0,
            'horizon_blocks': 1,
            'receding_blocks': 1,
            'action_block': int(cfg['model']['action_block']),
            'history': int(cfg['cem']['history']),
            'cem': {
                'num_samples': int(cfg['cem']['num_samples']),
                'n_steps': int(cfg['cem']['n_steps']),
                'topk': int(cfg['cem']['topk']),
            },
            'rank_cost': 'mean squared error of the encoded outcome CLS vs the goal CLS',
            'plan_cost': 'mean squared error of the last predicted CLS vs the goal CLS',
            'action_norm': 'full-table per-dim z-score, then flatten 5 native actions',
        },
        'data_exposure': exposure,
        'rank_summary': {
            'mean_spearman': _mean_metric(rank_rows, 'spearman'),
            'mean_best_distance_percentile': _mean_metric(rank_rows, 'best_distance_percentile'),
            'mean_top5_distance_percentile': _mean_metric(rank_rows, 'top5_distance_percentile'),
            'mean_closer_rank_percentile': _mean_metric(rank_rows, 'closer_mean_rank_percentile'),
            'mean_n_closer': _mean_metric(rank_rows, 'n_closer'),
        },
        'rank_pairs': rank_rows,
        'plan_summary': {
            'success_count': int(sum(row['success'] for row in plan_rows)),
            'goal_pos_initial': float(np.mean([row['goal_pos_initial'] for row in plan_rows])),
            'goal_pos_final': float(np.mean([row['goal_pos_final'] for row in plan_rows])),
            'goal_pos_reduction_pct': float(np.mean([row['goal_pos_reduction_pct'] for row in plan_rows])),
            'closer_count': int(sum(row['closer'] for row in plan_rows)),
        },
        'plan_pairs': plan_rows,
        'rdwm_5k_reference': _reference(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _print(report)
    out.write_text(json.dumps(report, indent=2))
    return report


def _prime_of(session, episode, start: int) -> dict:
    from rdwm.exp1.oracle_diag import _decision_prime

    return _decision_prime(session, episode, start)


def _closed_loop(model, session, episode, pair, cfg, device, planner, mean, std, full_cfg) -> tuple[dict, int]:
    start, goal_at = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal_at)
    initial = _pusht_goal_distance(session)['pos']
    goal = encode_frames(model, [dataset_frame(episode, goal_at)], full_cfg, device).to(device)
    hist = [encode_frames(model, [session.render()], full_cfg, device).to(device)[0]]
    past: list[torch.Tensor] = []
    steps, success, used = 0, False, 0
    budget = int(cfg['eval']['exec_budget'])
    history = int(cfg['cem']['history'])
    per_replan = int(cfg['cem']['n_steps']) * int(cfg['cem']['num_samples'])
    while steps < budget and not success:
        h = min(len(hist), history)
        ctx = torch.stack(hist[-h:], dim=0)
        if h > 1:
            past_z = torch.stack(past[-(h - 1):], dim=0)
        else:
            past_z = ctx.new_zeros((0, planner.block * planner.low.shape[0]))
        plan = _plan_block(model, planner, ctx, past_z, goal, mean, std)
        used += per_replan
        native = plan[0].detach().cpu().numpy()
        for action in native:
            session.step(np.asarray(action, dtype=np.float64))
            steps += 1
            if session.success():
                success = True
                break
            if steps >= budget:
                break
        if success or steps >= budget:
            break
        hist.append(encode_frames(model, [session.render()], full_cfg, device).to(device)[0])
        past.append(_flatten_norm(plan[0], mean, std))
    final = _pusht_goal_distance(session)['pos']
    reduction = (initial - final) / initial * 100.0 if initial > 0 else None
    return {
        'episode': int(pair['episode']),
        'start': start,
        'goal': goal_at,
        'success': bool(success),
        'goal_pos_initial': initial,
        'goal_pos_final': final,
        'goal_pos_reduction_pct': reduction,
        'closer': bool(final < initial),
        'steps': steps,
    }, used


def _print(report: dict) -> None:
    rank = report['rank_summary']
    plan = report['plan_summary']
    ref = report['rdwm_5k_reference']
    print('--- LeWM vs RD-WM 5k, horizon 1 block, 10 val pairs ---', flush=True)
    print(
        f"LeWM rank | spearman {rank['mean_spearman']:.3f} | "
        f"best pct {rank['mean_best_distance_percentile']:.2f} | "
        f"closer rank {rank['mean_closer_rank_percentile']:.2f} | "
        f"n_closer {rank['mean_n_closer']:.1f}",
        flush=True,
    )
    if 'rdwm_5k_rank_mean' in ref:
        base = ref['rdwm_5k_rank_mean']
        print(
            f"RD-WM mean | spearman {base['mean_spearman']:.3f} | "
            f"best pct {base['mean_best_distance_percentile']:.2f} | "
            f"closer rank {base['mean_closer_rank_percentile']:.2f} | "
            f"n_closer {base['mean_n_closer']:.1f}",
            flush=True,
        )
    print(
        f"LeWM plan | success {plan['success_count']}/10 | "
        f"dist {plan['goal_pos_initial']:.1f}->{plan['goal_pos_final']:.1f} | "
        f"reduction {plan['goal_pos_reduction_pct']:.1f}% | "
        f"closer {plan['closer_count']}/10",
        flush=True,
    )
    if 'rdwm_5k_oracle_A' in ref:
        base = ref['rdwm_5k_oracle_A']
        print(
            f"RD-WM A | success {base['success_count']}/10 | "
            f"dist {base['goal_pos_initial']:.1f}->{base['goal_pos_final']:.1f} | "
            f"reduction {base['goal_pos_reduction_pct']:.1f}% | "
            f"closer {base['closer_count']}/10",
            flush=True,
        )
