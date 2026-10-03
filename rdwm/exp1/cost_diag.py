"""PushT planning-cost diagnostic on one fixed checkpoint.

The model, dynamics, and training loss stay as trained. Only the visual
latent cost changes. State and true goal distance are evaluation labels,
never part of a cost. Every cost on a pair reads the same simulator rollout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.evaluate import get_pairs
from rdwm.exp1.horizon_diag import _cfg_horizon, _encode_frames
from rdwm.exp1.oracle_diag import (
    _bank,
    _decision_prime,
    _elite_update,
    _sample,
    _sim_outcomes,
)
from rdwm.exp1.oracle_pool import PushTPool
from rdwm.exp1.plan_diag import _spearman
from rdwm.exp1.planning import (
    CEMPlanner,
    _encode_frame,
    _pusht_goal_distance,
    open_episode_reader,
    open_eval_session,
)
from rdwm.exp1.progress import Progress
from rdwm.envs.sessions import dataset_frame

COST_NAMES = ('mean', 'weighted', 'top4', 'top8', 'top16')


def _token_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token L2. a and b broadcast to [..., tokens, dim]."""
    return (a.float() - b.float()).square().sum(dim=-1).sqrt()


def _token_mse(z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """[N, tokens] mean squared error against the goal latent."""
    return (z.float() - z_goal.float()).square().mean(dim=-1)


def _change(z_start: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """[tokens] start-to-goal change. Computed once per planning problem."""
    return _token_l2(z_start, z_goal).reshape(-1)


def _spec(name: str, z_start: torch.Tensor, z_goal: torch.Tensor) -> dict:
    delta = _change(z_start, z_goal)
    if name == 'mean':
        return {'name': name, 'kind': 'mean'}
    if name == 'weighted':
        total = float(delta.sum())
        if total < 1e-8:
            weight = torch.ones_like(delta) / delta.numel()
        else:
            weight = delta / delta.sum()
        return {'name': name, 'kind': 'weighted', 'weight': weight}
    if name.startswith('top'):
        k = int(name[3:])
        if k > delta.numel():
            raise ValueError(f'{name} needs {k} tokens, latent has {delta.numel()}')
        index = torch.topk(delta, k, largest=True).indices
        return {'name': name, 'kind': 'topk', 'k': k, 'index': index}
    raise ValueError(name)


def score_latents(z: torch.Tensor, z_goal: torch.Tensor, spec: dict) -> torch.Tensor:
    """Latent cost [N]. z is [N, tokens, dim], z_goal is [1, tokens, dim] or [tokens, dim]."""
    err = _token_mse(z, z_goal)
    kind = spec['kind']
    if kind == 'mean':
        return err.mean(dim=-1)
    if kind == 'weighted':
        return (err * spec['weight'].to(err.device)).sum(dim=-1)
    return err[:, spec['index'].to(err.device)].mean(dim=-1)


def _encode_batch(bank, frames, cfg) -> torch.Tensor:
    frames = np.asarray(frames)
    splits = np.array_split(np.arange(len(frames)), len(bank))
    pending = []
    for model, idx in zip(bank, splits):
        if len(idx) == 0:
            continue
        device = next(model.parameters()).device
        chunk = [np.ascontiguousarray(frames[int(i)]) for i in idx]
        pending.append((idx, _encode_frames(model, chunk, cfg, device)))
    out = None
    for idx, z in pending:
        cpu = z.detach().float().cpu()
        if out is None:
            out = torch.empty((len(frames),) + tuple(cpu.shape[1:]), dtype=torch.float32)
        out[torch.as_tensor(np.asarray(idx), dtype=torch.long)] = cpu
    return out


def _sample_n(planner: CEMPlanner, n: int, seed: int) -> torch.Tensor:
    """One-block candidates from the CEM initial distribution."""
    gen = torch.Generator(device=planner.device).manual_seed(int(seed))
    shape = (1, planner.block, planner.low.shape[0])
    noise = torch.randn((n, *shape), generator=gen, device=planner.device)
    mean = planner.center.expand(shape)
    cand = planner.clamp(noise * planner.init_std.expand(shape) + mean)
    cand[0] = mean
    return cand


def _distance_percentile(dist: np.ndarray, index: int) -> float:
    """0 means this candidate has the smallest true goal distance."""
    return 100.0 * float(np.sum(dist < dist[index])) / max(1, len(dist) - 1)


def _rank_metrics(cost: np.ndarray, dist: np.ndarray, initial: float) -> dict:
    order = np.argsort(cost, kind='mergesort')
    rank = np.empty(len(cost), dtype=np.float64)
    rank[order] = np.arange(len(cost))
    best = int(order[0])
    top5 = order[:5]
    closer = dist < initial
    closer_rank = None
    closer_median = None
    if int(closer.sum()) > 0:
        pct = 100.0 * rank[closer] / max(1, len(cost) - 1)
        closer_rank = float(pct.mean())
        closer_median = float(np.median(pct))
    return {
        'spearman': _spearman(cost, dist),
        'best_distance_percentile': _distance_percentile(dist, best),
        'best_distance': float(dist[best]),
        'top5_distance_percentile': float(np.mean([_distance_percentile(dist, int(i)) for i in top5])),
        'n_closer': int(closer.sum()),
        'closer_mean_rank_percentile': closer_rank,
        'closer_median_rank_percentile': closer_median,
    }


def _mean_metric(rows: list[dict], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row[key] is not None]
    return float(np.mean(vals)) if vals else None


def _load_pairs(cfg, session, reader, offset: int, n_pairs: int) -> list[dict]:
    pairs = get_pairs(cfg, 'pusht', 'val', cfg['eval']['pilot_val_pairs'], session, reader)
    chosen = pairs[offset : offset + n_pairs]
    if len(chosen) < n_pairs:
        raise RuntimeError(f'needed {n_pairs} pairs from offset {offset}, found {len(chosen)}')
    return chosen


def rank_diagnose(
    cfg: dict,
    ckpt: Path,
    out: Path,
    device,
    workers: int = 32,
    n_candidates: int = 512,
    n_pairs: int = 10,
    pair_offset: int = 0,
    seed0: int = 1000,
) -> dict:
    torch.set_grad_enabled(False)
    bank = _bank(cfg, ckpt)
    model = bank[0]
    hcfg = _cfg_horizon(cfg, 1)
    session = open_eval_session('pusht')
    reader = open_episode_reader(env_path(cfg, 'pusht'))
    pairs = _load_pairs(cfg, session, reader, pair_offset, n_pairs)
    log = out.parent / 'cost_rank_progress.log'
    pool = PushTPool(workers)
    rows = {name: [] for name in COST_NAMES}
    total = n_pairs * n_candidates
    bar = Progress(total, 0, 'Cost rank | PushT', log, unit='cand/s')
    try:
        done = 0
        for i, pair in enumerate(pairs):
            episode = reader.load_episode(int(pair['episode']))
            start, goal = int(pair['start']), int(pair['goal'])
            session.restore(episode, start)
            session.set_goal(episode, goal)
            initial = _pusht_goal_distance(session)['pos']
            planner = CEMPlanner(
                model, 'pusht', hcfg, session.action_low, session.action_high, device, seed0 + i
            )
            z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device).cpu()
            z_start = _encode_frame(model, session.render(), cfg, device).cpu()
            if int(z_goal.shape[-2]) != 64:
                raise RuntimeError(f'expected 64 spatial tokens, got {tuple(z_goal.shape)}')
            cand = _sample_n(planner, n_candidates, seed0 + pair_offset + i)
            prime = _decision_prime(session, episode, start)
            bar.desc = f'Cost rank | PushT | pair {i + 1}/{n_pairs} | stage rank'
            frames, dists = _sim_outcomes(session, cand, render=True, pool=pool, prime=prime)
            z = _encode_batch(bank, frames, cfg)
            specs = [_spec(name, z_start, z_goal) for name in COST_NAMES]
            for spec in specs:
                cost = score_latents(z, z_goal, spec).numpy()
                metrics = _rank_metrics(cost, dists, initial)
                metrics.update({
                    'episode': int(pair['episode']),
                    'start': start,
                    'goal': goal,
                    'initial_distance': initial,
                })
                rows[spec['name']].append(metrics)
            done += len(dists)
            bar.set_postfix(f'pair {i + 1}/{n_pairs} | stage rank')
            bar.update(done)
        bar.close()
    finally:
        pool.close()
        session.close()

    summary = {}
    for name, pair_rows in rows.items():
        summary[name] = {
            'mean_spearman': _mean_metric(pair_rows, 'spearman'),
            'mean_best_distance_percentile': _mean_metric(pair_rows, 'best_distance_percentile'),
            'mean_top5_distance_percentile': _mean_metric(pair_rows, 'top5_distance_percentile'),
            'mean_closer_rank_percentile': _mean_metric(pair_rows, 'closer_mean_rank_percentile'),
            'mean_closer_median_rank_percentile': _mean_metric(pair_rows, 'closer_median_rank_percentile'),
            'mean_n_closer': _mean_metric(pair_rows, 'n_closer'),
            'pairs': pair_rows,
        }
    report = {
        'ckpt': str(ckpt),
        'n_candidates': n_candidates,
        'n_pairs': n_pairs,
        'pair_offset': pair_offset,
        'seed0': seed0,
        'workers': workers,
        'note': (
            'One shared rollout per pair. mean is the average latent MSE over '
            '64 tokens. weighted uses start-to-goal per-token L2 as fixed '
            'weights. topk averages MSE on the k tokens with the largest '
            'start-to-goal L2. Distance is not inside the cost.'
        ),
        'costs': summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    _print_rank(report)
    return report


def _print_rank(report: dict) -> None:
    print(
        f"--- cost rank, {report['n_candidates']} candidates, "
        f"pairs {report['pair_offset']}:{report['pair_offset'] + report['n_pairs']} ---",
        flush=True,
    )
    print(
        'cost | spearman | best dist pct | top5 dist pct | closer rank pct | n_closer',
        flush=True,
    )
    for name, row in report['costs'].items():
        print(
            f"{name:8} | {_fmt(row['mean_spearman'])} | "
            f"{_fmt(row['mean_best_distance_percentile'])} | "
            f"{_fmt(row['mean_top5_distance_percentile'])} | "
            f"{_fmt(row['mean_closer_rank_percentile'])} | "
            f"{_fmt(row['mean_n_closer'])}",
            flush=True,
        )


def _fmt(value) -> str:
    if value is None:
        return '  nan'
    return f'{value:6.2f}'


def _plan_model(planner: CEMPlanner, zh, ah, z_goal, spec: dict) -> torch.Tensor:
    """CEM loop identical to CEMPlanner.plan, with a selectable latent cost."""
    shape = (planner.horizon, planner.block, planner.low.shape[0])
    mean = planner.center.expand(shape).clone()
    std = planner.init_std.expand(shape).clone()
    z = zh.expand(planner.samples, *zh.shape[1:])
    a_hist = ah.expand(planner.samples, *ah.shape[1:])
    goal = z_goal.float()
    for _ in range(planner.iters):
        cand = _sample(planner, mean, std)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=planner.bf16):
            preds = planner.model.rollout(z, a_hist, cand, planner.env)
        cost = score_latents(preds[:, -1].float(), goal, spec)
        mean, std, _ = _elite_update(cand, cost, planner.topk)
    return planner.clamp(mean)


def _closed_model(model, session, episode, pair, cfg, device, seed, spec, on_iter) -> dict:
    from rdwm.exp1.horizon_diag import _closed_loop

    if spec['kind'] == 'mean':
        return _closed_loop(model, session, episode, pair, cfg, device, seed)

    ecfg, ccfg = cfg['eval'], cfg['cem']
    block = int(cfg['model']['action_block'])
    start, goal = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal)
    initial = _pusht_goal_distance(session)['pos']
    planner = CEMPlanner(
        model, 'pusht', cfg, session.action_low, session.action_high, device, seed
    )
    z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
    z_start = _encode_frame(model, session.render(), cfg, device)
    fixed = _spec(spec['name'], z_start.cpu(), z_goal.cpu())
    for key in ('weight', 'index'):
        if key in fixed:
            fixed[key] = fixed[key].to(device)
    z_hist = [z_start]
    a_hist: list[torch.Tensor] = []
    steps, success = 0, False
    budget = int(ecfg['exec_budget'])
    while steps < budget and not success:
        h = min(len(z_hist), int(ccfg['history']))
        zh = torch.stack(z_hist[-h:], dim=1)
        if h > 1:
            ah = torch.stack(a_hist[-(h - 1):], dim=0)[None]
        else:
            ah = torch.zeros((1, 0, block, planner.low.shape[0]), device=device)
        plan = _plan_model(planner, zh, ah, z_goal, fixed)
        on_iter(planner.samples * planner.iters)
        native = plan[:1].reshape(-1, plan.shape[-1]).detach().cpu().numpy()
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
        z_hist.append(_encode_frame(model, session.render(), cfg, device))
        a_hist.append(plan[0])
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
    }


def _closed_sim(model, session, episode, pair, cfg, device, seed, spec, on_iter, pool, bank) -> dict:
    start, goal = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal)
    initial = _pusht_goal_distance(session)['pos']
    planner = CEMPlanner(
        model, 'pusht', cfg, session.action_low, session.action_high, device, seed
    )
    z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
    z_start = _encode_frame(model, session.render(), cfg, device)
    fixed = _spec(spec['name'], z_start.cpu(), z_goal.cpu())
    shape = (1, planner.block, planner.low.shape[0])
    prime = _decision_prime(session, episode, start)
    steps, success = 0, False
    budget = int(cfg['eval']['exec_budget'])

    def cost_fn(cand):
        frames, dists = _sim_outcomes(session, cand, render=True, pool=pool, prime=prime)
        on_iter(len(dists))
        z = _encode_batch(bank, frames, cfg)
        return score_latents(z, z_goal.cpu(), fixed).to(device)

    from rdwm.exp1.oracle_diag import _oracle_plan

    while steps < budget and not success:
        mean = planner.center.expand(shape).clone()
        std = planner.init_std.expand(shape).clone()
        plan, _ = _oracle_plan(planner, mean, std, cost_fn)
        for action in plan[0].detach().cpu().numpy():
            session.step(np.asarray(action, dtype=np.float64))
            steps += 1
            if session.success():
                success = True
                break
            if steps >= budget:
                break
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
    }
