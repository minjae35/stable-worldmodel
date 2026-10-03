"""Separate PushT dynamics error from the latent planning cost.

A scores one action block with the model. B executes that block in the
simulator and scores the encoded observation. C scores the true goal
distance. Horizon is one block because that is the block receding control
executes, and H=1 already failed. The frozen CEM horizon in the config is
left unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.evaluate import get_pairs, load_model
from rdwm.exp1.horizon_diag import _cfg_horizon, _closed_loop, _encode_frames
from rdwm.exp1.oracle_pool import PushTPool, _prime
from rdwm.exp1.plan_diag import _spearman
from rdwm.exp1.planning import (
    CEMPlanner,
    _encode_frame,
    _pusht_goal_distance,
    open_episode_reader,
    open_eval_session,
)
from rdwm.exp1.progress import Progress
from rdwm.envs.sessions import _core, dataset_frame


def _corr(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _snap(session) -> dict:
    core = _core(session.env)
    return {
        'ap': tuple(core.agent.position),
        'av': tuple(core.agent.velocity),
        'bp': tuple(core.block.position),
        'ba': float(core.block.angle),
        'bv': tuple(core.block.velocity),
        'bw': float(core.block.angular_velocity),
    }


def _restore(session, snap: dict) -> None:
    core = _core(session.env)
    core.agent.position = snap['ap']
    core.agent.velocity = snap['av']
    core.block.position = snap['bp']
    core.block.angle = snap['ba']
    core.block.velocity = snap['bv']
    core.block.angular_velocity = snap['bw']


def _play_block(session, block: np.ndarray) -> None:
    for action in block:
        session.step(np.asarray(action, dtype=np.float64))


def _sample(planner: CEMPlanner, mean, std) -> torch.Tensor:
    shape = mean.shape
    noise = torch.randn((planner.samples, *shape), generator=planner.gen, device=planner.device)
    cand = planner.clamp(noise * std + mean)
    cand[0] = mean
    return cand


def _elite_update(cand: torch.Tensor, cost: torch.Tensor, topk: int):
    vals, idx = torch.topk(cost, topk, largest=False)
    elites = cand[idx]
    return elites.mean(dim=0), elites.std(dim=0, correction=0), vals


def _decision_prime(session, episode, start: int) -> dict:
    """Dataset state that session.restore uses, plus the goal already set on the env."""
    core = _core(session.env)
    pose = getattr(core, 'goal_pose', None)
    return {
        'state': np.asarray(session._state_from_row(episode, start), dtype=np.float64).copy(),
        'goal_pose': None if pose is None else np.asarray(pose, dtype=np.float64).copy(),
        'goal_state': np.asarray(core.goal_state, dtype=np.float64).copy(),
    }


def _sim_outcomes(session, cand: torch.Tensor, render: bool, pool=None, prime=None):
    """Execute each 1-block candidate from the current state.

    One rollout returns the frame and the position goal distance together.
    """
    snap = _snap(session)
    goal = np.asarray(_core(session.env).goal_state, dtype=np.float64)
    blocks = cand.detach().cpu().numpy()[:, 0]
    if pool is not None:
        return pool.rollout(prime, blocks, render)
    if prime is not None:
        goal = np.asarray(prime['goal_state'], dtype=np.float64)
        frames, dists = [], []
        for block in blocks:
            _prime(session, prime)
            for action in block:
                session.step(np.asarray(action, dtype=np.float64))
            if render:
                frames.append(np.ascontiguousarray(session.render()))
            dists.append(float(np.linalg.norm(goal[:4] - session.state()[:4])))
        packed = np.stack(frames, axis=0) if render else None
        return packed, np.asarray(dists, dtype=np.float64)
    frames, dists = [], []
    try:
        for block in blocks:
            _play_block(session, block)
            if render:
                frames.append(np.ascontiguousarray(session.render()))
            dists.append(float(np.linalg.norm(goal[:4] - session.state()[:4])))
            _restore(session, snap)
    except Exception:
        _restore(session, snap)
        raise
    packed = np.stack(frames, axis=0) if render else None
    return packed, np.asarray(dists, dtype=np.float64)


def _latent_cost(bank, frames, z_goal, cfg) -> torch.Tensor:
    """Encode frames, splitting the batch across the models' GPUs."""
    if frames is None or len(frames) == 0:
        return torch.zeros(0)
    frames = np.asarray(frames)
    splits = np.array_split(np.arange(len(frames)), len(bank))
    costs = []
    for model, idx in zip(bank, splits):
        if len(idx) == 0:
            continue
        device = next(model.parameters()).device
        chunk = [np.ascontiguousarray(frames[i]) for i in idx]
        z = _encode_frames(model, chunk, cfg, device)
        zg = z_goal.to(device, non_blocking=True)
        costs.append((z.float() - zg.float()).square().mean(dim=(-1, -2)))
    return torch.cat([cost.detach().cpu() for cost in costs], dim=0)


def _oracle_plan(planner: CEMPlanner, mean, std, cost_fn, on_iter=None):
    trace = []
    for _ in range(planner.iters):
        cand = _sample(planner, mean, std)
        cost = cost_fn(cand)
        mean, std, vals = _elite_update(cand, cost, planner.topk)
        trace.append(float(vals.mean()))
        if on_iter:
            on_iter(trace[-1])
    return planner.clamp(mean), trace[-1]


def _finite_mean(vals) -> float | None:
    kept = [float(v) for v in vals if v is not None]
    return float(np.mean(kept)) if kept else None


def _summarize(rows: list[dict]) -> dict:
    return {
        'success_count': int(sum(r['success'] for r in rows)),
        'goal_pos_initial': float(np.mean([r['goal_pos_initial'] for r in rows])),
        'goal_pos_final': float(np.mean([r['goal_pos_final'] for r in rows])),
        'goal_pos_reduction_pct': float(np.mean([r['goal_pos_reduction_pct'] for r in rows])),
        'closer_count': int(sum(r['closer'] for r in rows)),
        'pairs': rows,
    }


def _oracle_closed_loop(model, session, episode, pair, cfg, device, seed, kind, on_iter, pool, bank):
    start, goal = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal)
    initial = _pusht_goal_distance(session)['pos']
    planner = CEMPlanner(
        model, 'pusht', cfg, session.action_low, session.action_high, device, seed
    )
    z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
    shape = (1, planner.block, planner.low.shape[0])
    steps, success = 0, False
    budget = int(cfg['eval']['exec_budget'])

    prime = _decision_prime(session, episode, start)

    def cost_fn(cand):
        frames, dists = _sim_outcomes(
            session, cand, render=(kind == 'B'), pool=pool, prime=prime
        )
        on_iter(len(dists))
        if kind == 'B':
            return _latent_cost(bank, frames, z_goal, cfg).to(device)
        return torch.as_tensor(dists, dtype=torch.float32, device=device)

    while steps < budget and not success:
        # Horizon is one block, so there is no leftover plan to warm-start.
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


def _correlation(model, session, episode, pair, cfg, device, seed, pool, bank) -> dict:
    start, goal = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal)
    planner = CEMPlanner(
        model, 'pusht', cfg, session.action_low, session.action_high, device, seed
    )
    z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
    z0 = _encode_frame(model, session.render(), cfg, device)[None]
    shape = (1, planner.block, planner.low.shape[0])
    mean = planner.center.expand(shape).clone()
    std = planner.init_std.expand(shape).clone()
    cand = _sample(planner, mean, std)
    ah = cand.new_zeros((1, 0, planner.block, planner.low.shape[0]))
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(cfg['train']['bf16'])):
        pred = model.rollout(z0.expand(cand.shape[0], -1, -1, -1), ah.expand(cand.shape[0], -1, -1, -1), cand, 'pusht')
    model_cost = (pred[:, -1].float() - z_goal.float()).square().mean(dim=(-1, -2)).cpu().numpy()
    prime = _decision_prime(session, episode, start)
    frames, _ = _sim_outcomes(session, cand, render=True, pool=pool, prime=prime)
    actual_cost = _latent_cost(bank, frames, z_goal, cfg).numpy()
    pick = int(np.argmin(model_cost))
    better = int(np.sum(actual_cost < actual_cost[pick]))
    return {
        'episode': int(pair['episode']),
        'n': int(len(model_cost)),
        'pearson': _corr(model_cost, actual_cost),
        'spearman': _spearman(model_cost, actual_cost),
        'model_pick_actual_percentile': 100.0 * better / max(1, len(actual_cost) - 1),
        'model_pick_actual_cost': float(actual_cost[pick]),
        'best_actual_cost': float(actual_cost.min()),
        'median_actual_cost': float(np.median(actual_cost)),
        'model_cost': model_cost,
        'actual_cost': actual_cost,
    }


def _bank(cfg, ckpt) -> list:
    count = min(4, torch.cuda.device_count())
    return [load_model(cfg, ckpt, torch.device(f'cuda:{i}')).eval() for i in range(count)]


def diagnose(cfg: dict, ckpt: Path, out: Path, device, workers: int = 32) -> dict:
    torch.set_grad_enabled(False)
    bank = _bank(cfg, ckpt)
    model = bank[0]
    hcfg = _cfg_horizon(cfg, 1)
    session = open_eval_session('pusht')
    reader = open_episode_reader(env_path(cfg, 'pusht'))
    pairs = get_pairs(cfg, 'pusht', 'val', cfg['eval']['pilot_val_pairs'], session, reader)[:10]
    cache: dict[int, dict] = {}

    def episode_of(ep: int):
        if ep not in cache:
            cache.clear()
            cache[ep] = reader.load_episode(ep)
        return cache[ep]

    log = out.parent / 'oracle_diag_progress.log'
    pool = PushTPool(workers)
    report_workers = workers
    report_gpus = [f'cuda:{i}' for i in range(len(bank))]
    report = {
        'ckpt': str(ckpt),
        'horizon_blocks': 1,
        'receding_blocks': 1,
        'cem': {
            'num_samples': cfg['cem']['num_samples'],
            'n_steps': cfg['cem']['n_steps'],
            'topk': cfg['cem']['topk'],
        },
        'seed0': 1000,
        'workers': report_workers,
        'encoder_devices': report_gpus,
        'note': (
            'A/B/C score one action block. A is model rollout + latent MSE. '
            'B is the simulator transition + the same latent MSE. '
            'C is the simulator transition + PushT position goal distance.'
        ),
    }

    bar = Progress(len(pairs), 0, 'Oracle A model+latent | PushT', log)
    rows_a = []
    for i, pair in enumerate(pairs):
        episode = episode_of(int(pair['episode']))
        row = _closed_loop(model, session, episode, pair, hcfg, device, 1000 + i)
        rows_a.append(row)
        bar.set_postfix(
            f"pair {i + 1}/10 success={row['success']} "
            f"dist {row['goal_pos_initial']:.0f}->{row['goal_pos_final']:.0f}"
        )
        bar.update(i + 1)
    bar.close()
    report['A'] = _summarize(rows_a)

    corr_total = len(pairs) * int(cfg['cem']['num_samples'])
    bar = Progress(corr_total, 0, 'A/B same candidates | PushT', log, unit='cand/s')
    corrs = []
    for i, pair in enumerate(pairs):
        episode = episode_of(int(pair['episode']))
        bar.desc = f'A/B same candidates | PushT | pair {i + 1}/10 | stage corr'
        corrs.append(_correlation(model, session, episode, pair, hcfg, device, 1000 + i, pool, bank))
        bar.set_postfix(
            f"pair {i + 1}/10 | stage corr | spearman {corrs[-1]['spearman']:.2f} "
            f"pick pct {corrs[-1]['model_pick_actual_percentile']:.0f}"
        )
        bar.update((i + 1) * int(cfg['cem']['num_samples']))
    bar.close()
    pooled_m = np.concatenate([c.pop('model_cost') for c in corrs])
    pooled_a = np.concatenate([c.pop('actual_cost') for c in corrs])
    report['candidate_correlation'] = {
        'per_pair': corrs,
        'pooled_pearson': _corr(pooled_m, pooled_a),
        'pooled_spearman': _spearman(pooled_m, pooled_a),
        'mean_pearson': float(np.nanmean([np.nan if c['pearson'] is None else c['pearson'] for c in corrs])),
        'mean_spearman': float(np.nanmean([np.nan if c['spearman'] is None else c['spearman'] for c in corrs])),
        'mean_model_pick_actual_percentile': float(np.mean([c['model_pick_actual_percentile'] for c in corrs])),
    }

    replans = cfg['eval']['exec_budget'] // cfg['model']['action_block']
    samples = int(cfg['cem']['num_samples'])
    total_candidates = len(pairs) * replans * int(cfg['cem']['n_steps']) * samples
    for kind, title in (('B', 'Oracle B sim+latent | PushT'), ('C', 'Oracle C sim+distance | PushT')):
        bar = Progress(total_candidates, 0, title, log, unit='cand/s')
        done = 0
        pair_box = {'i': 0}

        def on_iter(n_cand, _bar=bar, _kind=kind, _title=title):
            nonlocal done
            done += int(n_cand)
            _bar.desc = f'{_title} | pair {pair_box["i"] + 1}/10 | stage {_kind}'
            _bar.set_postfix(
                f"pair {pair_box['i'] + 1}/10 | stage {_kind}"
            )
            _bar.update(done)

        rows = []
        for i, pair in enumerate(pairs):
            pair_box['i'] = i
            episode = episode_of(int(pair['episode']))
            row = _oracle_closed_loop(
                model, session, episode, pair, hcfg, device, 1000 + i, kind, on_iter, pool, bank
            )
            rows.append(row)
            bar.desc = f'{title} | pair {i + 1}/10 | stage {kind}'
            bar.set_postfix(
                f"pair {i + 1}/10 | stage {kind} | "
                f"dist {row['goal_pos_initial']:.0f}->{row['goal_pos_final']:.0f}"
            )
            bar.update(done)
        bar.close()
        report[kind] = _summarize(rows)

    pool.close()
    session.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    _print(report)
    return report


def _print(report: dict) -> None:
    print('--- oracle planners, horizon 1 block, receding 1, 10 pairs ---', flush=True)
    for key, label in (
        ('A', 'A model+latent'),
        ('B', 'B sim+latent'),
        ('C', 'C sim+distance'),
    ):
        row = report[key]
        print(
            f"{label}: success {row['success_count']}/10 | "
            f"dist {row['goal_pos_initial']:.1f}->{row['goal_pos_final']:.1f} | "
            f"reduction {row['goal_pos_reduction_pct']:.1f}% | "
            f"closer {row['closer_count']}/10",
            flush=True,
        )
    corr = report['candidate_correlation']
    print(
        f"--- same 300 candidates: pooled pearson {corr['pooled_pearson']:.3f} | "
        f"pooled spearman {corr['pooled_spearman']:.3f} | "
        f"mean per-pair spearman {corr['mean_spearman']:.3f} | "
        f"model pick actual percentile {corr['mean_model_pick_actual_percentile']:.1f} ---",
        flush=True,
    )
    for i, row in enumerate(corr['per_pair']):
        def _n(v, spec):
            return 'na' if v is None else format(v, spec)

        print(
            f"pair {i}: pearson {_n(row['pearson'], '.2f')} spearman {_n(row['spearman'], '.2f')} "
            f"pick_pct {row['model_pick_actual_percentile']:.0f} "
            f"actual {row['model_pick_actual_cost']:.3f} best {row['best_actual_cost']:.3f}",
            flush=True,
        )
