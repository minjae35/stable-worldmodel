"""Does the frozen PushT latent contain local goal distance?

The encoder and world model stay frozen. A small MLP reads the 64
per-token squared differences between a current latent and a goal latent
and predicts PushT position distance. That distance is a probe label and
an evaluation label only. It is not fed to the world model or the planner.
The probe is fit on train-split episodes and scored on held-out val
candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rdwm.exp1.config import env_path
from rdwm.exp1.cost_diag import (
    _encode_batch,
    _load_pairs,
    _mean_metric,
    _rank_metrics,
    _sample_n,
    _token_mse,
    score_latents,
)
from rdwm.exp1.data import EnvIndex, split_episodes
from rdwm.exp1.horizon_diag import _cfg_horizon
from rdwm.exp1.oracle_diag import _bank, _decision_prime, _sim_outcomes
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


class DistanceProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _windows(index: EnvIndex, episodes: np.ndarray, n_episodes: int, seed: int, offset: int):
    rng = np.random.default_rng(seed)
    valid = [int(ep) for ep in episodes if int(index.lengths[ep]) > offset]
    rng.shuffle(valid)
    if len(valid) < n_episodes:
        raise RuntimeError(f'only {len(valid)} train episodes longer than {offset}')
    return valid[:n_episodes]


def _pair_rows(z: torch.Tensor, state: np.ndarray, offset: int, rng: np.random.Generator, n_goals: int):
    """Local pairs: current is within ``offset`` steps of a goal frame."""
    length = int(z.shape[0])
    goals = np.arange(offset, length)
    if len(goals) > n_goals:
        goals = rng.choice(goals, size=n_goals, replace=False)
    feats, labels = [], []
    goal_pos = state[:, :4]
    for goal in goals:
        zg = z[int(goal)]
        currents = range(int(goal) - offset, int(goal) + 1)
        diff = _token_mse(z[list(currents)], zg)
        dist = np.linalg.norm(goal_pos[list(currents)] - goal_pos[int(goal)], axis=1)
        feats.append(diff)
        labels.append(dist.astype(np.float32))
    return torch.cat(feats, dim=0), np.concatenate(labels)


def _collect(reader, bank, cfg, episodes: list[int], offset: int, n_goals: int, seed: int, bar: Progress, done: int):
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for ep in episodes:
        episode = reader.load_episode(int(ep))
        state = np.asarray(episode['state'], dtype=np.float64)
        frames = [dataset_frame(episode, t) for t in range(len(state))]
        z = _encode_batch(bank, frames, cfg)
        if int(z.shape[-2]) != 64:
            raise RuntimeError(f'expected 64 tokens, got {tuple(z.shape)}')
        feat, dist = _pair_rows(z, state, offset, rng, n_goals)
        feats.append(feat)
        labels.append(dist)
        done += 1
        bar.desc = f'Probe encode | PushT | episode {done}/{bar.total} | stage encode'
        bar.set_postfix(f'episode {done}/{bar.total} | stage encode')
        bar.update(done)
        del episode, frames
    return torch.cat(feats, dim=0), np.concatenate(labels)


def _fit(features: torch.Tensor, labels: np.ndarray, monitor_x: torch.Tensor, monitor_y: np.ndarray, device):
    y = torch.as_tensor(labels, dtype=torch.float32)
    x_mu, x_sd = features.mean(0), features.std(0).clamp_min(1e-6)
    y_mu, y_sd = y.mean(), y.std().clamp_min(1e-6)
    x = (features - x_mu) / x_sd
    target = (y - y_mu) / y_sd
    mx = (monitor_x - x_mu) / x_sd
    probe = DistanceProbe().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    n = len(x)
    gen = torch.Generator().manual_seed(0)
    best_state, best_loss, wait = None, float('inf'), 0
    history = []
    for epoch in range(1, 81):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        probe.train()
        for start in range(0, n, 512):
            batch = perm[start : start + 512]
            pred = probe(x[batch].to(device))
            loss = (pred - target[batch].to(device)).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        probe.eval()
        with torch.no_grad():
            monitor = probe(mx.to(device)).cpu() * y_sd + y_mu
        monitor_mse = float(np.mean((monitor.numpy() - monitor_y) ** 2))
        monitor_spear = _spearman(monitor.numpy(), monitor_y)
        history.append({'epoch': epoch, 'train_loss': total / n, 'monitor_mse': monitor_mse, 'monitor_spearman': monitor_spear})
        if monitor_mse < best_loss - 1e-4:
            best_loss = monitor_mse
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 12:
                break
    probe.load_state_dict(best_state)
    probe.eval()
    return probe, x_mu, x_sd, y_mu, y_sd, history


def _predict(probe, features: torch.Tensor, x_mu, x_sd, y_mu, y_sd, device) -> np.ndarray:
    with torch.no_grad():
        pred = probe(((features - x_mu) / x_sd).to(device))
    return (pred.cpu() * y_sd + y_mu).numpy()


def diagnose(cfg: dict, ckpt: Path, out: Path, device, workers: int = 32) -> dict:
    torch.set_grad_enabled(False)
    bank = _bank(cfg, ckpt)
    for model in bank:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    model = bank[0]
    offset = int(cfg['eval']['goal_offset'])
    index = EnvIndex('pusht', env_path(cfg, 'pusht'), cfg['paths']['cache_root'])
    splits = split_episodes(index.num_episodes, cfg['data']['split'])
    fit_eps = _windows(index, splits['train'], 200, seed=0, offset=offset)
    held = np.asarray([ep for ep in splits['train'] if int(ep) not in set(fit_eps)])
    monitor_eps = _windows(index, held, 50, seed=1, offset=offset)
    reader = open_episode_reader(env_path(cfg, 'pusht'))
    log = out.parent / 'probe_diag_progress.log'
    encode_total = len(fit_eps) + len(monitor_eps)
    bar = Progress(encode_total, 0, 'Probe encode | PushT', log, unit='ep/s')
    fit_x, fit_y = _collect(reader, bank, cfg, fit_eps, offset, n_goals=8, seed=2, bar=bar, done=0)
    mon_x, mon_y = _collect(reader, bank, cfg, monitor_eps, offset, n_goals=4, seed=3, bar=bar, done=len(fit_eps))
    bar.close()

    torch.set_grad_enabled(True)
    probe, x_mu, x_sd, y_mu, y_sd, history = _fit(fit_x, fit_y, mon_x, mon_y, device)
    torch.set_grad_enabled(False)
    monitor_pred = _predict(probe, mon_x, x_mu, x_sd, y_mu, y_sd, device)

    hcfg = _cfg_horizon(cfg, 1)
    session = open_eval_session('pusht')
    pairs = _load_pairs(cfg, session, reader, offset=0, n_pairs=10)
    val_ids = set(int(ep) for ep in splits['val'])
    if any(int(pair['episode']) not in val_ids for pair in pairs):
        raise RuntimeError('val pairs are not inside the val episode split')
    if any(int(pair['episode']) in set(fit_eps) for pair in pairs):
        raise RuntimeError('a val pair episode was used to fit the probe')
    pool = PushTPool(workers)
    rows = {'mean': [], 'probe': []}
    total = 10 * 512
    rank_bar = Progress(total, 0, 'Probe rank | PushT', log, unit='cand/s')
    done = 0
    try:
        for i, pair in enumerate(pairs):
            episode = reader.load_episode(int(pair['episode']))
            start, goal = int(pair['start']), int(pair['goal'])
            session.restore(episode, start)
            session.set_goal(episode, goal)
            initial = _pusht_goal_distance(session)['pos']
            planner = CEMPlanner(
                model, 'pusht', hcfg, session.action_low, session.action_high, device, 1000 + i
            )
            z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device).cpu()
            cand = _sample_n(planner, 512, 1000 + i)
            prime = _decision_prime(session, episode, start)
            rank_bar.desc = f'Probe rank | PushT | pair {i + 1}/10 | stage rank'
            frames, dists = _sim_outcomes(session, cand, render=True, pool=pool, prime=prime)
            z = _encode_batch(bank, frames, cfg)
            mean_cost = score_latents(z, z_goal, {'name': 'mean', 'kind': 'mean'}).numpy()
            probe_cost = _predict(probe, _token_mse(z, z_goal), x_mu, x_sd, y_mu, y_sd, device)
            for name, cost in (('mean', mean_cost), ('probe', probe_cost)):
                metrics = _rank_metrics(cost, dists, initial)
                metrics.update({
                    'episode': int(pair['episode']),
                    'start': start,
                    'goal': goal,
                    'initial_distance': initial,
                })
                rows[name].append(metrics)
            done += len(dists)
            rank_bar.set_postfix(f'pair {i + 1}/10 | stage rank')
            rank_bar.update(done)
        rank_bar.close()
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
            'mean_n_closer': _mean_metric(pair_rows, 'n_closer'),
            'pairs': pair_rows,
        }
    report = {
        'ckpt': str(ckpt),
        'probe': 'mlp 64-64-1 on per-token squared latent difference',
        'fit_episodes': len(fit_eps),
        'monitor_episodes': len(monitor_eps),
        'fit_pairs': int(len(fit_y)),
        'monitor_pairs': int(len(mon_y)),
        'monitor_spearman': _spearman(monitor_pred, mon_y),
        'monitor_mse': float(np.mean((monitor_pred - mon_y) ** 2)),
        'epochs': history,
        'candidate_costs': summary,
        'note': (
            'Probe fit uses train episodes only. Monitor episodes are a '
            'disjoint subset of the train split. Candidate scores use the '
            'fixed val pairs and are not used for fitting.'
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _print(report)
    out.write_text(json.dumps(report, indent=2))
    return report


def _fmt(value) -> str:
    if value is None:
        return '  nan'
    return f'{value:6.2f}'


def _print(report: dict) -> None:
    print(
        f"--- probe monitor spearman {report['monitor_spearman']:.3f} "
        f"mse {report['monitor_mse']:.2f} | fit pairs {report['fit_pairs']} ---",
        flush=True,
    )
    print('cost | spearman | best dist pct | top5 dist pct | closer rank pct | n_closer', flush=True)
    for name, row in report['candidate_costs'].items():
        print(
            f"{name:5} | {_fmt(row['mean_spearman'])} | "
            f"{_fmt(row['mean_best_distance_percentile'])} | "
            f"{_fmt(row['mean_top5_distance_percentile'])} | "
            f"{_fmt(row['mean_closer_rank_percentile'])} | "
            f"{_fmt(row['mean_n_closer'])}",
            flush=True,
        )
