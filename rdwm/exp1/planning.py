"""CEM planning and closed-loop evaluation (experiment-1.md Section 5).

- Candidates live in native action space. The same native bounds clamp the
  candidates fed to the model rollout, the elites used to update the
  distribution, and the actions executed in the simulator. The model
  normalizes actions internally, once.
- Cost: mean squared distance between the last predicted latent and the
  goal latent. Action and env embeddings are not part of the cost.
- Receding horizon 1 block: execute 5 native actions, observe, replan.
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.data import preprocess


class CEMPlanner:
    def __init__(self, model, env: str, cfg: dict, low, high, device, seed: int):
        ccfg = cfg['cem']
        self.model = model
        self.env = env
        self.cfg = cfg
        self.samples = ccfg['num_samples']
        self.iters = ccfg['n_steps']
        self.topk = ccfg['topk']
        self.horizon = ccfg['horizon']
        self.block = cfg['model']['action_block']
        self.device = device
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        self.center = (self.low + self.high) / 2
        self.init_std = ccfg['init_std'] * (self.high - self.low) / 2
        self.bf16 = bool(cfg['train']['bf16']) and device.type == 'cuda'
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def clamp(self, actions):
        return torch.maximum(torch.minimum(actions, self.high), self.low)

    @torch.no_grad()
    def plan(self, z_hist, a_hist, z_goal, init_mean=None) -> tuple[torch.Tensor, dict]:
        """z_hist [1,H,N,D], a_hist [1,H-1,5,d], z_goal [1,N,D] ->
        native plan [horizon, 5, d] plus per-iteration cost stats."""
        d = self.low.shape[0]
        shape = (self.horizon, self.block, d)
        mean = self.center.expand(shape).clone()
        if init_mean is not None and init_mean.shape[0] > 0:
            mean[: init_mean.shape[0]] = init_mean
        std = self.init_std.expand(shape).clone()
        z = z_hist.expand(self.samples, *z_hist.shape[1:])
        ah = a_hist.expand(self.samples, *a_hist.shape[1:])
        goal = z_goal.float()
        trace = []
        for _ in range(self.iters):
            noise = torch.randn(
                (self.samples, *shape), generator=self.gen, device=self.device
            )
            cand = noise * std + mean
            cand[0] = mean
            cand = self.clamp(cand)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
                preds = self.model.rollout(z, ah, cand, self.env)
            cost = (preds[:, -1].float() - goal).square().mean(dim=(-1, -2))
            vals, idx = torch.topk(cost, self.topk, largest=False)
            elites = cand[idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, correction=0)
            trace.append(
                (float(cost.mean()), float(vals.mean()), float(vals[0]))
            )
        return self.clamp(mean), {
            'cost_mean_first': trace[0][0],
            'elite_cost_first': trace[0][1],
            'elite_cost_last': trace[-1][1],
            'best_cost_last': trace[-1][2],
        }


def _encode_frame(model, frame_hwc: np.ndarray, cfg: dict, device) -> torch.Tensor:
    pixels = torch.from_numpy(np.ascontiguousarray(frame_hwc)).permute(2, 0, 1)
    pixels = preprocess(pixels[None, None].to(device), cfg)
    with torch.autocast(
        'cuda', dtype=torch.bfloat16, enabled=bool(cfg['train']['bf16'])
    ):
        return model.encode(pixels).float()[:, 0]


@torch.no_grad()
def run_pair(model, session, episode, pair: dict, cfg: dict, device, seed: int) -> dict:
    """Closed-loop CEM on one (start, goal) pair of a held-out trajectory."""
    from rdwm.envs.sessions import dataset_frame

    ecfg, ccfg = cfg['eval'], cfg['cem']
    env = session.name
    start, goal = int(pair['start']), int(pair['goal'])
    session.restore(episode, start)
    session.set_goal(episode, goal)
    initial = _pusht_goal_distance(session)
    if session.success():
        return {**pair, 'skipped_start_success': True, **_distance_fields(initial, initial)}
    planner = CEMPlanner(
        model, env, cfg, session.action_low, session.action_high, device, seed
    )
    z_goal = _encode_frame(model, dataset_frame(episode, goal), cfg, device)
    z_hist = [_encode_frame(model, session.render(), cfg, device)]
    a_hist: list[torch.Tensor] = []
    steps, success, warm = 0, False, None
    plans, t0 = [], time.time()
    while steps < ecfg['exec_budget'] and not success:
        h = min(len(z_hist), ccfg['history'])
        zh = torch.stack(z_hist[-h:], dim=1)
        if h > 1:
            ah = torch.stack(a_hist[-(h - 1):], dim=0)[None]
        else:
            ah = torch.zeros(
                (1, 0, planner.block, planner.low.shape[0]), device=device
            )
        plan, info = planner.plan(zh, ah, z_goal, warm)
        plans.append(info)
        block = plan[: ccfg['receding_horizon']].reshape(-1, plan.shape[-1])
        for action in block.cpu().numpy():
            session.step(action.astype(np.float64))
            steps += 1
            if session.success():
                success = True
                break
            if steps >= ecfg['exec_budget']:
                break
        if success or steps >= ecfg['exec_budget']:
            break
        z_hist.append(_encode_frame(model, session.render(), cfg, device))
        a_hist.append(plan[0])
        warm = plan[ccfg['receding_horizon']:] if ccfg['warm_start'] else None
    final = _pusht_goal_distance(session)
    return {
        **pair,
        **_distance_fields(initial, final),
        'success': bool(success),
        'steps': steps,
        'replans': len(plans),
        'seconds': time.time() - t0,
        'elite_cost_first_iter_mean': float(np.mean([p['elite_cost_first'] for p in plans])),
        'elite_cost_last_iter_mean': float(np.mean([p['elite_cost_last'] for p in plans])),
        'plans': plans,
    }


def _pusht_goal_distance(session) -> dict | None:
    """PushT position distance (first 4 state dims) and full-state L2."""
    if getattr(session, 'name', None) != 'pusht':
        return None
    from rdwm.envs.sessions import _core

    goal = np.asarray(_core(session.env).goal_state, dtype=np.float64).reshape(-1)
    cur = np.asarray(session.state(), dtype=np.float64).reshape(-1)
    n = min(goal.shape[0], cur.shape[0])
    pos_n = min(4, n)
    return {
        'pos': float(np.linalg.norm(goal[:pos_n] - cur[:pos_n])),
        'state': float(np.linalg.norm(goal[:n] - cur[:n])),
    }


def _distance_fields(initial: dict | None, final: dict | None) -> dict:
    if not initial or not final:
        return {}
    reduction = None
    if initial['pos'] > 0:
        reduction = (initial['pos'] - final['pos']) / initial['pos'] * 100.0
    return {
        'goal_pos_initial': initial['pos'],
        'goal_pos_final': final['pos'],
        'goal_pos_reduction_pct': reduction,
        'goal_state_initial': initial['state'],
        'goal_state_final': final['state'],
    }


def make_pairs(
    session, dataset, episodes: np.ndarray, lengths: np.ndarray, n: int, cfg: dict, tag: str
) -> dict:
    """Deterministic (episode, start, goal = start + 25) pairs from a split.

    Pairs that already satisfy the success predicate at the start are
    rejected and counted. The list is seed-independent and shared by every
    variant.
    """
    ecfg = cfg['eval']
    offset = ecfg['goal_offset']
    seed = [int(ecfg['pair_seed']), zlib.crc32(tag.encode())]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    eligible = np.array([e for e in episodes if lengths[e] > offset])
    pairs, seen, rejected, attempts = [], set(), 0, 0
    cache: dict[int, dict] = {}
    while len(pairs) < n and attempts < 50 * n:
        attempts += 1
        ep = int(rng.choice(eligible))
        start = int(rng.integers(0, int(lengths[ep]) - offset))
        if (ep, start) in seen:
            continue
        seen.add((ep, start))
        if ep not in cache:
            cache.clear()
            cache[ep] = dataset.load_episode(ep)
        episode = cache[ep]
        session.restore(episode, start)
        session.set_goal(episode, start + offset)
        if session.success():
            rejected += 1
            continue
        pairs.append({'episode': ep, 'start': start, 'goal': start + offset})
    if len(pairs) < n:
        raise RuntimeError(f'{tag}: only {len(pairs)} pairs after {attempts} draws')
    return {'tag': tag, 'pairs': pairs, 'rejected_start_success': rejected}


def open_eval_session(env: str):
    from rdwm.data.manifest import ENVS
    from rdwm.envs.sessions import open_session

    spec = ENVS[env]
    if env == 'pointmaze':
        return open_session(env, spec, width=224, height=224)
    if env == 'antmaze':
        return open_session(env, spec, maze_type='medium')
    return open_session(env, spec)


def open_episode_reader(path: str):
    from rdwm.data.io import open_lance

    return open_lance(path)


def pairs_path(root: Path, split: str, env: str, n: int) -> Path:
    return root / split / f'{env}_n{n}.json'
