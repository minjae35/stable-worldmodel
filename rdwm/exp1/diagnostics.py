"""Held-out collapse / prediction / action-sensitivity diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.data import EnvIndex, clip_table, decode_jpegs, preprocess, split_episodes


def val_clips(cfg: dict, env: str, n: int, seed: int = 0):
    index = EnvIndex(env, env_path(cfg, env), cfg['paths']['cache_root'])
    episodes = split_episodes(index.num_episodes, cfg['data']['split'])['val']
    eps, starts = clip_table(index, episodes, cfg['data']['num_obs'], cfg['data']['obs_stride'])
    pick = np.random.default_rng(seed).choice(len(eps), n, replace=False)
    return index, eps[pick], starts[pick]


def load_clips(cfg: dict, env: str, index, eps, starts):
    import lance

    table = lance.dataset(env_path(cfg, env))
    base = index.offsets[eps] + starts
    rows = base[:, None] + 5 * np.arange(5)[None]
    frames = decode_jpegs(table.take(rows.reshape(-1).tolist(), columns=['pixels'])
                          .column('pixels').to_pylist())
    pixels = frames.reshape(len(eps), 5, *frames.shape[1:])
    act_rows = base[:, None] + np.arange(20)[None]
    acts = np.asarray(index.actions[act_rows.reshape(-1)], dtype=np.float32)
    return pixels, torch.from_numpy(acts.reshape(len(eps), 4, 5, -1))


def effective_rank(x: torch.Tensor) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x.double())
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


@torch.no_grad()
def diagnose(model, cfg: dict, env: str, n: int = 128, device=None) -> dict:
    device = device or torch.device('cuda')
    index, eps, starts = val_clips(cfg, env, n)
    raw, actions = load_clips(cfg, env, index, eps, starts)
    pixels = preprocess(raw.to(device), cfg)
    actions = actions.to(device)
    bf16 = bool(cfg['train']['bf16'])
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        z = model.encode(pixels).float()
        pred = model.rollout(z[:, :3], actions[:, :2], actions[:, 2:4], env).float()
        perm = torch.roll(torch.arange(n, device=device), 1)
        shuffled = model.rollout(z[:, :3], actions[perm, :2], actions[perm, 2:4], env).float()
    t1, t2 = z[:, 3], z[:, 4]
    flat = z.reshape(-1, z.shape[-1])
    per_token = z[:, 2].reshape(n, -1)
    return {
        'n_clips': n,
        'z_batch_std_mean': float(z.std(dim=0).mean()),
        'z_batch_std_min_dim': float(z.std(dim=0).mean(dim=(0, 1)).min()),
        'z_norm_mean': float(z.norm(dim=-1).mean()),
        'eff_rank_tokens': effective_rank(flat[:: max(1, flat.shape[0] // 20000)]),
        'eff_rank_frames': effective_rank(per_token),
        'mse_t1': float((pred[:, 0] - t1).square().mean()),
        'mse_t2': float((pred[:, 1] - t2).square().mean()),
        'persist_t1': float((z[:, 2] - t1).square().mean()),
        'persist_t2': float((z[:, 2] - t2).square().mean()),
        'mse_t2_shuffled_actions': float((shuffled[:, 1] - t2).square().mean()),
        'action_effect': float((pred[:, 1] - shuffled[:, 1]).square().mean()),
    }
