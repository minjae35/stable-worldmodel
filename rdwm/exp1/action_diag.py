"""Action-conditioning diagnostics (PushT investigation after Stage 1).

Read-only: nothing here changes the model, the data, or the config.
"""

from __future__ import annotations

import numpy as np
import torch

from rdwm.exp1.config import env_path
from rdwm.exp1.data import EnvIndex, preprocess, split_episodes
from rdwm.exp1.diagnostics import load_clips, val_clips
from rdwm.exp1.losses import SIGReg, recursive_loss


def _corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))


def dataset_alignment(cfg: dict, env: str, n_episodes: int = 400) -> dict:
    """Is ``action[t]`` the action that moves the agent from t to t+1?

    Uses the PushT ``state`` column (agent xy in dims 0:2) only for this
    check; the model never sees it.
    """
    import lance

    index = EnvIndex(env, env_path(cfg, env), cfg['paths']['cache_root'])
    episodes = split_episodes(index.num_episodes, cfg['data']['split'])['val'][:n_episodes]
    table = lance.dataset(env_path(cfg, env))
    rows = np.concatenate(
        [np.arange(index.offsets[e], index.offsets[e] + index.lengths[e]) for e in episodes]
    )
    state = np.asarray(
        table.take(rows.tolist(), columns=['state']).column('state').to_pylist(),
        dtype=np.float64,
    )
    actions = np.asarray(index.actions[rows], dtype=np.float64)
    out = {'episodes': len(episodes), 'step_lag_corr': {}, 'block_lag_corr': {}}
    step_pairs = {lag: ([], []) for lag in (-2, -1, 0, 1, 2)}
    block_pairs = {lag: ([], []) for lag in (-1, 0, 1)}
    pos = 0
    for e in episodes:
        n = int(index.lengths[e])
        s = state[pos : pos + n, :2]
        a = actions[pos : pos + n]
        pos += n
        delta = s[1:] - s[:-1]  # step t -> t+1, length n-1
        for lag in step_pairs:
            t = np.arange(max(0, -lag), min(n - 1, n - 1 - lag))
            t = t[np.isfinite(a[t + lag]).all(axis=1)]
            step_pairs[lag][0].append(a[t + lag])
            step_pairs[lag][1].append(delta[t])
        # stride-5 blocks: block k = a[5k:5k+5], displacement o_k -> o_{k+1}
        nb = (n - 1) // 5
        for k in range(1, nb - 1):
            disp = s[5 * k + 5] - s[5 * k]
            for lag in block_pairs:
                blk = a[5 * (k + lag) : 5 * (k + lag) + 5]
                if np.isfinite(blk).all():
                    block_pairs[lag][0].append(blk.mean(axis=0))
                    block_pairs[lag][1].append(disp)
    for lag, (xa, ya) in step_pairs.items():
        x, y = np.concatenate(xa), np.concatenate(ya)
        out['step_lag_corr'][lag] = [_corr(x[:, d], y[:, d]) for d in range(2)]
    for lag, (xa, ya) in block_pairs.items():
        x, y = np.asarray(xa), np.asarray(ya)
        out['block_lag_corr'][lag] = [_corr(x[:, d], y[:, d]) for d in range(2)]
    return out


def visual_change(cfg: dict, env: str, n: int = 256) -> dict:
    """How much of the frame changes between consecutive stride-5 observations."""
    index, eps, starts = val_clips(cfg, env, n, seed=1)
    raw, _ = load_clips(cfg, env, index, eps, starts)
    x = raw.float()
    diff = (x[:, 1:] - x[:, :-1]).abs().mean(dim=2)  # [n, 4, H, W]
    h = diff.shape[-1]
    patch = h // 8
    grid = diff.reshape(n, 4, 8, patch, 8, patch).mean(dim=(3, 5))  # 8x8 = token grid
    return {
        'mean_abs_pixel_diff': float(diff.mean()),
        'frac_pixels_changed_gt10': float((diff > 10).float().mean()),
        'frac_tokens_changed_gt2': float((grid > 2).float().mean()),
        'frac_tokens_changed_gt5': float((grid > 5).float().mean()),
    }


def normalized_action_stats(model, env: str, actions: torch.Tensor) -> dict:
    a = model.action_norm[env](actions.float()).reshape(-1, actions.shape[-1])
    return {
        'mean': a.mean(dim=0).tolist(),
        'std': a.std(dim=0).tolist(),
        'min': a.min(dim=0).values.tolist(),
        'max': a.max(dim=0).values.tolist(),
        'frac_abs_gt3': float((a.abs() > 3).float().mean()),
        'raw_mean': actions.reshape(-1, actions.shape[-1]).mean(dim=0).tolist(),
        'raw_std': actions.reshape(-1, actions.shape[-1]).std(dim=0).tolist(),
        'train_mean': model.action_norm[env].mean.tolist(),
        'train_std': model.action_norm[env].std.tolist(),
    }


@torch.no_grad()
def embedding_magnitudes(model, env: str, actions: torch.Tensor) -> dict:
    a = model.action_norm[env](actions.float()).flatten(-2)  # [n, 4, 5d]
    emb = model.action_adapter[env](a)  # [n, 4, 192]
    env_emb = model.env_embed[env].float()
    flat = emb.reshape(-1, emb.shape[-1]).float()
    centered = flat - flat.mean(dim=0, keepdim=True)
    cond = flat + env_emb
    out = {
        'action_emb_norm_mean': float(flat.norm(dim=-1).mean()),
        'action_emb_mean_vector_norm': float(flat.mean(dim=0).norm()),
        'action_emb_varying_norm': float(centered.norm(dim=-1).mean()),
        'env_emb_norm': float(env_emb.norm()),
        'cond_norm_mean': float(cond.norm(dim=-1).mean()),
    }
    out['varying_over_env_ratio'] = out['action_emb_varying_norm'] / max(1e-12, out['env_emb_norm'])
    blocks = []
    for block in model.dynamics.blocks:
        mod = block.adaln(cond)  # [m, 6D]
        parts = mod.chunk(6, dim=-1)
        names = ['shift1', 'scale1', 'gate1', 'shift2', 'scale2', 'gate2']
        row = {}
        for name, part in zip(names, parts):
            row[f'{name}_norm'] = float(part.norm(dim=-1).mean())
            row[f'{name}_action_var_norm'] = float(
                (part - part.mean(dim=0, keepdim=True)).norm(dim=-1).mean()
            )
        blocks.append(row)
    out['adaln_blocks'] = blocks
    return out


def gradient_norms(model, cfg: dict, env: str, pixels, actions) -> dict:
    """One backward pass of the training loss on held-out clips (no step)."""
    model.zero_grad(set_to_none=True)
    model.eval()
    torch.manual_seed(0)
    terms = recursive_loss(model, pixels, actions, env, 3, SIGReg().to(pixels.device), cfg)
    terms['loss'].backward()

    def norm(prefix):
        total = 0.0
        for name, p in model.named_parameters():
            if name.startswith(prefix) and p.grad is not None:
                total += float(p.grad.float().pow(2).sum())
        return total ** 0.5

    out = {
        'action_adapter': norm(f'action_adapter.{env}.'),
        'env_embed': norm(f'env_embed.{env}'),
        'adaln_all_blocks': sum(
            float(p.grad.float().pow(2).sum())
            for n, p in model.named_parameters()
            if '.adaln.' in n and p.grad is not None
        ) ** 0.5,
        'dynamics': norm('dynamics.'),
        'encoder': norm('encoder.'),
        'head': norm('head.'),
        'loss_terms': {k: float(v.detach()) for k, v in terms.items()},
    }
    model.zero_grad(set_to_none=True)
    return out


@torch.no_grad()
def sensitivity(model, cfg: dict, env: str, pixels, actions) -> dict:
    """2-step rollout under true / shuffled / zero / mean actions."""
    n = pixels.shape[0]
    bf16 = bool(cfg['train']['bf16'])
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        z = model.encode(pixels).float()
    target = z[:, 3:5]
    perm = torch.roll(torch.arange(n, device=pixels.device), 1)
    variants = {
        'true': actions,
        'shuffled': actions[perm],
        'zero_native': torch.zeros_like(actions),
        'train_mean': model.action_norm[env].mean.expand_as(actions).clone(),
    }
    preds = {}
    for name, acts in variants.items():
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            preds[name] = model.rollout(z[:, :3], acts[:, :2], acts[:, 2:4], env).float()
    persist = (z[:, 2:3] - target).square().mean(dim=(0, 2, 3))
    out = {'persistence_mse': persist.tolist()}
    for name, p in preds.items():
        out[f'mse_{name}'] = (p - target).square().mean(dim=(0, 2, 3)).tolist()
        if name != 'true':
            d = (p - preds['true']).square()
            out[f'diff_true_vs_{name}'] = d.mean(dim=(0, 2, 3)).tolist()
            per_token = d[:, 1].mean(dim=(0, 2))
            share = torch.sort(per_token, descending=True).values
            out[f'diff_{name}_top4_token_share'] = float(share[:4].sum() / share.sum().clamp_min(1e-12))
    out['true_target_change_mse'] = (z[:, 3] - z[:, 2]).square().mean().item()
    out['rel_action_effect_t2'] = out['diff_true_vs_shuffled'][1] / max(1e-12, persist[1].item())
    return out


def checkpoint_report(model, cfg: dict, env: str, n: int = 256, device=None) -> dict:
    device = device or torch.device('cuda')
    index, eps, starts = val_clips(cfg, env, n)
    raw, actions = load_clips(cfg, env, index, eps, starts)
    pixels = preprocess(raw.to(device), cfg)
    actions = actions.to(device)
    return {
        'normalized_actions': normalized_action_stats(model, env, actions),
        'magnitudes': embedding_magnitudes(model, env, actions),
        'grad_norms': gradient_norms(model, cfg, env, pixels[:32], actions[:32]),
        'sensitivity': sensitivity(model, cfg, env, pixels, actions),
    }
