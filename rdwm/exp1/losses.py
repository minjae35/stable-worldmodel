"""2-step recursive latent prediction + SIGReg (experiment-1.md Section 2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (Epps-Pulley on random projections).

    Same statistic as ``stable_worldmodel.wm.loss.SIGReg``, without the
    einops import that module pulls in.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, x):
        """x: [G, B, D]; the statistic runs over B for each of the G groups."""
        x = x.float()
        proj = torch.randn(x.shape[-1], self.num_proj, device=x.device)
        proj = proj / proj.norm(p=2, dim=0)
        x_t = (x @ proj).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(
            -3
        ).square()
        statistic = (err @ self.weights) * x.shape[-2]
        return statistic.mean()


def sigreg_groups(z: torch.Tensor, positions: int) -> torch.Tensor:
    """[B, T, N, D] encoded latents -> [G, B, D] groups for SIGReg.

    Spatial models pick ``positions`` token positions per batch and form one
    group per (position, frame) over the batch axis. A batch holds a single
    env, so every group is same-env. Pooled models (N = 1) use one group per
    frame.
    """
    b, t, n, d = z.shape
    if n > positions:
        pick = torch.randperm(n, device=z.device)[:positions]
        z = z[:, :, pick]
    return z.permute(1, 2, 0, 3).reshape(-1, b, d)


def recursive_loss(
    model,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    env: str,
    history: int,
    sigreg: SIGReg,
    cfg: dict,
) -> dict[str, torch.Tensor]:
    """pixels: [B, 5, 3, 224, 224], actions: [B, 4, 5, d].

    Observations o0..o4; block k connects o_k -> o_{k+1}. Targets are o3
    (t+1) and o4 (t+2). History ``h`` keeps the last ``h`` context frames
    (o_{3-h}..o2) and drops earlier ones. The second prediction consumes the
    first predicted latent; nothing is detached and the targets come from
    the same encoder with gradient.
    """
    if history not in (1, 2, 3):
        raise ValueError(f'history must be 1, 2, or 3, got {history}')
    tcfg = cfg['train']
    first = 3 - history
    use_bf16 = bool(tcfg['bf16']) and pixels.is_cuda
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        z = model.encode(pixels[:, first:])
        context = z[:, :history]
        preds = model.rollout(context, actions[:, first:2], actions[:, 2:4], env)
    z = z.float()
    preds = preds.float()
    target1, target2 = z[:, history], z[:, history + 1]
    mse1 = F.mse_loss(preds[:, 0], target1)
    mse2 = F.mse_loss(preds[:, 1], target2)
    sig = sigreg(sigreg_groups(z, tcfg['sigreg']['positions']))
    w = tcfg['loss_weights']
    loss = w['t1'] * mse1 + w['t2'] * mse2 + w['sigreg'] * sig
    with torch.no_grad():
        persist1 = F.mse_loss(context[:, -1].float(), target1)
        flat = z.reshape(-1, z.shape[-1])
        std = flat.std(dim=0)
    return {
        'loss': loss,
        'mse_t1': mse1,
        'mse_t2': mse2,
        'sigreg': sig,
        'persist_t1': persist1,
        'z_std_mean': std.mean(),
        'z_std_min': std.min(),
        'z_dead_frac': (std < 1e-2).float().mean(),
    }


def one_step_loss(
    model,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    env: str,
    history: int,
    sigreg: SIGReg,
    cfg: dict,
) -> dict[str, torch.Tensor]:
    """Same clips as ``recursive_loss``, without the second recursive step.

    Context is the real encoded history. The target is the next encoded
    observation, not a rollout of the model's own prediction. The one-step
    term keeps the combined t+1 and t+2 weight so SIGReg is not rescaled.
    """
    if history not in (1, 2, 3):
        raise ValueError(f'history must be 1, 2, or 3, got {history}')
    tcfg = cfg['train']
    first = 3 - history
    use_bf16 = bool(tcfg['bf16']) and pixels.is_cuda
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        z = model.encode(pixels[:, first : first + history + 1])
        context = z[:, :history]
        preds = model.rollout(context, actions[:, first:2], actions[:, 2:3], env)
    z = z.float()
    preds = preds.float()
    target = z[:, history]
    mse1 = F.mse_loss(preds[:, 0], target)
    sig = sigreg(sigreg_groups(z, tcfg['sigreg']['positions']))
    w = tcfg['loss_weights']
    loss = (w['t1'] + w['t2']) * mse1 + w['sigreg'] * sig
    with torch.no_grad():
        persist1 = F.mse_loss(context[:, -1].float(), target)
        flat = z.reshape(-1, z.shape[-1])
        std = flat.std(dim=0)
    return {
        'loss': loss,
        'mse_t1': mse1,
        'sigreg': sig,
        'persist_t1': persist1,
        'z_std_mean': std.mean(),
        'z_std_min': std.min(),
        'z_dead_frac': (std < 1e-2).float().mean(),
    }
