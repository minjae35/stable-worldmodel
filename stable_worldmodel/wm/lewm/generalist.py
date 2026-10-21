"""Stage 2-A generalist: one LeWM encoder and one dynamics model.

PushT, TwoRoom, and OGBench Cube share the visual encoder, the CLS
projector, the temporal predictor, and the prediction head. Each
environment keeps its own action adapter because the blocked action
width differs. An environment embedding is added to the action
conditioning that AdaLN already reads, so the predictor stays the
LeWM predictor.
"""

from __future__ import annotations

import torch
from torch import nn

from stable_worldmodel.wm.lewm.lewm import LeWM


class EnvActionBank(nn.Module):
    """Per-environment action adapters with one active environment.

    ``forward`` is what ``LeWM.rollout`` calls. Planning runs one
    environment at a time, so the bank dispatches on ``active_env``.
    Joint training sets that attribute before each environment's
    adapter call.
    """

    def __init__(self, adapters: dict[str, nn.Module], active_env: str):
        super().__init__()
        if active_env not in adapters:
            raise KeyError(
                f'active_env {active_env!r} is not in {list(adapters)}'
            )
        self.adapters = nn.ModuleDict(adapters)
        self.active_env = active_env

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.adapters[self.active_env](action)


class GeneralistLeWM(LeWM):
    """Shared LeWM trunk plus environment-specific action adapters."""

    def __init__(
        self,
        encoder,
        predictor,
        action_adapters: dict[str, nn.Module],
        env_names: list[str],
        projector=None,
        pred_proj=None,
        active_env: str | None = None,
        **kwargs,
    ):
        names = [str(name) for name in env_names]
        if not names:
            raise ValueError('env_names is empty')
        unknown = [name for name in names if name not in action_adapters]
        extra = [name for name in action_adapters if name not in names]
        if unknown or extra:
            raise ValueError(
                'action_adapters and env_names disagree: '
                f'missing={unknown}, extra={extra}'
            )
        ordered = {name: action_adapters[name] for name in names}
        chosen = names[0] if active_env is None else str(active_env)
        bank = EnvActionBank(ordered, chosen)
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=bank,
            projector=projector,
            pred_proj=pred_proj,
        )
        self.env_names = names
        self.env_to_id = {name: i for i, name in enumerate(names)}
        self.env_embed = nn.Embedding(len(names), predictor.input_dim)
        self.active_env = chosen

    def set_active_env(self, name: str) -> None:
        name = str(name)
        if name not in self.env_to_id:
            raise KeyError(
                f'Unknown environment {name!r}. Known: {self.env_names}'
            )
        self.active_env = name
        self.action_encoder.active_env = name

    def _action_condition(self, act_emb: torch.Tensor) -> torch.Tensor:
        index = torch.tensor(
            self.env_to_id[self.action_encoder.active_env],
            device=act_emb.device,
        )
        bias = self.env_embed(index).to(dtype=act_emb.dtype)
        return act_emb + bias.view(1, 1, -1)

    def predict(self, emb, act_emb):
        return super().predict(emb, self._action_condition(act_emb))


def latent_stats(emb: torch.Tensor) -> dict[str, torch.Tensor]:
    """Batch statistics used to notice a collapsed latent."""
    flat = emb.detach().float()
    per_dim = flat.std(dim=(0, 1), unbiased=False)
    return {
        'latent_mean': flat.mean(),
        'latent_std': flat.std(unbiased=False),
        'latent_dim_std_mean': per_dim.mean(),
        'latent_dim_std_min': per_dim.min(),
    }


def generalist_loss(
    model: GeneralistLeWM,
    sigreg: nn.Module,
    batch: dict[str, dict[str, torch.Tensor]],
    *,
    history_size: int,
    num_preds: int,
    sigreg_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Next-latent MSE plus per-environment SIGReg, then a mean.

    Pixels from every environment go through the shared encoder in one
    pass. Action adapters stay separate because the blocked action
    width is not the same. ``batch`` maps an environment name to the
    collated clip dict (``pixels``, ``action``).
    """
    missing = [name for name in model.env_names if name not in batch]
    if missing:
        raise KeyError(f'batch is missing environments: {missing}')

    pixels = []
    counts = []
    for name in model.env_names:
        sample = batch[name]
        sample['action'] = torch.nan_to_num(sample['action'], 0.0)
        pixels.append(sample['pixels'])
        counts.append(sample['pixels'].shape[0])

    encoded = model.encode({'pixels': torch.cat(pixels, dim=0)})
    emb_all = encoded['emb']

    pred_losses = []
    sig_losses = []
    logs: dict[str, torch.Tensor] = {}
    offset = 0
    for name, count in zip(model.env_names, counts):
        emb = emb_all[offset : offset + count]
        offset += count
        model.set_active_env(name)
        act_emb = model.action_encoder(batch[name]['action'])
        pred_emb = model.predict(
            emb[:, :history_size], act_emb[:, :history_size]
        )
        target = emb[:, num_preds:]
        pred_loss = (pred_emb - target).pow(2).mean()
        sig_loss = sigreg(emb.transpose(0, 1))
        pred_losses.append(pred_loss)
        sig_losses.append(sig_loss)
        env_loss = pred_loss + sigreg_weight * sig_loss
        logs[f'{name}/pred_loss'] = pred_loss.detach()
        logs[f'{name}/sigreg_loss'] = sig_loss.detach()
        logs[f'{name}/loss'] = env_loss.detach()
        for key, value in latent_stats(emb).items():
            logs[f'{name}/{key}'] = value

    pred_loss = torch.stack(pred_losses).mean()
    sig_loss = torch.stack(sig_losses).mean()
    loss = pred_loss + sigreg_weight * sig_loss
    logs['pred_loss'] = pred_loss.detach()
    logs['sigreg_loss'] = sig_loss.detach()
    logs['loss'] = loss.detach()
    return loss, logs


__all__ = [
    'EnvActionBank',
    'GeneralistLeWM',
    'generalist_loss',
    'latent_stats',
]
