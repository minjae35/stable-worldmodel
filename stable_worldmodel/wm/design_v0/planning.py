"""Wrap Design v0 as a Stable-WM ``Dynamics`` model for planning.

This adapter does not change the trained core. It only maps Lightning
checkpoints and tensor APIs onto the ``encode`` / ``rollout`` surface
consumed by ``ShootingCostEvaluator``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .core import DesignV0Core
from .objective import recursive_rollout
from .visual_encoder import FrozenVisualEncoder


CORE_STATE_PREFIX = 'objective.core.'


def core_kwargs_from_metadata(
    metadata: dict[str, Any],
) -> dict[str, int]:
    """Return ``DesignV0Core`` kwargs stored in checkpoint metadata."""
    env_names = list(metadata['env_names'])
    if not env_names:
        raise ValueError('Checkpoint metadata is missing env_names')
    return {
        'history_size': int(metadata['history_size']),
        'max_action_dim': int(metadata['max_action_dim']),
        'action_embedding_dim': int(
            metadata['action_embedding_dim']
        ),
        'num_environments': len(env_names),
        'environment_embedding_dim': int(
            metadata['environment_embedding_dim']
        ),
        'dynamics_hidden_dim': int(metadata['dynamics_hidden_dim']),
    }


def build_core_from_metadata(
    metadata: dict[str, Any],
    visual_encoder: FrozenVisualEncoder | None = None,
) -> DesignV0Core:
    """Construct an unweighted core matching checkpoint metadata."""
    if visual_encoder is None:
        visual_encoder = FrozenVisualEncoder.from_pretrained(
            str(metadata['backbone'])
        )
    return DesignV0Core(
        visual_encoder, **core_kwargs_from_metadata(metadata)
    )


def core_state_dict_from_checkpoint(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip the Lightning ``objective.core.`` prefix from weights."""
    core_state = {}
    for key, value in state_dict.items():
        if key.startswith(CORE_STATE_PREFIX):
            core_state[key[len(CORE_STATE_PREFIX) :]] = value
    if not core_state:
        raise ValueError('Checkpoint has no Design v0 core weights')
    return core_state


def load_core_from_checkpoint(
    path: str | Path,
    *,
    visual_encoder: FrozenVisualEncoder | None = None,
    map_location: str | torch.device = 'cpu',
) -> tuple[DesignV0Core, dict[str, Any]]:
    """Restore ``DesignV0Core`` and metadata from a Lightning checkpoint."""
    checkpoint = torch.load(
        path, map_location=map_location, weights_only=False
    )
    metadata = checkpoint.get('design_v0_metadata')
    if metadata is None:
        raise ValueError('Checkpoint is missing design_v0_metadata')
    core = build_core_from_metadata(
        metadata, visual_encoder=visual_encoder
    )
    core.load_state_dict(
        core_state_dict_from_checkpoint(checkpoint['state_dict']),
        strict=True,
    )
    return core, metadata


def env_id_from_metadata(
    metadata: dict[str, Any], environment: str
) -> int:
    """Return ``env_to_id[environment]`` from checkpoint metadata."""
    mapping = metadata.get('env_to_id') or {}
    if environment not in mapping:
        raise KeyError(
            f'{environment!r} is not in checkpoint env_to_id '
            f'{sorted(mapping)}'
        )
    return int(mapping[environment])


def effective_action_dim_from_metadata(
    metadata: dict[str, Any], environment: str
) -> int:
    """Return the env's training-time effective action width."""
    dims = metadata.get('effective_action_dims') or {}
    if environment in dims:
        return int(dims[environment])
    datasets = metadata.get('datasets') or {}
    spec = datasets.get(environment) or {}
    for key in ('effective_action_dim', 'action_block_dim'):
        if spec.get(key) is not None:
            return int(spec[key])
    mapping = metadata.get('env_to_id') or {}
    if len(mapping) == 1 and metadata.get('max_action_dim') is not None:
        return int(metadata['max_action_dim'])
    raise KeyError(
        f'No effective action dim for {environment!r} in metadata'
    )


def resolve_planning_environment(
    metadata: dict[str, Any],
    environment: str | None = None,
) -> tuple[str, int, int]:
    """Return ``(name, env_id, action_dim)`` for a planning environment."""
    mapping = metadata.get('env_to_id') or {}
    if environment is None:
        if len(mapping) != 1:
            raise ValueError(
                'environment is required when the checkpoint has '
                f'{len(mapping)} environments'
            )
        environment = str(next(iter(mapping)))
    env_id = env_id_from_metadata(metadata, environment)
    action_dim = effective_action_dim_from_metadata(
        metadata, environment
    )
    return environment, env_id, action_dim


def load_planning_adapter(
    path: str | Path,
    *,
    visual_encoder: FrozenVisualEncoder | None = None,
    map_location: str | torch.device = 'cpu',
    environment: str | None = None,
) -> DesignV0PlanningAdapter:
    """Load a planning adapter from a Design v0 Lightning checkpoint."""
    core, metadata = load_core_from_checkpoint(
        path,
        visual_encoder=visual_encoder,
        map_location=map_location,
    )
    _, env_id, action_dim = resolve_planning_environment(
        metadata, environment
    )
    adapter = DesignV0PlanningAdapter(
        core,
        default_env_id=env_id,
        action_dim=action_dim,
    )
    adapter.metadata = metadata
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


class DesignV0PlanningAdapter(nn.Module):
    """``Dynamics`` wrapper around a frozen Design v0 core.

    CEM candidates use the current environment's effective action width.
    Joint checkpoints pad that vector to ``max_action_dim`` with the
    same trailing-zero / leading-True mask used in training.
    """

    def __init__(
        self,
        core: DesignV0Core,
        *,
        default_env_id: int = 0,
        action_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.default_env_id = int(default_env_id)
        if action_dim is None:
            action_dim = core.action_encoder.max_action_dim
        self.action_dim = int(action_dim)
        self.metadata: dict[str, Any] | None = None

    def encode(self, info: dict) -> dict:
        """Encode ``pixels`` ``(B, T, C, H, W)`` into ``emb`` ``(B, T, D)``."""
        if 'pixels' not in info:
            raise KeyError('pixels not in info_dict')
        info['emb'] = self.core.encode(info['pixels'])
        return info

    def rollout(
        self, info: dict, action_candidates: torch.Tensor
    ) -> dict:
        """Roll future action blocks and write ``predicted_emb``.

        ``pixels`` is ``(B, S, H, C, h, w)``. Candidates are strictly
        future blocks ``(B, S, T, A)``. ``predicted_emb`` is
        ``(B, S, H + T, D)``: encoded context followed by H-step-free
        recursive latent predictions.
        """
        if 'pixels' not in info:
            raise KeyError('pixels not in info_dict')
        context_len = int(info['pixels'].size(2))
        batch, samples, horizon = action_candidates.shape[:3]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_candidates.new_zeros(
                batch,
                samples,
                0,
                action_candidates.size(-1),
            )
        if int(act_past.size(2)) != context_len - 1:
            raise ValueError(
                'action_history must hold H-1='
                f'{context_len - 1} executed blocks, '
                f'got {int(act_past.size(2))}'
            )

        if 'emb' not in info:
            init = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value)
            }
            init = self.encode(init)
            info['emb'] = (
                init['emb']
                .detach()
                .unsqueeze(1)
                .expand(batch, samples, -1, -1)
                .contiguous()
            )

        flat_context = info['emb'].reshape(
            batch * samples, context_len, info['emb'].size(-1)
        )
        flat_action = action_candidates.reshape(
            batch * samples, horizon, action_candidates.size(-1)
        )
        history = self._history_latents(flat_context)
        action, action_mask = self._action_and_mask(flat_action)
        env_id = self._batch_env_id(
            info,
            batch_size=flat_context.shape[0],
            device=flat_context.device,
        )
        predicted = recursive_rollout(
            self.core, history, action, action_mask, env_id
        )
        predicted_emb = torch.cat([flat_context, predicted], dim=1)
        info['predicted_emb'] = predicted_emb.reshape(
            batch, samples, context_len + horizon, predicted_emb.size(-1)
        )
        return info

    def _history_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Build a K-frame history, left-padding short context."""
        history_size = self.core.history_size
        context_len = latents.shape[1]
        if context_len == history_size:
            return latents
        if context_len > history_size:
            return latents[:, -history_size:]
        pad = latents[:, :1].repeat(1, history_size - context_len, 1)
        return torch.cat([pad, latents], dim=1)

    def _action_and_mask(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad env actions to ``max_action_dim`` with a boolean mask.

        Matches ``MultiEnvironmentDataset``: zeros on the trailing pad,
        ``True`` only on the first ``action_dim`` features.
        """
        max_action_dim = self.core.action_encoder.max_action_dim
        action_dim = self.action_dim
        if action_dim <= 0 or action_dim > max_action_dim:
            raise ValueError(
                'action_dim must be in 1..max_action_dim '
                f'{max_action_dim}, got {action_dim}'
            )
        last = int(action.shape[-1])
        if last not in (action_dim, max_action_dim):
            raise ValueError(
                'action last dim must be '
                f'{action_dim} or {max_action_dim}, got {last}'
            )
        padded = action.new_zeros(*action.shape[:-1], max_action_dim)
        padded[..., :action_dim] = action[..., :action_dim]
        action_mask = torch.zeros(
            padded.shape, dtype=torch.bool, device=action.device
        )
        action_mask[..., :action_dim] = True
        return padded, action_mask

    def _batch_env_id(
        self,
        info: dict,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        env_id = info.get('env_id')
        if env_id is None:
            return torch.full(
                (batch_size,),
                self.default_env_id,
                dtype=torch.long,
                device=device,
            )
        if not torch.is_tensor(env_id):
            env_id = torch.as_tensor(env_id)
        return env_id.reshape(batch_size).to(
            device=device, dtype=torch.long
        )
