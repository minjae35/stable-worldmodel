"""Frozen DINOv2-S/14 visual encoder for the PushT specialist prototype.

This is not an Experiment 1 variant. Action adapter, env embedding, dynamics,
recursive prediction, and the specialist (no residual adapter) stay as they
are. Only the pixels-to-token map changes:

RGB 224 -> frozen DINOv2-S/14 -> drop CLS (16x16 = 256) -> 2x2 average pool
-> 64 tokens -> the existing MLP projector (384 -> 768 -> 192).

Loading uses PreJEPA's ``create_backbone``. PreJEPA dynamics, loss, and
planner are not used.
"""

from __future__ import annotations

import torch
from torch import nn

from rdwm.exp1.model import RDWM

_KEEP_OWN = ('encoder.', 'projector.')


class FrozenDINOv2Encoder(nn.Module):
    """Patch tokens only. ``grid`` is 16 and ``width`` is 384."""

    frozen = True

    def __init__(self):
        super().__init__()
        from stable_worldmodel.wm.prejepa.module import create_backbone

        backbone = create_backbone('dinov2_small')
        patch = int(backbone.config.patch_size)
        hidden = int(backbone.config.hidden_size)
        if patch != 14 or hidden != 384:
            raise RuntimeError(
                f'expected DINOv2-S/14 (patch 14, width 384), got patch {patch} width {hidden}'
            )
        self.backbone = backbone
        self.grid = 224 // patch
        self.width = hidden
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: [B, 3, 224, 224] ImageNet-normalized -> [B, 256, 384]."""
        if pixels.shape[-2:] != (224, 224):
            raise ValueError(f'DINOv2 prototype expects 224x224, got {tuple(pixels.shape[-2:])}')
        chunks = []
        for start in range(0, pixels.shape[0], 32):
            batch = pixels[start : start + 32]
            with torch.no_grad():
                out = self.backbone(batch, interpolate_pos_encoding=True)
                tokens = out.last_hidden_state[:, 1:]
            if tokens.shape[1:] != (self.grid * self.grid, self.width):
                raise RuntimeError(
                    f'expected {(self.grid * self.grid, self.width)} patch tokens, got {tuple(tokens.shape[1:])}'
                )
            chunks.append(tokens)
        return torch.cat(chunks, dim=0)


def build_dino_specialist(cfg: dict, envs: list[str], seed: int, shared_state: dict):
    """PushT specialist A with a frozen DINOv2 encoder.

    Dynamics, action adapter, env embedding, and the prediction head load the
    seed's shared init. The encoder stays at the pretrained DINOv2 weights.
    The projector input is 384, so it keeps its own init.
    """
    if list(envs) != ['pusht']:
        raise ValueError('the DINOv2 prototype trains a PushT specialist only')
    dims = {env: int(cfg['envs'][env]['action_dim']) for env in envs}
    encoder = FrozenDINOv2Encoder()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1_000_003)
        model = RDWM(cfg, 'A', envs, dims, encoder=encoder)
    own = model.state_dict()
    new_state = dict(own)
    loaded = 0
    kept = 0
    for key, value in own.items():
        if key.startswith('action_norm.') or key.startswith(_KEEP_OWN):
            if key.startswith(_KEEP_OWN):
                kept += 1
            continue
        source = shared_state.get(key)
        if source is not None and tuple(source.shape) == tuple(value.shape):
            new_state[key] = source.clone()
            loaded += 1
            continue
        raise RuntimeError(
            f'{key} has no matching shared init '
            f'(shared shape {None if source is None else tuple(source.shape)}, '
            f'model shape {tuple(value.shape)})'
        )
    model.load_state_dict(new_state)
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    return model, {
        'loaded': loaded,
        'own_init_count': kept,
        'own_init_prefixes': list(_KEEP_OWN),
        'encoder': 'frozen_dinov2_small',
    }


def build_dino_ablation(cfg: dict, envs: list[str], seed: int, shared_state: dict):
    """Frozen DINOv2 specialist that may widen the predictor or retie action.

    Keys that still match the canonical shared init are loaded. A wider
    dynamics block, or the concat action projection, keeps its own init.
    The canonical shared-init file is not rewritten.
    """
    if list(envs) != ['pusht']:
        raise ValueError('the DINOv2 ablation trains a PushT specialist only')
    dims = {env: int(cfg['envs'][env]['action_dim']) for env in envs}
    encoder = FrozenDINOv2Encoder()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1_000_003)
        model = RDWM(cfg, 'A', envs, dims, encoder=encoder)
    own = model.state_dict()
    new_state = dict(own)
    loaded, kept = [], []
    own_prefixes = ('encoder.', 'projector.', 'action_embed.', 'input_proj.')
    for key, value in own.items():
        if key.startswith('action_norm.'):
            continue
        source = shared_state.get(key)
        if source is not None and tuple(source.shape) == tuple(value.shape):
            new_state[key] = source.clone()
            loaded.append(key)
            continue
        if key.startswith(own_prefixes) or (
            source is not None and key.startswith('dynamics.')
        ):
            kept.append(key)
            continue
        raise RuntimeError(
            f'{key} has no matching shared init '
            f'(shared shape {None if source is None else tuple(source.shape)}, '
            f'model shape {tuple(value.shape)})'
        )
    model.load_state_dict(new_state)
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    return model, {
        'loaded': len(loaded),
        'own_init_count': len(kept),
        'own_init_sample': kept[:8],
        'encoder': 'frozen_dinov2_small',
        'action_conditioning': model.action_conditioning,
    }
