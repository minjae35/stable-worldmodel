"""Per-seed shared initialization and optimizer setup.

Every variant of a seed starts from one canonical state: a dense B model
over the 7 canonical envs, built on CPU under ``torch.manual_seed(seed)``.
A variant loads every key with the same name and shape. The only keys a
variant may keep from its own construction are C's residual adapters
(up projection zero) and D's single-token spatial position embedding.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import torch
from torch import nn

from rdwm.exp1.config import env_order, shared_dir
from rdwm.exp1.model import RDWM

_OWN_KEYS = ('dynamics.adapters.',)
_POOLED_OWN = ('dynamics.spatial_pos',)


def _action_dims(cfg: dict, envs: list[str]) -> dict[str, int]:
    return {env: int(cfg['envs'][env]['action_dim']) for env in envs}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _shareable(state: dict) -> dict:
    return {k: v for k, v in state.items() if not k.startswith('action_norm.')}


def canonical_shared_state(cfg: dict, seed: int) -> dict[str, torch.Tensor]:
    envs = env_order(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RDWM(cfg, 'B', envs, _action_dims(cfg, envs))
    return _shareable(model.state_dict())


def shared_init(cfg: dict, seed: int) -> tuple[dict, str]:
    """Load (or create once) ``shared/seed<k>/shared_init.pt`` and its hash.

    Rebuilding is deterministic, so the stored file is a check: a job that
    rebuilds a different state refuses to run.
    """
    state = canonical_shared_state(cfg, seed)
    digest = state_hash(state)
    folder = shared_dir(cfg, seed)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'shared_init.pt'
    hash_path = folder / 'shared_init.sha256'
    if hash_path.exists():
        stored = hash_path.read_text().strip()
        if stored != digest:
            raise RuntimeError(
                f'shared init for seed {seed} rebuilt with hash {digest[:12]}, '
                f'stored {stored[:12]}. Torch version or model code changed.'
            )
    else:
        tmp = path.with_suffix(f'.tmp{os.getpid()}')
        torch.save(state, tmp)
        os.replace(tmp, path)
        tmp_hash = hash_path.with_suffix(f'.tmp{os.getpid()}')
        tmp_hash.write_text(digest + '\n')
        os.replace(tmp_hash, hash_path)
    return state, digest


def build_model(
    cfg: dict,
    variant: str,
    envs: list[str],
    seed: int,
    shared_state: dict | None = None,
) -> tuple[RDWM, dict]:
    """Construct a variant and load the seed's shared initialization."""
    if shared_state is None:
        shared_state = canonical_shared_state(cfg, seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1_000_003)
        model = RDWM(cfg, variant, envs, _action_dims(cfg, envs))
    own = model.state_dict()
    allowed_own = _OWN_KEYS + (_POOLED_OWN if model.pooled else ())
    loaded, kept = [], []
    new_state = dict(own)
    for key, value in own.items():
        if key.startswith('action_norm.'):
            continue
        source = shared_state.get(key)
        if source is not None and source.shape == value.shape:
            new_state[key] = source.clone()
            loaded.append(key)
        elif key.startswith(allowed_own):
            kept.append(key)
        else:
            raise RuntimeError(
                f'variant {variant}: key {key} has no matching shared init '
                f'(shared shape {None if source is None else tuple(source.shape)}, '
                f'model shape {tuple(value.shape)})'
            )
    model.load_state_dict(new_state)
    return model, {'loaded': len(loaded), 'own_init': kept}


def shared_subset_hash(model: RDWM) -> str:
    """Hash of parameters every B/C pair must share (everything but adapters)."""
    state = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith(_OWN_KEYS)
    }
    return state_hash(state)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on everything except biases and normalization parameters."""
    no_decay: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters(recurse=False):
                no_decay.add(id(param))
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('.bias') or id(param) in no_decay:
            plain.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': plain, 'weight_decay': 0.0},
    ]


def make_optimizer(model: nn.Module, cfg: dict, total_updates: int):
    tcfg = cfg['train']
    opt = torch.optim.AdamW(
        param_groups(model, tcfg['weight_decay']), lr=tcfg['lr']
    )
    warmup = max(1, int(round(tcfg['warmup_frac'] * total_updates)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: lr_factor(step, warmup, total_updates)
    )
    return opt, sched


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup over ``warmup`` updates, then step-wise cosine to 0."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(path: Path, model, opt, sched, update: int, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.tmp{os.getpid()}')
    torch.save(
        {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'update': update,
            'meta': meta,
        },
        tmp,
    )
    os.replace(tmp, path)
