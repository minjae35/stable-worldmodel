"""Environment-agnostic data and optimization utilities for Stage-3 G2.

The training path in this module may inspect episode ids, pixels, and actions.
It must not inspect state, masks, rewards, contacts, or event labels. Those are
reserved for :mod:`stable_worldmodel.wm.spatial.g2_eval`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from stable_worldmodel.wm.spatial.data import pack_blocked_action

ENV_NAMES = ('pusht', 'tworoom', 'cube')
RAW_ACTION_DIMS = {'pusht': 2, 'tworoom': 2, 'cube': 5}
TRAINING_COLUMNS = ('episode_idx', 'pixels', 'action')
FORBIDDEN_TRAINING_TERMS = (
    'state',
    'mask',
    'reward',
    'contact',
    'event',
    'success',
    'target',
    'privileged',
    'proprio',
)
ALLOWED_CLIP_FIELDS = {
    'episode',
    'start',
    'row',
    'length',
    'normalized_time',
    'quartile',
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    sampling: str
    loss: str


ARM_SPECS = {
    'A': ArmSpec('A', 's0', 'mse'),
    'B': ArmSpec('B', 's1', 'mse'),
    'C': ArmSpec('C', 's0', 'l1'),
    'D': ArmSpec('D', 's1', 'l1'),
}


def arm_spec(name: str) -> ArmSpec:
    try:
        return ARM_SPECS[name.upper()]
    except KeyError as exc:
        raise ValueError(f'unknown G2 arm {name!r}; expected A, B, C, or D') from exc


def assert_label_free_columns(columns) -> tuple[str, ...]:
    columns = tuple(str(column) for column in columns)
    bad = [
        column
        for column in columns
        if any(term in column.lower() for term in FORBIDDEN_TRAINING_TERMS)
    ]
    if bad:
        raise ValueError(f'training requested forbidden columns: {bad}')
    return columns


def validate_manifest(document: dict) -> dict:
    """Validate split isolation and the label-free clip schema."""
    frameskip = int(document['frameskip'])
    span = int(document['span'])
    clips_per_episode = int(document['clips_per_episode'])
    if frameskip < 1 or span < frameskip or clips_per_episode < 1:
        raise ValueError('invalid frameskip/span/clips_per_episode')
    summary = {}
    for env in ENV_NAMES:
        body = document[env]
        split = {name: [int(x) for x in body['split'][name]] for name in ('train', 'val', 'test')}
        sets = {name: set(values) for name, values in split.items()}
        if sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test']:
            raise ValueError(f'{env} episode split leakage')
        for sampling in ('s0', 's1'):
            for split_name in ('train', 'val', 'test'):
                clips = body[sampling][split_name]
                expected = len(split[split_name]) * clips_per_episode
                if len(clips) != expected:
                    raise ValueError(
                        f'{env}/{sampling}/{split_name} has {len(clips)} clips, expected {expected}'
                    )
                counts = {episode: 0 for episode in split[split_name]}
                for clip in clips:
                    extra = set(clip) - ALLOWED_CLIP_FIELDS
                    if extra:
                        raise ValueError(
                            f'{env}/{sampling}/{split_name} has forbidden clip fields {sorted(extra)}'
                        )
                    episode = int(clip['episode'])
                    if episode not in counts:
                        raise ValueError(f'{env} clip episode {episode} is outside {split_name}')
                    start = int(clip['start'])
                    length = int(clip['length'])
                    if start < 0 or start + span > length:
                        raise ValueError(f'{env} clip crosses an episode boundary')
                    if int(clip['row']) < 0:
                        raise ValueError(f'{env} clip has a negative row')
                    counts[episode] += 1
                if set(counts.values()) != {clips_per_episode}:
                    raise ValueError(f'{env}/{sampling}/{split_name} is not balanced by episode')
        summary[env] = {
            'episodes': {name: len(values) for name, values in split.items()},
            'clips': {
                sampling: {
                    name: len(body[sampling][name])
                    for name in ('train', 'val', 'test')
                }
                for sampling in ('s0', 's1')
            },
        }
    return summary


def clip_rows(clip: dict, frameskip: int) -> tuple[int, int, list[int]]:
    """Return context, target, and causal action rows for one transition."""
    context = int(clip['row'])
    target = context + int(frameskip)
    actions = list(range(context, target))
    if not actions or actions[-1] != target - 1 or target in actions:
        raise RuntimeError('action block is not causally aligned')
    return context, target, actions


def decode_rgb(blob) -> np.ndarray:
    from PIL import Image

    raw = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    with Image.open(io.BytesIO(raw)) as image:
        value = np.asarray(image.convert('RGB'))
    if value.shape != (224, 224, 3):
        raise ValueError(f'expected 224x224 RGB, got {value.shape}')
    return value


def _take(dataset, rows: list[int], columns: tuple[str, ...], batch: int = 512):
    unique = sorted(set(int(row) for row in rows))
    values = {name: [] for name in columns}
    for start in range(0, len(unique), batch):
        chunk = unique[start : start + batch]
        table = dataset.take(chunk, columns=list(columns))
        for name in columns:
            values[name].extend(table[name].to_pylist())
    return unique, {
        row: {name: values[name][index] for name in columns}
        for index, row in enumerate(unique)
    }


def load_label_free_clips(
    dataset,
    clips: list[dict],
    env: str,
    frameskip: int,
) -> dict:
    """Read real clips without requesting environment-specific labels."""
    columns = assert_label_free_columns(TRAINING_COLUMNS)
    rows = []
    plans = []
    for clip in clips:
        context, target, actions = clip_rows(clip, frameskip)
        plans.append((clip, context, target, actions))
        rows.extend([context, target, *actions])
    _, by_row = _take(dataset, rows, columns)
    images_context = []
    images_target = []
    blocked_actions = []
    for clip, context, target, actions in plans:
        expected_episode = int(clip['episode'])
        observed = {int(by_row[row]['episode_idx']) for row in [context, target, *actions]}
        if observed != {expected_episode}:
            raise RuntimeError(
                f'{env} clip crosses or mismatches episode: expected {expected_episode}, got {observed}'
            )
        raw = np.stack(
            [np.asarray(by_row[row]['action'], dtype=np.float32).reshape(-1) for row in actions]
        )
        expected_shape = (frameskip, RAW_ACTION_DIMS[env])
        if raw.shape != expected_shape:
            raise RuntimeError(f'{env} action shape {raw.shape}, expected {expected_shape}')
        blocked_actions.append(pack_blocked_action(raw, 1, frameskip)[0])
        images_context.append(decode_rgb(by_row[context]['pixels']))
        images_target.append(decode_rgb(by_row[target]['pixels']))
    return {
        'images_context': np.stack(images_context),
        'images_target': np.stack(images_target),
        'actions': np.stack(blocked_actions).astype(np.float32),
        'episode': np.asarray([int(clip['episode']) for clip in clips], dtype=np.int64),
        'start': np.asarray([int(clip['start']) for clip in clips], dtype=np.int64),
    }


def make_index_schedule(
    n_items: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    """Deterministic epoch-shuffle schedule shared by every arm/environment."""
    if n_items < batch_size or batch_size < 1 or steps < 1:
        raise ValueError('schedule needs n_items >= batch_size >= 1 and steps >= 1')
    rng = np.random.default_rng(seed)
    stream = []
    needed = batch_size * steps
    while len(stream) < needed:
        stream.extend(rng.permutation(n_items).tolist())
    return np.asarray(stream[:needed], dtype=np.int64).reshape(steps, batch_size)


def schedule_sha256(schedule: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(schedule).tobytes()).hexdigest()


def spatial_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    scale: float = 1.0,
) -> torch.Tensor:
    target = target.detach()
    if kind == 'mse':
        raw = (prediction - target).pow(2).mean()
    elif kind == 'l1':
        raw = (prediction - target).abs().mean()
    else:
        raise ValueError(f'unknown loss {kind!r}')
    return raw * float(scale)


def grad_norm(parameters) -> float:
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    return 0.0 if total is None else float(total.sqrt())


def calibrate_loss_scales(model, batches: list[tuple], reference: str = 'mse') -> dict:
    """Match raw initial gradient norms on one common, label-free batch."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norms = {}
    was_training = model.training
    model.train()
    for kind in ('mse', 'l1'):
        model.zero_grad(set_to_none=True)
        losses = []
        for tokens, actions, env_id, target in batches:
            prediction = model(tokens, actions, env_id)
            losses.append(spatial_prediction_loss(prediction, target, kind))
        torch.stack(losses).mean().backward()
        norms[kind] = grad_norm(parameters)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    if min(norms.values()) <= 0 or not all(np.isfinite(list(norms.values()))):
        raise RuntimeError(f'invalid calibration gradient norms: {norms}')
    scales = {kind: norms[reference] / value for kind, value in norms.items()}
    return {'raw_grad_norm': norms, 'scale': scales, 'reference': reference}


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(state[name].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def save_checkpoint_no_overwrite(
    path: Path,
    *,
    model,
    optimizer,
    step: int,
    run_config: dict,
    extra: dict | None = None,
):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite checkpoint {path}')
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': int(step),
        'config_sha256': config_sha256(run_config),
        'extra': {} if extra is None else extra,
    }
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_resume_checkpoint(path: Path, model, optimizer, run_config: dict) -> dict:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected = config_sha256(run_config)
    if payload.get('config_sha256') != expected:
        raise ValueError('resume checkpoint config does not match this run')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(parameter.device)
    return payload


def fairness_signature(base: dict, spec: ArmSpec) -> dict:
    """Return the conditions that must match apart from sampling and loss."""
    ignored = {'arm', 'name', 'sampling', 'loss', 'output_dir'}
    return {key: value for key, value in base.items() if key not in ignored}


def arm_config(base: dict, name: str) -> dict:
    spec = arm_spec(name)
    config = dict(base)
    config.update(asdict(spec))
    return config
