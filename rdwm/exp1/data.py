"""Experiment 1 data: local Lance clips, episode split, action stats, and
the per-seed training sample schedule (experiment-1.md Section 4).

A clip is ``(episode, start)``: observations at native steps
``start + 5k`` (k = 0..4) and action blocks ``actions[start+5k : start+5k+5]``
(k = 0..3). Actions are outgoing (action[t] leaves state t), as validated in
the Phase 1 gate, so no shift is applied here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCHEDULE_FILES = ('env', 'history', 'episode', 'start')


# ---------------------------------------------------------------- Lance I/O


def _column_numpy(table, name: str) -> np.ndarray:
    import pyarrow as pa

    col = table.column(name).combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        dim = col.type.list_size
        flat = col.flatten().to_numpy(zero_copy_only=False)
        out = flat.astype(np.float32).reshape(len(col), dim)
        if col.null_count:
            out[np.asarray(col.is_null())] = np.nan
        return out
    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
        rows = col.to_pylist()
        dim = max(len(r) for r in rows if r is not None)
        out = np.full((len(rows), dim), np.nan, dtype=np.float32)
        for i, r in enumerate(rows):
            if r is not None:
                out[i, : len(r)] = np.asarray(
                    [np.nan if v is None else v for v in r], dtype=np.float32
                )
        return out
    return col.to_numpy(zero_copy_only=False)


def _file_key(path: str) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:10]


class EnvIndex:
    """Episode structure and actions of one local Lance table, cached as npy."""

    def __init__(self, env: str, lance_path: str, cache_root: str | Path):
        self.env = env
        self.lance_path = lance_path
        folder = Path(cache_root) / 'index' / f'{env}_{_file_key(lance_path)}'
        self.folder = folder
        if not (folder / 'done').exists():
            self._build(folder)
        self.offsets = np.load(folder / 'offsets.npy')
        self.lengths = np.load(folder / 'lengths.npy')
        self.actions = np.load(folder / 'actions.npy', mmap_mode='r')

    def _build(self, folder: Path) -> None:
        import lance

        ds = lance.dataset(self.lance_path)
        table = ds.to_table(columns=['episode_idx', 'step_idx', 'action'])
        ep = np.asarray(table.column('episode_idx').to_numpy(), dtype=np.int64)
        step = np.rint(
            np.asarray(table.column('step_idx').to_numpy(), dtype=np.float64)
        ).astype(np.int64)
        if len(ep) > 1 and (np.diff(ep) < 0).any():
            raise ValueError(f'{self.env}: episode_idx is not contiguous')
        change = np.flatnonzero(np.diff(ep) != 0) + 1
        offsets = np.concatenate([[0], change]).astype(np.int64)
        lengths = np.diff(np.concatenate([offsets, [len(ep)]])).astype(np.int64)
        expect = np.concatenate([np.arange(n) for n in lengths])
        if not np.array_equal(step, expect):
            raise ValueError(f'{self.env}: step_idx is not 0..T-1 per episode')
        actions = _column_numpy(table, 'action')
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'offsets.npy', offsets)
        np.save(folder / 'lengths.npy', lengths)
        np.save(folder / 'actions.npy', actions)
        (folder / 'done').write_text(self.lance_path + '\n')

    @property
    def num_episodes(self) -> int:
        return int(len(self.lengths))

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])


def split_episodes(num_episodes: int, split_cfg: dict) -> dict[str, np.ndarray]:
    """Episode-level split. Clips never cross split boundaries because a clip
    lives inside one episode."""
    if split_cfg['order'] != 'contiguous':
        raise ValueError(f"unsupported split order {split_cfg['order']!r}")
    n_val = int(round(split_cfg['val_frac'] * num_episodes))
    n_test = int(round(split_cfg['test_frac'] * num_episodes))
    n_train = num_episodes - n_val - n_test
    ids = np.arange(num_episodes)
    return {
        'train': ids[:n_train],
        'val': ids[n_train : n_train + n_val],
        'test': ids[n_train + n_val :],
    }


def clip_table(
    index: EnvIndex, episodes: np.ndarray, num_obs: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """All valid clips in ``episodes``: the last observation stays inside the
    episode and every action of the 4 blocks is finite."""
    span = (num_obs - 1) * stride
    finite = np.isfinite(np.asarray(index.actions)).all(axis=1)
    eps, starts = [], []
    for ep in episodes:
        off, length = int(index.offsets[ep]), int(index.lengths[ep])
        if length <= span:
            continue
        ok = finite[off : off + length].astype(np.int64)
        csum = np.concatenate([[0], np.cumsum(ok)])
        cand = np.arange(0, length - span)
        good = (csum[cand + span] - csum[cand]) == span
        cand = cand[good]
        eps.append(np.full(len(cand), ep, dtype=np.int32))
        starts.append(cand.astype(np.int32))
    if not eps:
        raise RuntimeError(f'{index.env}: no valid clips')
    return np.concatenate(eps), np.concatenate(starts)


def action_stats(
    index: EnvIndex, train_episodes: np.ndarray, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std over finite actions of train-split episodes only."""
    rows = np.concatenate(
        [
            np.arange(index.offsets[e], index.offsets[e] + index.lengths[e])
            for e in train_episodes
        ]
    )
    acts = np.asarray(index.actions[rows], dtype=np.float64)
    acts = acts[np.isfinite(acts).all(axis=1)]
    mean = acts.mean(axis=0)
    std = np.maximum(acts.std(axis=0), floor)
    return mean.astype(np.float32), std.astype(np.float32)


class EnvData:
    """Everything training needs for one env, derived from config + Lance."""

    def __init__(self, cfg: dict, env: str, lance_path: str):
        self.env = env
        self.index = EnvIndex(env, lance_path, cfg['paths']['cache_root'])
        expect = int(cfg['envs'][env]['action_dim'])
        if self.index.action_dim != expect:
            raise ValueError(
                f'{env}: dataset action dim {self.index.action_dim} != '
                f'config {expect}'
            )
        dcfg = cfg['data']
        self.splits = split_episodes(self.index.num_episodes, dcfg['split'])
        self.clip_ep, self.clip_start = clip_table(
            self.index, self.splits['train'], dcfg['num_obs'], dcfg['obs_stride']
        )
        self.action_mean, self.action_std = action_stats(
            self.index, self.splits['train'], dcfg['action_std_floor']
        )

    def summary(self) -> dict:
        return {
            'episodes': self.index.num_episodes,
            'split_sizes': {k: int(len(v)) for k, v in self.splits.items()},
            'train_clips': int(len(self.clip_ep)),
            'action_mean': self.action_mean.tolist(),
            'action_std': self.action_std.tolist(),
        }


# ----------------------------------------------------------------- schedule


def build_schedule(
    seed: int,
    envs: list[str],
    clip_counts: dict[str, int],
    updates: int,
    batch_size: int,
    history_probs: dict[int, float],
) -> dict[str, np.ndarray]:
    """Fixed per-seed training sample schedule.

    Generalists use rounds of ``len(envs)`` updates; each round visits every
    env once in a shuffled order. Each update draws one history length and
    ``batch_size`` distinct train clips from that env. Specialists (one env)
    reduce to one env per update.
    """
    if updates % len(envs):
        raise ValueError(f'{updates} updates is not a multiple of {len(envs)} envs')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
    hist_values = np.array(sorted(history_probs, reverse=True))
    hist_p = np.array([history_probs[h] for h in hist_values])
    env_seq = np.concatenate(
        [rng.permutation(len(envs)) for _ in range(updates // len(envs))]
    ).astype(np.int16)
    history = rng.choice(hist_values, size=updates, p=hist_p).astype(np.int8)
    clips = np.empty((updates, batch_size), dtype=np.int64)
    for i, env_id in enumerate(env_seq):
        clips[i] = rng.choice(clip_counts[envs[env_id]], batch_size, replace=False)
    return {'env': env_seq, 'history': history, 'clip': clips}


def resolve_schedule(
    schedule: dict[str, np.ndarray], envs: list[str], data: dict[str, EnvData]
) -> dict[str, np.ndarray]:
    """Clip indices -> ``(episode, start)`` clip IDs."""
    episode = np.empty_like(schedule['clip'], dtype=np.int32)
    start = np.empty_like(schedule['clip'], dtype=np.int32)
    for env_id, env in enumerate(envs):
        rows = schedule['env'] == env_id
        idx = schedule['clip'][rows]
        episode[rows] = data[env].clip_ep[idx]
        start[rows] = data[env].clip_start[idx]
    return {
        'env': schedule['env'],
        'history': schedule['history'],
        'episode': episode,
        'start': start,
    }


def restrict_schedule(
    schedule: dict[str, np.ndarray], envs: list[str], env: str
) -> dict[str, np.ndarray]:
    """Specialist A schedule: the env's own batches from the generalist
    schedule, in the same order, so per-env sample exposure matches."""
    rows = schedule['env'] == envs.index(env)
    out = {k: v[rows].copy() for k, v in schedule.items()}
    out['env'] = np.zeros(int(rows.sum()), dtype=np.int16)
    return out


def schedule_digest(schedule: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in SCHEDULE_FILES:
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(schedule[key]).tobytes())
    return digest.hexdigest()


def save_schedule(folder: Path, schedule: dict, meta: dict) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    digest = schedule_digest(schedule)
    for key in SCHEDULE_FILES:
        tmp = folder / f'{key}.tmp{os.getpid()}.npy'
        np.save(tmp, schedule[key])
        os.replace(tmp, folder / f'{key}.npy')
    meta = {**meta, 'sha256': digest}
    tmp = folder / f'meta.tmp{os.getpid()}.json'
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, folder / 'meta.json')
    return digest


def load_schedule(folder: Path, mmap: bool = True) -> dict[str, np.ndarray]:
    mode = 'r' if mmap else None
    return {k: np.load(folder / f'{k}.npy', mmap_mode=mode) for k in SCHEDULE_FILES}


# ------------------------------------------------------------------ batches


def preprocess(pixels: torch.Tensor, cfg: dict) -> torch.Tensor:
    """uint8 [..., 3, h, w] -> float [..., 3, 224, 224], ImageNet-normalized.

    The same function serves training clips, live observations, and goals.
    64x64 sources (Scene, AntMaze) are bilinearly resized to 224.
    """
    size = cfg['model']['image_size']
    lead = pixels.shape[:-3]
    x = pixels.reshape(-1, *pixels.shape[-3:]).float().div_(255.0)
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    mean = torch.tensor(cfg['data']['imagenet_mean'], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg['data']['imagenet_std'], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.reshape(*lead, 3, size, size)


def decode_jpegs(blobs) -> torch.Tensor:
    from torchvision.io import ImageReadMode, decode_jpeg

    tensors = [
        torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in blobs
    ]
    return torch.stack(decode_jpeg(tensors, mode=ImageReadMode.RGB))


class ScheduleBatches(torch.utils.data.Dataset):
    """Item i = the full batch of schedule update ``first + i``.

    Pickles only paths, so it works with spawn workers. Each worker opens its
    own Lance handles and memory-maps the cached actions and schedule.
    """

    def __init__(
        self,
        envs: list[str],
        lance_paths: dict[str, str],
        index_dirs: dict[str, str],
        schedule_dir: str,
        first: int,
        num_obs: int,
        stride: int,
        action_block: int,
    ):
        self.envs = list(envs)
        self.lance_paths = dict(lance_paths)
        self.index_dirs = dict(index_dirs)
        self.schedule_dir = str(schedule_dir)
        self.first = int(first)
        self.num_obs = num_obs
        self.stride = stride
        self.action_block = action_block
        self._state = None
        sched = load_schedule(Path(schedule_dir))
        self.total = int(len(sched['env']))

    def __len__(self) -> int:
        return self.total - self.first

    def _open(self):
        import lance

        sched = load_schedule(Path(self.schedule_dir))
        tables = {e: lance.dataset(self.lance_paths[e]) for e in self.envs}
        offsets = {e: np.load(Path(self.index_dirs[e]) / 'offsets.npy') for e in self.envs}
        actions = {
            e: np.load(Path(self.index_dirs[e]) / 'actions.npy', mmap_mode='r')
            for e in self.envs
        }
        self._state = (sched, tables, offsets, actions)

    def __getitem__(self, i: int) -> dict:
        if self._state is None:
            self._open()
        sched, tables, offsets, actions = self._state
        u = self.first + int(i)
        env = self.envs[int(sched['env'][u])]
        eps = np.asarray(sched['episode'][u], dtype=np.int64)
        starts = np.asarray(sched['start'][u], dtype=np.int64)
        base = offsets[env][eps] + starts
        obs_rows = base[:, None] + self.stride * np.arange(self.num_obs)[None]
        table = tables[env].take(obs_rows.reshape(-1).tolist(), columns=['pixels'])
        frames = decode_jpegs(table.column('pixels').to_pylist())
        b = len(eps)
        pixels = frames.reshape(b, self.num_obs, *frames.shape[1:])
        act_rows = base[:, None] + np.arange((self.num_obs - 1) * self.stride)[None]
        acts = np.asarray(actions[env][act_rows.reshape(-1)], dtype=np.float32)
        acts = acts.reshape(b, self.num_obs - 1, self.action_block, -1)
        if not np.isfinite(acts).all():
            raise RuntimeError(f'{env}: non-finite action in update {u}')
        return {
            'update': u,
            'env': env,
            'history': int(sched['history'][u]),
            'pixels': pixels,
            'actions': torch.from_numpy(acts),
            'episode': torch.from_numpy(eps),
            'start': torch.from_numpy(starts),
        }
