"""Equal-count batches across environments with different action widths.

A single stacked batch cannot hold PushT/TwoRoom actions and Cube
actions together. Each yielded batch is a dict of per-environment
collated clips, with the same number of clips from each environment.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
from torch.utils.data import Dataset, Sampler, default_collate


class EnvTaggedDataset(Dataset):
    """Tag every clip with the environment it came from."""

    def __init__(self, dataset, env_name: str):
        self.dataset = dataset
        self.env_name = env_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        sample['env_name'] = self.env_name
        return sample


class ConcatEnvDataset(Dataset):
    """Concatenate tagged datasets without mixing their columns."""

    def __init__(self, datasets: Mapping[str, Dataset]):
        if not datasets:
            raise ValueError('Need at least one environment dataset')
        self.names = list(datasets)
        self.datasets = dict(datasets)
        self.offsets: dict[str, int] = {}
        cursor = 0
        for name in self.names:
            self.offsets[name] = cursor
            cursor += len(self.datasets[name])
        self._length = cursor

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        for name in self.names:
            start = self.offsets[name]
            end = start + len(self.datasets[name])
            if start <= idx < end:
                return self.datasets[name][idx - start]
        raise IndexError(idx)

    def env_lengths(self) -> dict[str, int]:
        return {name: len(self.datasets[name]) for name in self.names}


class BalancedEnvBatchSampler(Sampler[list[int]]):
    """Draw the same number of clips from each environment per batch.

    One epoch contains ``min(length) // per_env`` batches. That is one
    pass over the shortest split. Longer splits contribute a random
    subset of the same size, so the step loss is not dominated by the
    largest table.
    """

    def __init__(
        self,
        lengths: Mapping[str, int],
        per_env: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        if per_env < 1:
            raise ValueError(f'per_env must be >= 1, got {per_env}')
        if world_size < 1:
            raise ValueError(f'world_size must be >= 1, got {world_size}')
        if not 0 <= rank < world_size:
            raise ValueError(
                f'rank {rank} is outside world_size {world_size}'
            )
        self.names = list(lengths)
        self.lengths = {name: int(lengths[name]) for name in self.names}
        # per_env is the clip count on this rank. The step consumes
        # per_env * world_size clips from each environment.
        self.per_env = int(per_env)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_per_env = self.per_env * self.world_size
        self.seed = int(seed)
        self._epoch = 0
        offsets: dict[str, int] = {}
        cursor = 0
        for name in self.names:
            offsets[name] = cursor
            cursor += self.lengths[name]
        self.offsets = offsets
        shortest = min(self.lengths.values())
        if shortest < self.global_per_env:
            raise ValueError(
                f'global batch per_env*world={self.global_per_env} is '
                f'larger than the shortest split ({shortest}). '
                f'lengths={self.lengths}'
            )
        self._steps = shortest // self.global_per_env

    def __len__(self) -> int:
        return self._steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        pools = {
            name: rng.permutation(length)
            for name, length in self.lengths.items()
        }
        cursors = {name: 0 for name in self.names}
        span = self.global_per_env
        local = slice(self.rank * self.per_env, (self.rank + 1) * self.per_env)
        for _ in range(self._steps):
            batch: list[int] = []
            for name in self.names:
                cursor = cursors[name]
                length = self.lengths[name]
                if cursor + span > length:
                    pools[name] = rng.permutation(length)
                    cursor = 0
                chosen = pools[name][cursor : cursor + span]
                cursors[name] = cursor + span
                mine = chosen[local]
                base = self.offsets[name]
                batch.extend(base + int(index) for index in mine)
            yield batch


def collate_by_env(samples: list[dict]) -> dict[str, dict]:
    """Group a balanced batch into one tensor dict per environment."""
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        name = sample['env_name']
        item = {
            key: value
            for key, value in sample.items()
            if key != 'env_name'
        }
        grouped.setdefault(name, []).append(item)
    empty = [name for name, group in grouped.items() if not group]
    if empty:
        raise RuntimeError(f'empty environment groups: {empty}')
    return {
        name: default_collate(group) for name, group in grouped.items()
    }


__all__ = [
    'BalancedEnvBatchSampler',
    'ConcatEnvDataset',
    'EnvTaggedDataset',
    'collate_by_env',
]
