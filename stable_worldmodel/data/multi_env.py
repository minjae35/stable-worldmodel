"""Multi-environment dataset composition and balanced batch sampling.

This module keeps the underlying episode datasets independent.  It only adds
the metadata needed by a joint training run and makes heterogeneous continuous
actions collatable; it does not merge or rewrite dataset storage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler


class MultiEnvironmentDataset:
    """Expose several datasets through one global sample index.

    The wrapper exposes a canonical view of every source sample so that raw
    environment-specific fields do not break collation.  Each returned sample
    also contains an integer ``env_id``.  Its continuous ``action`` is
    zero-padded on the last axis to ``max_action_dim`` and a boolean
    ``action_mask`` of the same shape marks the original entries.

    Args:
        datasets: Environment name to dataset mapping.  Mapping order defines
            the integer environment IDs.
        max_action_dim: Padded action width.  Defaults to the largest action
            width observed in the wrapped datasets.
        action_dims: Optional environment name to action width mapping.  When
            omitted, the width is inferred from one sample per dataset.
        canonical_keys: Source fields exposed to joint training.  Defaults to
            ``pixels`` and the configured action key.  Raw datasets are not
            modified and may retain arbitrary environment-specific fields.
        action_key: Sample key containing continuous actions.
        env_id_key: Output key containing the integer environment ID.
        action_mask_key: Output key containing the validity mask.
    """

    def __init__(
        self,
        datasets: Mapping[str, Any],
        *,
        max_action_dim: int | None = None,
        action_dims: Mapping[str, int] | None = None,
        canonical_keys: Sequence[str] | None = None,
        action_key: str = 'action',
        env_id_key: str = 'env_id',
        action_mask_key: str = 'action_mask',
    ) -> None:
        if not datasets:
            raise ValueError('Need at least one environment dataset')

        self.env_names = tuple(datasets.keys())
        self.datasets = tuple(datasets.values())
        self.env_to_id = {
            name: env_id for env_id, name in enumerate(self.env_names)
        }
        self.action_key = action_key
        self.env_id_key = env_id_key
        self.action_mask_key = action_mask_key
        if canonical_keys is None:
            canonical_keys = ('pixels', action_key)
        self.canonical_keys = tuple(
            dict.fromkeys((*canonical_keys, action_key))
        )
        reserved = {env_id_key, action_mask_key}
        collision = reserved.intersection(self.canonical_keys)
        if collision:
            raise ValueError(
                'canonical_keys cannot contain generated fields: '
                f'{sorted(collision)}'
            )

        lengths = [len(dataset) for dataset in self.datasets]
        if any(length <= 0 for length in lengths):
            empty = [
                name
                for name, length in zip(self.env_names, lengths)
                if length <= 0
            ]
            raise ValueError(
                f'Environment datasets must be non-empty: {empty}'
            )
        self.env_lengths = tuple(lengths)
        self._cum = np.cumsum([0, *lengths], dtype=np.int64)

        if action_dims is None:
            inferred = {
                name: self._infer_action_dim(dataset)
                for name, dataset in datasets.items()
            }
        else:
            missing = set(self.env_names) - set(action_dims)
            extra = set(action_dims) - set(self.env_names)
            if missing or extra:
                raise ValueError(
                    'action_dims keys must match datasets keys; '
                    f'missing={sorted(missing)}, extra={sorted(extra)}'
                )
            inferred = {
                name: int(action_dims[name]) for name in self.env_names
            }

        if any(dim <= 0 for dim in inferred.values()):
            raise ValueError(f'Action dimensions must be positive: {inferred}')
        self.action_dims = tuple(inferred[name] for name in self.env_names)

        largest_action_dim = max(self.action_dims)
        self.max_action_dim = (
            largest_action_dim
            if max_action_dim is None
            else int(max_action_dim)
        )
        if self.max_action_dim < largest_action_dim:
            raise ValueError(
                f'max_action_dim={self.max_action_dim} is smaller than '
                f'the largest action dimension {largest_action_dim}'
            )

    @property
    def column_names(self) -> list[str]:
        """Canonical source columns plus the two joint-training fields."""
        return [
            *self.canonical_keys,
            self.action_mask_key,
            self.env_id_key,
        ]

    def __len__(self) -> int:
        return int(self._cum[-1])

    def environment_indices(self, env_id: int) -> range:
        """Return the global sample-index range for one environment."""
        if env_id < 0 or env_id >= len(self.datasets):
            raise IndexError(f'Invalid env_id: {env_id}')
        return range(int(self._cum[env_id]), int(self._cum[env_id + 1]))

    def _loc(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        env_id = int(np.searchsorted(self._cum[1:], idx, side='right'))
        local_idx = idx - int(self._cum[env_id])
        return env_id, local_idx

    def _infer_action_dim(self, dataset: Any) -> int:
        sample = dataset[0]
        if self.action_key not in sample:
            raise KeyError(
                f"Dataset sample does not contain action key '{self.action_key}'"
            )
        action = sample[self.action_key]
        if not hasattr(action, 'shape') or len(action.shape) == 0:
            raise ValueError(
                f"Action '{self.action_key}' must have a feature axis"
            )
        return int(action.shape[-1])

    def _augment(self, item: dict, env_id: int) -> dict:
        missing = [key for key in self.canonical_keys if key not in item]
        if missing:
            raise KeyError(
                f'Environment {self.env_names[env_id]!r} is missing '
                f'canonical fields {missing}'
            )
        out = {key: item[key] for key in self.canonical_keys}

        action = out[self.action_key]
        expected_dim = self.action_dims[env_id]
        if not hasattr(action, 'shape') or len(action.shape) == 0:
            raise ValueError(
                f"Action '{self.action_key}' must have a feature axis"
            )
        if int(action.shape[-1]) != expected_dim:
            raise ValueError(
                f'Environment {self.env_names[env_id]!r} returned action '
                f'width {action.shape[-1]}, expected {expected_dim}'
            )

        if isinstance(action, torch.Tensor):
            padded = action.new_zeros(*action.shape[:-1], self.max_action_dim)
            padded[..., :expected_dim] = action
            mask = torch.zeros_like(padded, dtype=torch.bool)
            mask[..., :expected_dim] = True
        else:
            action_array = np.asarray(action)
            padded = np.zeros(
                (*action_array.shape[:-1], self.max_action_dim),
                dtype=action_array.dtype,
            )
            padded[..., :expected_dim] = action_array
            mask = np.zeros(padded.shape, dtype=np.bool_)
            mask[..., :expected_dim] = True

        out[self.action_key] = padded
        out[self.action_mask_key] = mask
        out[self.env_id_key] = torch.tensor(env_id, dtype=torch.long)
        return out

    def __getitem__(self, idx: int) -> dict:
        env_id, local_idx = self._loc(idx)
        return self._augment(self.datasets[env_id][local_idx], env_id)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        mapped = [self._loc(int(idx)) for idx in indices]
        groups: dict[int, list[tuple[int, int]]] = {}
        for position, (env_id, local_idx) in enumerate(mapped):
            groups.setdefault(env_id, []).append((position, local_idx))

        results: list[dict | None] = [None] * len(indices)
        for env_id, group in groups.items():
            dataset = self.datasets[env_id]
            positions = [position for position, _ in group]
            local_indices = [local_idx for _, local_idx in group]
            if hasattr(dataset, '__getitems__'):
                items = dataset.__getitems__(local_indices)
            else:
                items = [dataset[idx] for idx in local_indices]
            for position, item in zip(positions, items):
                results[position] = self._augment(item, env_id)

        return results  # type: ignore[return-value]


class BalancedEnvironmentBatchSampler(Sampler[list[int]]):
    """Yield full batches with equal samples from every environment.

    Smaller datasets are reshuffled and cycled as needed.  This gives every
    environment exactly ``batch_size / num_environments`` samples per update
    without depending on dataset size.

    Args:
        dataset: A :class:`MultiEnvironmentDataset`.
        batch_size: Total batch size; must be divisible by the environment
            count.
        num_batches: Batches per epoch.  Defaults to enough batches to cover
            the largest environment once.
        shuffle: Shuffle within each environment and within each batch.
        generator: Optional torch RNG.
    """

    def __init__(
        self,
        dataset: MultiEnvironmentDataset,
        batch_size: int,
        *,
        num_batches: int | None = None,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if not isinstance(dataset, MultiEnvironmentDataset):
            raise TypeError('dataset must be a MultiEnvironmentDataset')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_environments = len(dataset.datasets)
        if self.batch_size % self.num_environments != 0:
            raise ValueError(
                'batch_size must be divisible by the number of environments'
            )
        self.samples_per_environment = self.batch_size // self.num_environments
        if num_batches is None:
            largest = max(dataset.env_lengths)
            num_batches = (
                largest + self.samples_per_environment - 1
            ) // self.samples_per_environment
        if num_batches <= 0:
            raise ValueError('num_batches must be positive')

        self.num_batches = int(num_batches)
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self) -> int:
        return self.num_batches

    def _draw(self, env_id: int, count: int) -> list[int]:
        source = list(self.dataset.environment_indices(env_id))
        drawn: list[int] = []
        while len(drawn) < count:
            if self.shuffle:
                order = torch.randperm(
                    len(source), generator=self.generator
                ).tolist()
                cycle = [source[idx] for idx in order]
            else:
                cycle = source
            drawn.extend(cycle[: count - len(drawn)])
        return drawn

    def __iter__(self):
        count = self.num_batches * self.samples_per_environment
        streams = [
            self._draw(env_id, count)
            for env_id in range(self.num_environments)
        ]

        for batch_idx in range(self.num_batches):
            start = batch_idx * self.samples_per_environment
            end = start + self.samples_per_environment
            batch = []
            for stream in streams:
                batch.extend(stream[start:end])
            if self.shuffle:
                order = torch.randperm(
                    len(batch), generator=self.generator
                ).tolist()
                batch = [batch[idx] for idx in order]
            yield batch
