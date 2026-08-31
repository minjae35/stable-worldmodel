"""Smoke tests for the Design v0 multi-environment data pipeline."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from stable_worldmodel.data import (
    BalancedEnvironmentBatchSampler,
    MultiEnvironmentDataset,
)


class _EnvironmentDataset:
    def __init__(
        self,
        name: str,
        action_dim: int,
        length: int = 5,
        extra_fields: tuple[str, ...] = (),
    ):
        self.name = name
        self.action_dim = action_dim
        self.length = length
        self.extra_fields = extra_fields
        self.column_names = ['pixels', 'action', *extra_fields]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        action = torch.arange(
            1,
            self.action_dim + 1,
            dtype=torch.float32,
        ).repeat(3, 1)
        sample = {
            'pixels': torch.full((3, 3, 8, 8), idx, dtype=torch.uint8),
            'action': action,
        }
        widths = {'proprio': 4, 'state': 5, 'observation': 6}
        for field in self.extra_fields:
            sample[field] = torch.zeros(3, widths[field])
        return sample


def test_tworoom_pusht_ogbcube_balanced_batch():
    datasets = {
        'TwoRoom': _EnvironmentDataset('TwoRoom', action_dim=2),
        'PushT': _EnvironmentDataset('PushT', action_dim=2),
        'OGBCube': _EnvironmentDataset('OGBCube', action_dim=5),
    }
    dataset = MultiEnvironmentDataset(datasets)
    sampler = BalancedEnvironmentBatchSampler(
        dataset,
        batch_size=6,
        num_batches=1,
        shuffle=False,
    )

    batch = next(iter(DataLoader(dataset, batch_sampler=sampler)))

    assert batch['pixels'].shape == (6, 3, 3, 8, 8)
    assert batch['action'].shape == (6, 3, 5)
    assert batch['action_mask'].shape == (6, 3, 5)
    assert batch['env_id'].shape == (6,)
    assert batch['env_id'].tolist() == [0, 0, 1, 1, 2, 2]

    assert torch.count_nonzero(batch['action'][:4, :, 2:]) == 0
    assert not batch['action_mask'][:4, :, 2:].any()
    assert batch['action_mask'][:4, :, :2].all()
    assert batch['action_mask'][4:, :, :5].all()


def test_heterogeneous_environment_schemas_collate_canonical_fields():
    datasets = {
        'PushT': _EnvironmentDataset(
            'PushT',
            action_dim=2,
            extra_fields=('proprio', 'state'),
        ),
        'TwoRoom': _EnvironmentDataset(
            'TwoRoom',
            action_dim=2,
            extra_fields=('proprio',),
        ),
        'OGBCube': _EnvironmentDataset(
            'OGBCube',
            action_dim=5,
            extra_fields=('observation',),
        ),
    }
    dataset = MultiEnvironmentDataset(datasets)
    sampler = BalancedEnvironmentBatchSampler(
        dataset,
        batch_size=6,
        num_batches=1,
        shuffle=False,
    )

    batch = next(iter(DataLoader(dataset, batch_sampler=sampler)))

    assert set(batch) == {'pixels', 'action', 'action_mask', 'env_id'}
    assert batch['pixels'].shape == (6, 3, 3, 8, 8)
    assert batch['action'].shape == (6, 3, 5)
    assert batch['action_mask'].shape == (6, 3, 5)
    assert batch['env_id'].shape == (6,)
    assert torch.bincount(batch['env_id'], minlength=3).tolist() == [2, 2, 2]

    # The wrapper presents a canonical view without changing raw samples.
    assert set(datasets['PushT'][0]) == {
        'pixels',
        'action',
        'proprio',
        'state',
    }
    assert set(datasets['TwoRoom'][0]) == {'pixels', 'action', 'proprio'}
    assert set(datasets['OGBCube'][0]) == {
        'pixels',
        'action',
        'observation',
    }


def test_balanced_sampler_ignores_environment_dataset_size():
    dataset = MultiEnvironmentDataset(
        {
            'TwoRoom': _EnvironmentDataset('TwoRoom', 2, length=2),
            'PushT': _EnvironmentDataset('PushT', 2, length=5),
            'OGBCube': _EnvironmentDataset('OGBCube', 5, length=8),
        }
    )
    sampler = BalancedEnvironmentBatchSampler(
        dataset,
        batch_size=6,
        num_batches=4,
        shuffle=False,
    )

    for batch in DataLoader(dataset, batch_sampler=sampler):
        counts = torch.bincount(batch['env_id'], minlength=3)
        assert counts.tolist() == [2, 2, 2]


def test_existing_single_environment_loader_is_unchanged():
    dataset = _EnvironmentDataset('PushT', action_dim=2)
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert set(batch) == {'pixels', 'action'}
    assert batch['action'].shape == (2, 3, 2)
