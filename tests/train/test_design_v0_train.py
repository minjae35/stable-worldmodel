"""Fixture smoke tests for the Design v0 training runner."""

from types import SimpleNamespace

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from scripts.train.design_v0 import (
    DesignV0TrainingModule,
    build_checkpoint_metadata,
    build_design_v0_data,
    split_environment_datasets,
)
from stable_worldmodel.wm.design_v0 import (
    DesignV0Core,
    FrozenVisualEncoder,
)


K, H, D = 2, 2, 4


class _EnvironmentDataset(Dataset):
    def __init__(self, length: int, action_dim: int):
        self.length = length
        self.action_dim = action_dim

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pixels = torch.arange(
            (K + H) * D, dtype=torch.float32
        ).reshape(K + H, 1, 2, 2)
        pixels = pixels + float(index)
        action = torch.full(
            (K + H, self.action_dim),
            float(index + 1) / self.length,
        )
        return {'pixels': pixels, 'action': action}


class _StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=D)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, pixels):
        cls = pixels.flatten(1)[:, :D] * self.scale
        return SimpleNamespace(
            last_hidden_state=torch.stack([cls, cls + 100], dim=1)
        )


def _core(max_action_dim: int, num_environments: int):
    return DesignV0Core(
        FrozenVisualEncoder(_StubBackbone()),
        history_size=K,
        max_action_dim=max_action_dim,
        action_embedding_dim=5,
        num_environments=num_environments,
        environment_embedding_dim=3,
        dynamics_hidden_dim=8,
    )


def _split(datasets):
    identifiers = {name: f'fixture://{name}' for name in datasets}
    return split_environment_datasets(
        datasets,
        train_fraction=0.75,
        split_seed=123,
        dataset_identifiers=identifiers,
    )


def test_environment_split_is_identical_across_run_compositions():
    pusht = _EnvironmentDataset(length=20, action_dim=2)
    per_train, per_val, per_meta = _split({'PushT': pusht})
    joint_train, joint_val, joint_meta = _split(
        {
            'TwoRoom': _EnvironmentDataset(length=12, action_dim=2),
            'PushT': pusht,
            'OGBCube': _EnvironmentDataset(length=28, action_dim=7),
        }
    )

    assert per_train['PushT'].indices == joint_train['PushT'].indices
    assert per_val['PushT'].indices == joint_val['PushT'].indices
    assert (
        per_meta['environments']['PushT']
        == joint_meta['environments']['PushT']
    )


def test_explicit_exposure_balances_unequal_environment_datasets():
    train, val, metadata = _split(
        {
            'TwoRoom': _EnvironmentDataset(length=12, action_dim=2),
            'PushT': _EnvironmentDataset(length=20, action_dim=2),
            'OGBCube': _EnvironmentDataset(length=28, action_dim=7),
        }
    )
    data = build_design_v0_data(
        train,
        val,
        split_metadata=metadata,
        per_environment_batch_size=2,
        steps_per_epoch=3,
        validation_steps=1,
        sampler_seed=456,
    )

    assert len(data.train_loader) == 3
    assert data.exposure_metadata[
        'samples_per_environment_per_epoch'
    ] == 6
    for batch in data.train_loader:
        assert batch['pixels'].shape == (6, K + H, 1, 2, 2)
        assert batch['action'].shape == (6, K + H, 7)
        assert batch['action_mask'].shape == (6, K + H, 7)
        assert torch.bincount(
            batch['env_id'], minlength=3
        ).tolist() == [2, 2, 2]


def test_per_environment_run_uses_the_same_balanced_data_path():
    train, val, metadata = _split(
        {'PushT': _EnvironmentDataset(length=20, action_dim=2)}
    )
    data = build_design_v0_data(
        train,
        val,
        split_metadata=metadata,
        per_environment_batch_size=3,
        steps_per_epoch=2,
        validation_steps=1,
        sampler_seed=456,
    )
    batch = next(iter(data.train_loader))

    assert isinstance(data.train_dataset.env_to_id, dict)
    assert data.train_dataset.env_to_id == {'PushT': 0}
    assert batch['pixels'].shape[0] == 3
    assert batch['env_id'].tolist() == [0, 0, 0]
    assert len(data.train_loader) == 2


def test_checkpoint_metadata_preserves_mapping_split_and_exposure():
    train, val, split_metadata = _split(
        {
            'PushT': _EnvironmentDataset(length=20, action_dim=2),
            'OGBCube': _EnvironmentDataset(length=28, action_dim=7),
        }
    )
    data = build_design_v0_data(
        train,
        val,
        split_metadata=split_metadata,
        per_environment_batch_size=2,
        steps_per_epoch=3,
        validation_steps=1,
        sampler_seed=456,
    )
    core = _core(max_action_dim=7, num_environments=2)
    cfg = OmegaConf.create(
        {
            'wm': {'horizon': H},
            'backbone': {'name': 'fixture'},
            'image_size': 2,
            'loader': {'sampler_seed': 456},
            'seed': 789,
            'split': {'seed': 123, 'train_fraction': 0.75},
        }
    )
    metadata = build_checkpoint_metadata(
        cfg,
        data,
        {
            'PushT': {'name': 'fixture://PushT', 'frameskip': 1},
            'OGBCube': {'name': 'fixture://OGBCube', 'frameskip': 1},
        },
        core,
    )

    assert metadata['env_names'] == ['PushT', 'OGBCube']
    assert metadata['env_to_id'] == {'PushT': 0, 'OGBCube': 1}
    assert metadata['split'] == split_metadata
    assert metadata['exposure'] == data.exposure_metadata
    assert metadata['action_dims'] == {'PushT': 2, 'OGBCube': 7}
    assert metadata['max_action_dim'] == 7


def test_one_batch_training_and_checkpoint_round_trip(tmp_path):
    train, val, split_metadata = _split(
        {
            'TwoRoom': _EnvironmentDataset(length=12, action_dim=2),
            'PushT': _EnvironmentDataset(length=20, action_dim=2),
            'OGBCube': _EnvironmentDataset(length=28, action_dim=7),
        }
    )
    data = build_design_v0_data(
        train,
        val,
        split_metadata=split_metadata,
        per_environment_batch_size=1,
        steps_per_epoch=1,
        validation_steps=1,
        sampler_seed=456,
    )
    metadata = {
        'env_to_id': dict(data.train_dataset.env_to_id),
        'split': split_metadata,
        'exposure': data.exposure_metadata,
    }
    core = _core(
        max_action_dim=data.train_dataset.max_action_dim,
        num_environments=3,
    )
    module = DesignV0TrainingModule(
        core,
        optimizer_config={
            'type': 'SGD',
            'lr': 1e-3,
        },
        checkpoint_metadata=metadata,
    )

    optimizer = module.configure_optimizers()
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    assert all(
        id(parameter) not in optimized
        for parameter in core.visual_encoder.parameters()
    )
    assert {
        id(parameter)
        for parameter in core.parameters()
        if parameter.requires_grad
    } == optimized

    encoder_before = {
        name: value.detach().clone()
        for name, value in core.visual_encoder.state_dict().items()
    }
    dynamics_before = core.dynamics.fc1.weight.detach().clone()
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=data.train_loader)

    assert trainer.global_step == 1
    assert not torch.equal(
        dynamics_before, core.dynamics.fc1.weight.detach()
    )
    for name, value in core.visual_encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[name])
    assert all(
        parameter.grad is None
        for parameter in core.visual_encoder.parameters()
    )

    checkpoint_path = tmp_path / 'design_v0.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint['design_v0_metadata'] == metadata
    assert 'state_dict' in checkpoint

    restored = DesignV0TrainingModule(
        _core(max_action_dim=7, num_environments=3),
        optimizer_config={'type': 'SGD', 'lr': 1e-3},
        checkpoint_metadata=metadata,
    )
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint['state_dict'])
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])

    mismatched = DesignV0TrainingModule(
        _core(max_action_dim=7, num_environments=3),
        optimizer_config={'type': 'SGD', 'lr': 1e-3},
        checkpoint_metadata={
            **metadata,
            'env_to_id': {'PushT': 0, 'TwoRoom': 1, 'OGBCube': 2},
        },
    )
    with pytest.raises(ValueError, match='env_to_id'):
        mismatched.on_load_checkpoint(checkpoint)
