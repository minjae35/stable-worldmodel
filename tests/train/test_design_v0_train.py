"""Fixture smoke tests for the Design v0 training runner."""

from types import SimpleNamespace

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from scripts.data.design_v0.manifest import (
    build_split_manifest,
    write_json,
)
from scripts.train.design_v0 import (
    DesignV0TrainingModule,
    build_checkpoint_metadata,
    build_design_v0_data,
    clip_indices_for_episodes,
    effective_action_dim,
    load_episode_split_manifest,
    resolve_dataset_path,
    split_datasets_with_episode_manifests,
    validate_environment_action_dims,
)
from stable_worldmodel.wm.design_v0 import (
    DesignV0Core,
    FrozenVisualEncoder,
)


K, H, D = 2, 2, 4
FRAMESKIP = 5


class _EpisodeClipDataset(Dataset):
    """Clip dataset with frameskip-flattened actions, matching SWM Dataset."""

    def __init__(
        self,
        num_episodes: int,
        raw_action_dim: int,
        *,
        episode_length: int = 24,
        num_steps: int = K + H,
        frameskip: int = FRAMESKIP,
    ):
        self.num_episodes = num_episodes
        self.raw_action_dim = raw_action_dim
        self.episode_length = episode_length
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.effective_action_dim = raw_action_dim * frameskip
        self.span = num_steps * frameskip
        self.lengths = [episode_length] * num_episodes
        self.clip_indices = [
            (episode_index, start)
            for episode_index, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

    def __len__(self):
        return len(self.clip_indices)

    def load_episode(self, episode_idx: int) -> dict:
        action = torch.full(
            (self.episode_length, self.raw_action_dim),
            float(episode_idx),
        )
        pixels = torch.full(
            (self.episode_length, 1, 2, 2),
            float(episode_idx),
        )
        return {'pixels': pixels, 'action': action}

    def __getitem__(self, index):
        episode_index, start = self.clip_indices[index]
        pixels = torch.full(
            (self.num_steps, 1, 2, 2),
            float(episode_index),
        )
        action = torch.full(
            (self.num_steps, self.effective_action_dim),
            float(episode_index),
        )
        del start
        return {'pixels': pixels, 'action': action}


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


def _fixture_manifest(
    environment: str,
    num_episodes: int,
    *,
    artifact_sha256: str = 'artifact-sha',
    split_manifest_sha256: str = 'split-sha',
):
    payload = build_split_manifest(
        environment=environment,
        artifact_path=f'fixture://{environment}',
        num_episodes=num_episodes,
        artifact_sha256=artifact_sha256,
    )
    payload['split_manifest_sha256'] = split_manifest_sha256
    return payload


def _split(datasets, **sha_overrides):
    manifests = {}
    artifact_ids = {}
    artifact_paths = {}
    split_manifest_ids = {}
    for name, dataset in datasets.items():
        extra = sha_overrides.get(name, {})
        manifests[name] = _fixture_manifest(
            name,
            len(dataset.lengths),
            artifact_sha256=extra.get('artifact_sha256', f'{name}-art'),
            split_manifest_sha256=extra.get(
                'split_manifest_sha256', f'{name}-split'
            ),
        )
        artifact_ids[name] = f'design_v0/{name.lower()}'
        artifact_paths[name] = f'/cache/datasets/design_v0/{name.lower()}'
        split_manifest_ids[name] = f'design_v0/splits/{name.lower()}.json'
    return split_datasets_with_episode_manifests(
        datasets,
        manifests,
        artifact_ids=artifact_ids,
        artifact_paths=artifact_paths,
        split_manifest_ids=split_manifest_ids,
    )


def test_resolve_dataset_path_uses_cache_root(tmp_path, monkeypatch):
    monkeypatch.delenv('STABLEWM_HOME', raising=False)
    resolved = resolve_dataset_path(
        'design_v0/tworoom_expert.lance',
        cache_dir=tmp_path,
    )
    assert resolved == (
        tmp_path / 'datasets' / 'design_v0' / 'tworoom_expert.lance'
    )
    assert not str(resolved).startswith('/home/')


def test_load_episode_split_manifest_round_trip(tmp_path):
    payload = _fixture_manifest('PushT', 20)
    path = write_json(tmp_path / 'pusht.json', payload)
    loaded = load_episode_split_manifest(path)
    assert loaded['train_episode_indices'] == payload[
        'train_episode_indices'
    ]
    assert loaded['val_episode_indices'] == payload['val_episode_indices']
    assert loaded['split_manifest_sha256']
    assert set(loaded['train_episode_indices']).isdisjoint(
        loaded['val_episode_indices']
    )


def test_episode_split_is_identical_across_run_compositions():
    pusht = _EpisodeClipDataset(num_episodes=20, raw_action_dim=2)
    per_train, per_val, per_meta = _split({'PushT': pusht})
    joint_train, joint_val, joint_meta = _split(
        {
            'TwoRoom': _EpisodeClipDataset(num_episodes=12, raw_action_dim=2),
            'PushT': pusht,
            'OGBCube': _EpisodeClipDataset(num_episodes=28, raw_action_dim=5),
        }
    )

    assert (
        per_meta['environments']['PushT']['train_episode_indices']
        == joint_meta['environments']['PushT']['train_episode_indices']
    )
    assert (
        per_meta['environments']['PushT']['val_episode_indices']
        == joint_meta['environments']['PushT']['val_episode_indices']
    )
    assert per_train['PushT'].indices == joint_train['PushT'].indices
    assert per_val['PushT'].indices == joint_val['PushT'].indices


def test_train_and_val_episode_sets_do_not_leak():
    datasets = {
        'TwoRoom': _EpisodeClipDataset(num_episodes=12, raw_action_dim=2),
        'PushT': _EpisodeClipDataset(num_episodes=20, raw_action_dim=2),
        'OGBCube': _EpisodeClipDataset(num_episodes=28, raw_action_dim=5),
    }
    train, val, metadata = _split(datasets)
    for name, dataset in datasets.items():
        train_episodes = set(
            metadata['environments'][name]['train_episode_indices']
        )
        val_episodes = set(
            metadata['environments'][name]['val_episode_indices']
        )
        assert train_episodes.isdisjoint(val_episodes)
        train_eps = {
            dataset.clip_indices[index][0]
            for index in train[name].indices
        }
        val_eps = {
            dataset.clip_indices[index][0]
            for index in val[name].indices
        }
        assert train_eps <= train_episodes
        assert val_eps <= val_episodes
        assert train_eps.isdisjoint(val_eps)


def test_effective_action_dim_is_raw_times_frameskip():
    assert effective_action_dim(2, 5) == 10
    assert effective_action_dim(5, 5) == 25
    with pytest.raises(ValueError, match='positive'):
        effective_action_dim(2, 0)


def test_validate_environment_action_dims_uses_raw_then_effective():
    assert validate_environment_action_dims(
        'TwoRoom',
        raw_action_dim=2,
        clip_action_dim=10,
        frameskip=5,
    ) == {
        'raw_action_dim': 2,
        'frameskip': 5,
        'effective_action_dim': 10,
        'action_block_dim': 10,
    }
    assert validate_environment_action_dims(
        'OGBCube',
        raw_action_dim=5,
        clip_action_dim=25,
        frameskip=5,
    )['effective_action_dim'] == 25
    with pytest.raises(ValueError, match='raw action dim mismatch'):
        validate_environment_action_dims(
            'TwoRoom',
            raw_action_dim=5,
            clip_action_dim=25,
            frameskip=5,
        )
    with pytest.raises(ValueError, match='effective action dim mismatch'):
        validate_environment_action_dims(
            'TwoRoom',
            raw_action_dim=2,
            clip_action_dim=2,
            frameskip=5,
        )


def test_fixture_dataset_keeps_raw_and_flattened_clip_dims():
    dataset = _EpisodeClipDataset(num_episodes=4, raw_action_dim=2)
    assert dataset.load_episode(0)['action'].shape[-1] == 2
    assert dataset[0]['action'].shape[-1] == 10
    ogb = _EpisodeClipDataset(num_episodes=4, raw_action_dim=5)
    assert ogb.load_episode(0)['action'].shape[-1] == 5
    assert ogb[0]['action'].shape[-1] == 25


def test_explicit_exposure_balances_unequal_environment_datasets():
    train, val, metadata = _split(
        {
            'TwoRoom': _EpisodeClipDataset(num_episodes=12, raw_action_dim=2),
            'PushT': _EpisodeClipDataset(num_episodes=20, raw_action_dim=2),
            'OGBCube': _EpisodeClipDataset(num_episodes=28, raw_action_dim=5),
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
        assert batch['action'].shape == (6, K + H, 25)
        assert batch['action_mask'].shape == (6, K + H, 25)
        assert torch.bincount(
            batch['env_id'], minlength=3
        ).tolist() == [2, 2, 2]


def test_action_padding_is_10_10_25_with_joint_max_25():
    train, val, metadata = _split(
        {
            'TwoRoom': _EpisodeClipDataset(num_episodes=12, raw_action_dim=2),
            'PushT': _EpisodeClipDataset(num_episodes=20, raw_action_dim=2),
            'OGBCube': _EpisodeClipDataset(num_episodes=28, raw_action_dim=5),
        }
    )
    data = build_design_v0_data(
        train,
        val,
        split_metadata=metadata,
        per_environment_batch_size=1,
        steps_per_epoch=1,
        validation_steps=1,
        sampler_seed=456,
    )
    assert data.train_dataset.action_dims == (10, 10, 25)
    assert data.train_dataset.max_action_dim == 25
    batch = next(iter(data.train_loader))
    assert batch['action'].shape[-1] == 25
    # env order is TwoRoom, PushT, OGBCube.
    masks = {
        int(env_id): batch['action_mask'][index, 0]
        for index, env_id in enumerate(batch['env_id'].tolist())
    }
    assert masks[0].tolist() == [True] * 10 + [False] * 15
    assert masks[1].tolist() == [True] * 10 + [False] * 15
    assert masks[2].tolist() == [True] * 25


def test_per_environment_run_uses_the_same_balanced_data_path():
    train, val, metadata = _split(
        {'PushT': _EpisodeClipDataset(num_episodes=20, raw_action_dim=2)}
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
            'PushT': _EpisodeClipDataset(num_episodes=20, raw_action_dim=2),
            'OGBCube': _EpisodeClipDataset(num_episodes=28, raw_action_dim=5),
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
    core = _core(max_action_dim=25, num_environments=2)
    cfg = OmegaConf.create(
        {
            'wm': {'horizon': H},
            'backbone': {'name': 'fixture'},
            'image_size': 2,
            'loader': {'sampler_seed': 456},
            'seed': 789,
        }
    )
    metadata = build_checkpoint_metadata(
        cfg,
        data,
        {
            'PushT': {
                'name': 'design_v0/pusht_expert_train.h5',
                'frameskip': FRAMESKIP,
                'raw_action_dim': 2,
                'effective_action_dim': 10,
                'action_block_dim': 10,
            },
            'OGBCube': {
                'name': 'design_v0/ogbcube/cube_single_front_expert.lance',
                'frameskip': FRAMESKIP,
                'raw_action_dim': 5,
                'effective_action_dim': 25,
                'action_block_dim': 25,
            },
        },
        core,
    )

    assert metadata['env_names'] == ['PushT', 'OGBCube']
    assert metadata['env_to_id'] == {'PushT': 0, 'OGBCube': 1}
    assert metadata['split'] == split_metadata
    assert metadata['exposure'] == data.exposure_metadata
    assert metadata['raw_action_dims'] == {'PushT': 2, 'OGBCube': 5}
    assert metadata['effective_action_dims'] == {'PushT': 10, 'OGBCube': 25}
    assert metadata['action_dims'] == {'PushT': 10, 'OGBCube': 25}
    assert metadata['max_action_dim'] == 25
    assert metadata['history_size'] == K
    assert metadata['horizon'] == H
    pusht = metadata['datasets']['PushT']
    assert pusht['artifact_sha256'] == 'PushT-art'
    assert pusht['split_manifest_sha256'] == 'PushT-split'
    assert pusht['train_episodes'] == 18
    assert pusht['val_episodes'] == 2
    assert pusht['raw_action_dim'] == 2
    assert pusht['frameskip'] == FRAMESKIP
    assert pusht['effective_action_dim'] == 10
    assert pusht['action_block_dim'] == 10
    ogb = metadata['datasets']['OGBCube']
    assert ogb['raw_action_dim'] == 5
    assert ogb['frameskip'] == FRAMESKIP
    assert ogb['effective_action_dim'] == 25
    assert ogb['action_block_dim'] == 25


def test_one_batch_training_and_checkpoint_round_trip(tmp_path):
    train, val, split_metadata = _split(
        {
            'TwoRoom': _EpisodeClipDataset(num_episodes=12, raw_action_dim=2),
            'PushT': _EpisodeClipDataset(num_episodes=20, raw_action_dim=2),
            'OGBCube': _EpisodeClipDataset(num_episodes=28, raw_action_dim=5),
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
        _core(max_action_dim=25, num_environments=3),
        optimizer_config={'type': 'SGD', 'lr': 1e-3},
        checkpoint_metadata=metadata,
    )
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint['state_dict'])
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])

    mismatched = DesignV0TrainingModule(
        _core(max_action_dim=25, num_environments=3),
        optimizer_config={'type': 'SGD', 'lr': 1e-3},
        checkpoint_metadata={
            **metadata,
            'env_to_id': {'PushT': 0, 'TwoRoom': 1, 'OGBCube': 2},
        },
    )
    with pytest.raises(ValueError, match='env_to_id'):
        mismatched.on_load_checkpoint(checkpoint)

    sha_mismatch_split = {
        **split_metadata,
        'environments': {
            **split_metadata['environments'],
            'PushT': {
                **split_metadata['environments']['PushT'],
                'artifact_sha256': 'other-artifact',
            },
        },
    }
    sha_mismatched = DesignV0TrainingModule(
        _core(max_action_dim=25, num_environments=3),
        optimizer_config={'type': 'SGD', 'lr': 1e-3},
        checkpoint_metadata={
            **metadata,
            'split': sha_mismatch_split,
        },
    )
    with pytest.raises(ValueError, match='artifact/split'):
        sha_mismatched.on_load_checkpoint(checkpoint)


def test_clip_indices_only_keep_selected_episodes():
    dataset = _EpisodeClipDataset(num_episodes=10, raw_action_dim=2)
    selected = [0, 3, 9]
    clip_indices = clip_indices_for_episodes(dataset, selected)
    episodes = {
        dataset.clip_indices[index][0] for index in clip_indices
    }
    assert episodes == set(selected)
