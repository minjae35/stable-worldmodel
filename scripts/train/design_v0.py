"""Train the independent Design v0 model on one or more environments."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from stable_worldmodel.data import (
    BalancedEnvironmentBatchSampler,
    MultiEnvironmentDataset,
)
from stable_worldmodel.wm.design_v0 import (
    DesignV0Core,
    DesignV0Objective,
    FrozenVisualEncoder,
)
from scripts.data.design_v0.manifest import (
    indices_sha256,
    sha256_file,
)
from scripts.data.design_v0.spec import (
    EXPECTED_ACTION_DIMS,
    cache_root,
)


SPLIT_VERSION = 'design_v0_canonical_episode_v1'
METADATA_VERSION = 1
_RESUME_SPLIT_KEYS = (
    'artifact_id',
    'artifact_sha256',
    'split_manifest_id',
    'split_manifest_sha256',
    'train_indices_sha256',
    'val_indices_sha256',
)


@dataclass
class DesignV0Data:
    """Datasets, loaders, and deterministic split metadata for one run."""

    train_dataset: MultiEnvironmentDataset
    val_dataset: MultiEnvironmentDataset
    train_loader: DataLoader
    val_loader: DataLoader
    split_metadata: dict[str, Any]
    exposure_metadata: dict[str, Any]


def get_img_preprocessor(
    source: str = 'pixels',
    target: str = 'pixels',
    image_size: int = 224,
):
    """Return the ImageNet-normalized preprocessing used by visual baselines."""
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(
            **spt.data.dataset_stats.ImageNet,
            source=source,
            target=target,
        ),
        spt.data.transforms.Resize(
            image_size,
            source=source,
            target=target,
        ),
    )


def datasets_root(cache_dir: str | Path | None = None) -> Path:
    """Return ``<cache>/datasets`` without a machine-specific path."""
    return cache_root(cache_dir) / 'datasets'


def resolve_dataset_path(
    name: str,
    cache_dir: str | Path | None = None,
) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = datasets_root(cache_dir) / path
    return path


def load_episode_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a canonical episode-level split manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    if payload.get('unit') != 'episode':
        raise ValueError(
            f'Split manifest {manifest_path} must use unit=episode, '
            f'got {payload.get("unit")!r}'
        )
    train = [int(index) for index in payload['train_episode_indices']]
    val = [int(index) for index in payload['val_episode_indices']]
    if set(train).intersection(val):
        raise ValueError(
            f'Split manifest {manifest_path} leaks episodes between train/val'
        )
    if indices_sha256(train) != payload['train_indices_sha256']:
        raise ValueError(
            f'Split manifest {manifest_path} train checksum mismatch'
        )
    if indices_sha256(val) != payload['val_indices_sha256']:
        raise ValueError(
            f'Split manifest {manifest_path} val checksum mismatch'
        )
    if not train or not val:
        raise ValueError(
            f'Split manifest {manifest_path} needs non-empty train and val'
        )
    payload = dict(payload)
    payload['train_episode_indices'] = train
    payload['val_episode_indices'] = val
    payload['split_manifest_sha256'] = sha256_file(manifest_path)
    return payload


def clip_indices_for_episodes(
    dataset: Dataset,
    episode_ids: list[int],
) -> list[int]:
    """Return clip indices whose source episode is in ``episode_ids``."""
    if not hasattr(dataset, 'clip_indices'):
        raise TypeError(
            'dataset must expose clip_indices so clips can be built '
            'from selected episodes'
        )
    allowed = {int(episode_id) for episode_id in episode_ids}
    return [
        clip_index
        for clip_index, (episode_index, _start) in enumerate(
            dataset.clip_indices
        )
        if int(episode_index) in allowed
    ]


def split_datasets_with_episode_manifests(
    datasets: Mapping[str, Dataset],
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    artifact_ids: Mapping[str, str] | None = None,
    artifact_paths: Mapping[str, str] | None = None,
    split_manifest_ids: Mapping[str, str] | None = None,
) -> tuple[
    dict[str, Subset],
    dict[str, Subset],
    dict[str, Any],
]:
    """Select canonical train/val episodes, then keep only those K+H clips."""
    if tuple(datasets) != tuple(manifests):
        raise ValueError(
            'Episode manifests must cover the same environments as datasets, '
            f'got datasets={list(datasets)} manifests={list(manifests)}'
        )

    train_datasets: dict[str, Subset] = {}
    val_datasets: dict[str, Subset] = {}
    environments: dict[str, Any] = {}

    for environment_name, dataset in datasets.items():
        payload = dict(manifests[environment_name])
        manifest_environment = payload.get('environment')
        if (
            manifest_environment is not None
            and manifest_environment != environment_name
        ):
            raise ValueError(
                f'Split manifest environment {manifest_environment!r} '
                f'does not match config key {environment_name!r}'
            )
        num_episodes = int(payload['num_episodes'])
        lengths = getattr(dataset, 'lengths', None)
        if lengths is not None and int(len(lengths)) != num_episodes:
            raise ValueError(
                f'{environment_name} episode count mismatch: dataset has '
                f'{len(lengths)}, manifest has {num_episodes}'
            )
        train_episodes = [int(i) for i in payload['train_episode_indices']]
        val_episodes = [int(i) for i in payload['val_episode_indices']]
        if set(train_episodes).intersection(val_episodes):
            raise ValueError(
                f'{environment_name} train/val episode sets overlap'
            )
        train_clip_indices = clip_indices_for_episodes(
            dataset, train_episodes
        )
        val_clip_indices = clip_indices_for_episodes(dataset, val_episodes)
        if not train_clip_indices or not val_clip_indices:
            raise ValueError(
                f'{environment_name} needs clips in both splits after '
                f'selecting episodes, got train_clips='
                f'{len(train_clip_indices)}, val_clips='
                f'{len(val_clip_indices)}'
            )
        train_datasets[environment_name] = Subset(
            dataset, train_clip_indices
        )
        val_datasets[environment_name] = Subset(dataset, val_clip_indices)
        environments[environment_name] = {
            'dataset': (
                artifact_ids[environment_name]
                if artifact_ids is not None
                else environment_name
            ),
            'artifact_id': (
                None
                if artifact_ids is None
                else artifact_ids[environment_name]
            ),
            'artifact_path': (
                None
                if artifact_paths is None
                else str(artifact_paths[environment_name])
            ),
            'artifact_sha256': payload.get('artifact_sha256'),
            'split_manifest_id': (
                None
                if split_manifest_ids is None
                else split_manifest_ids[environment_name]
            ),
            'split_manifest_sha256': payload.get('split_manifest_sha256'),
            'num_episodes': num_episodes,
            'train_episodes': len(train_episodes),
            'val_episodes': len(val_episodes),
            'train_clips': len(train_clip_indices),
            'val_clips': len(val_clip_indices),
            'train_episode_indices': train_episodes,
            'val_episode_indices': val_episodes,
            'train_indices_sha256': indices_sha256(train_episodes),
            'val_indices_sha256': indices_sha256(val_episodes),
        }

    metadata = {
        'version': SPLIT_VERSION,
        'unit': 'episode',
        'environments': environments,
    }
    return train_datasets, val_datasets, metadata


def _loader_kwargs(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'num_workers': int(num_workers),
        'pin_memory': bool(pin_memory),
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = int(prefetch_factor)
    return kwargs


def build_design_v0_data(
    train_datasets: Mapping[str, Dataset],
    val_datasets: Mapping[str, Dataset],
    *,
    split_metadata: dict[str, Any],
    per_environment_batch_size: int,
    steps_per_epoch: int,
    validation_steps: int,
    sampler_seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DesignV0Data:
    """Build the common balanced path for both per-env and joint runs."""
    if tuple(train_datasets) != tuple(val_datasets):
        raise ValueError(
            'Train and validation environment order must match exactly'
        )
    if per_environment_batch_size <= 0:
        raise ValueError('per_environment_batch_size must be positive')
    if steps_per_epoch <= 0:
        raise ValueError('steps_per_epoch must be positive')
    if validation_steps <= 0:
        raise ValueError('validation_steps must be positive')

    train_dataset = MultiEnvironmentDataset(train_datasets)
    action_dims = dict(
        zip(train_dataset.env_names, train_dataset.action_dims)
    )
    val_dataset = MultiEnvironmentDataset(
        val_datasets,
        max_action_dim=train_dataset.max_action_dim,
        action_dims=action_dims,
    )

    num_environments = len(train_dataset.env_names)
    total_batch_size = (
        num_environments * int(per_environment_batch_size)
    )
    train_generator = torch.Generator().manual_seed(int(sampler_seed))
    val_generator = torch.Generator().manual_seed(int(sampler_seed) + 1)
    train_sampler = BalancedEnvironmentBatchSampler(
        train_dataset,
        batch_size=total_batch_size,
        num_batches=int(steps_per_epoch),
        shuffle=True,
        generator=train_generator,
    )
    val_sampler = BalancedEnvironmentBatchSampler(
        val_dataset,
        batch_size=total_batch_size,
        num_batches=int(validation_steps),
        shuffle=False,
        generator=val_generator,
    )

    kwargs = _loader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        **kwargs,
    )
    exposure_metadata = {
        'per_environment_batch_size': int(per_environment_batch_size),
        'steps_per_epoch': int(steps_per_epoch),
        'validation_steps': int(validation_steps),
        'total_batch_size': total_batch_size,
        'samples_per_environment_per_epoch': (
            int(per_environment_batch_size) * int(steps_per_epoch)
        ),
    }
    return DesignV0Data(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        split_metadata=deepcopy(split_metadata),
        exposure_metadata=exposure_metadata,
    )


class DesignV0TrainingModule(pl.LightningModule):
    """Lightning adapter for the fixed Design v0 recursive objective."""

    def __init__(
        self,
        core: DesignV0Core,
        *,
        optimizer_config: Mapping[str, Any],
        checkpoint_metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.objective = DesignV0Objective(core)
        self.optimizer_config = dict(optimizer_config)
        self.checkpoint_metadata = deepcopy(dict(checkpoint_metadata))

    @property
    def core(self) -> DesignV0Core:
        return self.objective.core

    def forward(self, batch: dict[str, torch.Tensor]):
        return self.objective(
            batch['pixels'],
            batch['action'],
            batch['action_mask'],
            batch['env_id'],
        )

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        output = self(batch)
        self.log(
            f'{stage}/loss',
            output['loss'],
            on_step=stage == 'train',
            on_epoch=True,
            sync_dist=False,
            batch_size=batch['pixels'].shape[0],
        )
        for horizon, value in enumerate(output['per_horizon_mse'], start=1):
            self.log(
                f'{stage}/mse_h{horizon}',
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                batch_size=batch['pixels'].shape[0],
            )
        return output['loss']

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, 'train')

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, 'validation')

    def configure_optimizers(self):
        config = dict(self.optimizer_config)
        optimizer_type = config.pop('type', None)
        if not optimizer_type:
            raise ValueError('optimizer.type must be configured')
        try:
            optimizer_class = getattr(torch.optim, optimizer_type)
        except AttributeError as exc:
            raise ValueError(
                f'Unknown torch optimizer type {optimizer_type!r}'
            ) from exc

        trainable = [
            parameter
            for parameter in self.core.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError('Design v0 core has no trainable parameters')
        return optimizer_class(trainable, **config)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint['design_v0_metadata'] = deepcopy(
            self.checkpoint_metadata
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        saved = checkpoint.get('design_v0_metadata')
        if saved is None:
            raise ValueError('Checkpoint is missing design_v0_metadata')
        current = self.checkpoint_metadata
        if saved.get('env_to_id') != current.get('env_to_id'):
            raise ValueError(
                'Checkpoint env_to_id does not match the current run'
            )
        saved_split = _resume_split_identity(saved)
        current_split = _resume_split_identity(current)
        if saved_split != current_split:
            raise ValueError(
                'Checkpoint artifact/split identity does not match '
                'the current run'
            )


def _resume_split_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    environments = metadata.get('split', {}).get('environments', {})
    identity = {}
    for name, spec in environments.items():
        identity[name] = {
            key: spec.get(key) for key in _RESUME_SPLIT_KEYS
        }
    return identity


def effective_action_dim(raw_action_dim: int, frameskip: int) -> int:
    """Return the flattened model action width ``raw * frameskip``."""
    if int(raw_action_dim) <= 0 or int(frameskip) <= 0:
        raise ValueError(
            'raw_action_dim and frameskip must be positive, '
            f'got {raw_action_dim}, {frameskip}'
        )
    return int(raw_action_dim) * int(frameskip)


def validate_environment_action_dims(
    environment_name: str,
    *,
    raw_action_dim: int,
    clip_action_dim: int,
    frameskip: int,
) -> dict[str, int]:
    """Check raw env dim, then clip dim == raw * frameskip."""
    expected_raw = EXPECTED_ACTION_DIMS.get(environment_name)
    if expected_raw is not None and int(raw_action_dim) != int(expected_raw):
        raise ValueError(
            f'{environment_name} raw action dim mismatch: '
            f'got {raw_action_dim}, expected {expected_raw}'
        )
    effective = effective_action_dim(raw_action_dim, frameskip)
    if int(clip_action_dim) != effective:
        raise ValueError(
            f'{environment_name} effective action dim mismatch: '
            f'clip={clip_action_dim}, expected '
            f'{raw_action_dim}*{frameskip}={effective}'
        )
    return {
        'raw_action_dim': int(raw_action_dim),
        'frameskip': int(frameskip),
        'effective_action_dim': effective,
        'action_block_dim': effective,
    }


def _action_last_dim(action: Any) -> int:
    if not hasattr(action, 'shape') or len(action.shape) == 0:
        raise ValueError('action must have a feature axis')
    return int(action.shape[-1])


def _infer_raw_action_dim(dataset: Dataset) -> int:
    if not hasattr(dataset, 'load_episode'):
        raise TypeError(
            'dataset must expose load_episode to read raw action dim'
        )
    episode = dataset.load_episode(0)
    if 'action' not in episode:
        raise KeyError('episode is missing action')
    return _action_last_dim(episode['action'])


def _infer_clip_action_dim(dataset: Dataset) -> int:
    sample = dataset[0]
    if 'action' not in sample:
        raise KeyError('dataset sample is missing action')
    return _action_last_dim(sample['action'])


def load_environment_datasets(
    cfg: DictConfig,
) -> tuple[dict[str, Dataset], dict[str, dict[str, Any]]]:
    """Load canonical artifacts and preprocess the ordered environments."""
    datasets: dict[str, Dataset] = {}
    specifications: dict[str, dict[str, Any]] = {}
    cache_dir = cfg.get('cache_dir')
    if cache_dir is None or cache_dir == '':
        cache_dir = os.environ.get('STABLEWM_HOME')
    clip_steps = int(cfg.wm.history_size + cfg.wm.horizon)

    for environment_name, environment_cfg in cfg.data.environments.items():
        specification = OmegaConf.to_container(
            environment_cfg, resolve=True
        )
        dataset_name = specification.pop('name')
        split_manifest_id = specification.pop('split_manifest')
        specification['num_steps'] = clip_steps
        specification.setdefault('keys_to_load', ['pixels', 'action'])
        artifact_path = resolve_dataset_path(dataset_name, cache_dir)
        split_path = resolve_dataset_path(split_manifest_id, cache_dir)
        dataset = swm.data.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            transform=None,
            **specification,
        )
        frameskip = int(specification.get('frameskip', 1))
        raw_action_dim = _infer_raw_action_dim(dataset)
        clip_action_dim = _infer_clip_action_dim(dataset)
        action_dims = validate_environment_action_dims(
            environment_name,
            raw_action_dim=raw_action_dim,
            clip_action_dim=clip_action_dim,
            frameskip=frameskip,
        )
        dataset.transform = get_img_preprocessor(
            image_size=int(cfg.image_size)
        )
        datasets[environment_name] = dataset
        specifications[environment_name] = {
            'name': dataset_name,
            'artifact_id': dataset_name,
            'artifact_path': str(artifact_path),
            'split_manifest_id': split_manifest_id,
            'split_manifest_path': str(split_path),
            **action_dims,
            **specification,
        }
    return datasets, specifications


def build_checkpoint_metadata(
    cfg: DictConfig,
    data: DesignV0Data,
    dataset_specifications: Mapping[str, Mapping[str, Any]],
    core: DesignV0Core,
) -> dict[str, Any]:
    """Build the metadata required to reproduce and safely resume a run."""
    environment_names = list(data.train_dataset.env_names)
    effective_action_dims = dict(
        zip(environment_names, data.train_dataset.action_dims)
    )
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    datasets_meta = {}
    raw_action_dims = {}
    for name in environment_names:
        spec = dict(dataset_specifications[name])
        split_env = data.split_metadata['environments'][name]
        frameskip = int(spec.get('frameskip', 1))
        raw_dim = int(
            spec.get(
                'raw_action_dim',
                effective_action_dims[name] // max(frameskip, 1),
            )
        )
        effective = int(
            spec.get(
                'effective_action_dim',
                effective_action_dims[name],
            )
        )
        raw_action_dims[name] = raw_dim
        datasets_meta[name] = {
            **spec,
            'artifact_id': split_env.get('artifact_id', spec.get('name')),
            'artifact_path': split_env.get(
                'artifact_path', spec.get('artifact_path')
            ),
            'artifact_sha256': split_env.get('artifact_sha256'),
            'split_manifest_id': split_env.get('split_manifest_id'),
            'split_manifest_sha256': split_env.get(
                'split_manifest_sha256'
            ),
            'train_episodes': split_env.get('train_episodes'),
            'val_episodes': split_env.get('val_episodes'),
            'raw_action_dim': raw_dim,
            'frameskip': frameskip,
            'effective_action_dim': effective,
            'action_block_dim': int(
                spec.get('action_block_dim', effective)
            ),
        }
    return {
        'schema_version': METADATA_VERSION,
        'env_names': environment_names,
        'env_to_id': dict(data.train_dataset.env_to_id),
        'datasets': datasets_meta,
        'raw_action_dims': raw_action_dims,
        'effective_action_dims': effective_action_dims,
        'action_dims': effective_action_dims,
        'max_action_dim': data.train_dataset.max_action_dim,
        'history_size': core.history_size,
        'horizon': int(cfg.wm.horizon),
        'backbone': str(cfg.backbone.name),
        'latent_dim': core.latent_dim,
        'action_embedding_dim': core.action_encoder.embedding_dim,
        'environment_embedding_dim': (
            core.environment_embedding.embedding_dim
        ),
        'dynamics_hidden_dim': core.dynamics.fc1.out_features,
        'image_size': int(cfg.image_size),
        'preprocessing': 'imagenet_to_image_resize',
        'split': deepcopy(data.split_metadata),
        'exposure': deepcopy(data.exposure_metadata),
        'sampler_seed': int(cfg.loader.sampler_seed),
        'run_seed': int(cfg.seed),
        'resolved_config': resolved_config,
    }


def _validate_single_process(cfg: DictConfig) -> None:
    devices = cfg.trainer.get('devices', 1)
    if isinstance(devices, int):
        count = devices
    elif isinstance(devices, (list, tuple)):
        count = len(devices)
    else:
        raise ValueError(
            'Design v0 balanced training currently requires trainer.devices=1'
        )
    if count != 1:
        raise ValueError(
            'Design v0 balanced training currently supports one process only'
        )


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='design_v0',
)
def run(cfg: DictConfig) -> None:
    """Run per-environment or balanced joint Design v0 training."""
    _validate_single_process(cfg)
    pl.seed_everything(int(cfg.seed), workers=True)

    datasets, dataset_specifications = load_environment_datasets(cfg)
    manifests = {}
    artifact_ids = {}
    artifact_paths = {}
    split_manifest_ids = {}
    for name, specification in dataset_specifications.items():
        payload = load_episode_split_manifest(
            specification['split_manifest_path']
        )
        manifests[name] = payload
        artifact_ids[name] = specification['artifact_id']
        artifact_paths[name] = specification['artifact_path']
        split_manifest_ids[name] = specification['split_manifest_id']
    train_datasets, val_datasets, split_metadata = (
        split_datasets_with_episode_manifests(
            datasets,
            manifests,
            artifact_ids=artifact_ids,
            artifact_paths=artifact_paths,
            split_manifest_ids=split_manifest_ids,
        )
    )
    data = build_design_v0_data(
        train_datasets,
        val_datasets,
        split_metadata=split_metadata,
        per_environment_batch_size=int(
            cfg.loader.per_environment_batch_size
        ),
        steps_per_epoch=int(cfg.loader.steps_per_epoch),
        validation_steps=int(cfg.loader.validation_steps),
        sampler_seed=int(cfg.loader.sampler_seed),
        num_workers=int(cfg.loader.num_workers),
        pin_memory=bool(cfg.loader.pin_memory),
        persistent_workers=bool(cfg.loader.persistent_workers),
        prefetch_factor=cfg.loader.get('prefetch_factor'),
    )

    visual_encoder = FrozenVisualEncoder.from_pretrained(
        str(cfg.backbone.name)
    )
    core = DesignV0Core(
        visual_encoder,
        history_size=int(cfg.wm.history_size),
        max_action_dim=data.train_dataset.max_action_dim,
        action_embedding_dim=int(cfg.model.action_embedding_dim),
        num_environments=len(data.train_dataset.env_names),
        environment_embedding_dim=int(
            cfg.model.environment_embedding_dim
        ),
        dynamics_hidden_dim=int(cfg.model.dynamics_hidden_dim),
    )
    metadata = build_checkpoint_metadata(
        cfg, data, dataset_specifications, core
    )
    module = DesignV0TrainingModule(
        core,
        optimizer_config=OmegaConf.to_container(
            cfg.optimizer, resolve=True
        ),
        checkpoint_metadata=metadata,
    )

    with open_dict(cfg):
        cfg.runtime = metadata

    run_id = cfg.get('subdir') or cfg.output_model_name
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
        str(run_id),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'config.yaml')

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**OmegaConf.to_container(cfg.wandb.config))

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename='epoch-{epoch:04d}',
        save_last=True,
        save_top_k=-1,
        every_n_epochs=1,
    )
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_config['use_distributed_sampler'] = False
    trainer = pl.Trainer(
        **trainer_config,
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    data_module = spt.data.DataModule(
        train=data.train_loader,
        val=data.val_loader,
    )
    resume_checkpoint = cfg.get('resume_checkpoint')
    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=resume_checkpoint,
    )


if __name__ == '__main__':
    run()
