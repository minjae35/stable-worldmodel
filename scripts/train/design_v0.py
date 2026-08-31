"""Train the independent Design v0 model on one or more environments."""

from __future__ import annotations

import hashlib
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


SPLIT_VERSION = 'design_v0_per_environment_v1'
METADATA_VERSION = 1


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


def environment_split_seed(base_seed: int, environment_name: str) -> int:
    """Derive a stable per-environment seed independent of run composition."""
    payload = (
        f'{SPLIT_VERSION}:{int(base_seed)}:{environment_name}'.encode()
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder='little') % (2**63 - 1)


def _indices_digest(indices: list[int]) -> str:
    encoded = ','.join(str(index) for index in indices).encode()
    return hashlib.sha256(encoded).hexdigest()


def split_environment_datasets(
    datasets: Mapping[str, Dataset],
    *,
    train_fraction: float,
    split_seed: int,
    dataset_identifiers: Mapping[str, str] | None = None,
) -> tuple[
    dict[str, Subset],
    dict[str, Subset],
    dict[str, Any],
]:
    """Split every environment deterministically with an independent RNG."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('train_fraction must be strictly between 0 and 1')

    train_datasets: dict[str, Subset] = {}
    val_datasets: dict[str, Subset] = {}
    environments: dict[str, Any] = {}

    for environment_name, dataset in datasets.items():
        length = len(dataset)
        train_length = int(length * train_fraction)
        val_length = length - train_length
        if train_length <= 0 or val_length <= 0:
            raise ValueError(
                f'Environment {environment_name!r} needs non-empty train and '
                f'validation splits, got length={length}, '
                f'train={train_length}, val={val_length}'
            )

        derived_seed = environment_split_seed(
            split_seed, environment_name
        )
        generator = torch.Generator().manual_seed(derived_seed)
        permutation = torch.randperm(
            length, generator=generator
        ).tolist()
        train_indices = permutation[:train_length]
        val_indices = permutation[train_length:]

        train_datasets[environment_name] = Subset(
            dataset, train_indices
        )
        val_datasets[environment_name] = Subset(dataset, val_indices)
        environments[environment_name] = {
            'dataset': (
                dataset_identifiers[environment_name]
                if dataset_identifiers is not None
                else environment_name
            ),
            'dataset_length': length,
            'derived_seed': derived_seed,
            'train_length': train_length,
            'validation_length': val_length,
            'train_indices_sha256': _indices_digest(train_indices),
            'validation_indices_sha256': _indices_digest(val_indices),
        }

    metadata = {
        'version': SPLIT_VERSION,
        'base_seed': int(split_seed),
        'train_fraction': float(train_fraction),
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
        for key in ('env_to_id', 'split'):
            if saved.get(key) != self.checkpoint_metadata.get(key):
                raise ValueError(
                    f'Checkpoint {key} does not match the current run'
                )


def load_environment_datasets(
    cfg: DictConfig,
) -> tuple[dict[str, Dataset], dict[str, dict[str, Any]]]:
    """Load and preprocess the ordered environment datasets from config."""
    datasets: dict[str, Dataset] = {}
    specifications: dict[str, dict[str, Any]] = {}
    cache_dir = cfg.get('cache_dir') or os.environ.get(
        'LOCAL_DATASET_DIR'
    )
    clip_steps = int(cfg.wm.history_size + cfg.wm.horizon)

    for environment_name, environment_cfg in cfg.data.environments.items():
        specification = OmegaConf.to_container(
            environment_cfg, resolve=True
        )
        dataset_name = specification.pop('name')
        specification['num_steps'] = clip_steps
        specification.setdefault('keys_to_load', ['pixels', 'action'])
        dataset = swm.data.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            transform=None,
            **specification,
        )
        dataset.transform = get_img_preprocessor(
            image_size=int(cfg.image_size)
        )
        datasets[environment_name] = dataset
        specifications[environment_name] = {
            'name': dataset_name,
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
    action_dims = dict(
        zip(environment_names, data.train_dataset.action_dims)
    )
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    return {
        'schema_version': METADATA_VERSION,
        'env_names': environment_names,
        'env_to_id': dict(data.train_dataset.env_to_id),
        'datasets': {
            name: dict(dataset_specifications[name])
            for name in environment_names
        },
        'action_dims': action_dims,
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
    dataset_identifiers = {
        name: specification['name']
        for name, specification in dataset_specifications.items()
    }
    train_datasets, val_datasets, split_metadata = (
        split_environment_datasets(
            datasets,
            train_fraction=float(cfg.split.train_fraction),
            split_seed=int(cfg.split.seed),
            dataset_identifiers=dataset_identifiers,
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
