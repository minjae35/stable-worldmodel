"""Joint LeWM training on PushT, TwoRoom, and OGBench Cube.

The trunk matches ``scripts/train/lewm.py``: CLS latent, next-latent
MSE, and SIGReg. The only added pieces are one action adapter per
environment and an environment embedding on the AdaLN condition.
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.data.multienv import (
    BalancedEnvBatchSampler,
    ConcatEnvDataset,
    EnvTaggedDataset,
    collate_by_env,
)
from stable_worldmodel.wm.lewm.generalist import generalist_loss
from stable_worldmodel.wm.utils import save_pretrained

LOG = logging.getLogger('stage2a')
REPO_ROOT = Path(__file__).resolve().parents[2]


def _rank() -> int:
    raw = os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0'))
    return int(raw)


def _is_main() -> bool:
    # Lightning's parent process is rank 0 and only receives LOCAL_RANK
    # once fit() starts. Children already have it when the script restarts.
    return _rank() == 0


def _fitting_per_env(lengths: dict, per_env: int, world: int) -> int:
    shortest = min(lengths.values())
    if shortest >= per_env * world:
        return per_env
    fit = shortest // world
    if fit < 1:
        raise ValueError(
            f'validation split ({shortest}) is smaller than one clip '
            f'per GPU (world={world})'
        )
    return fit


def _planned_world(cfg) -> int:
    devices = cfg.trainer.get('devices', 1)
    if str(devices) == 'auto':
        return max(1, torch.cuda.device_count())
    if isinstance(devices, (list, tuple)):
        return max(1, len(list(devices)))
    return max(1, int(devices))


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = spt.data.transforms.Resize(img_size, source=source, target=target)
    return spt.data.transforms.Compose(to_image, resize)


class StopAfterEpoch(Callback):
    """Finish one launch early without shortening the 100-epoch cosine.

    ``trainer.max_epochs`` still defines the scheduler horizon. This only
    sets ``should_stop`` after the requested epoch's validation. Sanity
    validation does not count, and a later launch can pass a higher
    value to continue from the same checkpoint.
    """

    def __init__(self, epoch: int):
        super().__init__()
        self.epoch = int(epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        del pl_module
        if trainer.sanity_checking:
            return
        completed = int(trainer.current_epoch) + 1
        if completed < self.epoch:
            return
        trainer.should_stop = True
        if trainer.is_global_zero:
            LOG.info(
                'stop_after_epoch=%s reached at epoch %s global_step=%s',
                self.epoch,
                completed,
                int(trainer.global_step),
            )


class StopAtGlobalStep(Callback):
    """Stop at a global step without shortening the cosine horizon.

    ``trainer.max_steps`` stays unset, so the scheduler length is still
    ``max_epochs`` times the epoch. This only ends the current launch.
    """

    def __init__(self, step: int, snapshot_steps: list[int], snapshot_dir: Path):
        super().__init__()
        self.step = int(step)
        self.snapshot_steps = {int(value) for value in snapshot_steps}
        self.snapshot_dir = Path(snapshot_dir)

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if int(trainer.global_step) == 0:
            path = self.snapshot_dir / 'weights_step_0.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pl_module.model.state_dict(), path)
            LOG.info('snapshot %s', path)
        sched = trainer.lr_scheduler_configs[0].scheduler
        lr = trainer.optimizers[0].param_groups[0]['lr']
        LOG.info(
            'restored global_step=%s last_epoch=%s max_steps=%s '
            'warmup=%s lr=%s',
            int(trainer.global_step),
            int(sched.last_epoch),
            int(sched.max_steps),
            int(sched.warmup_steps),
            lr,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        step = int(trainer.global_step)
        if trainer.is_global_zero and step in self.snapshot_steps:
            path = self.snapshot_dir / f'weights_step_{step}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pl_module.model.state_dict(), path)
            LOG.info('snapshot %s', path)
        if step < self.step:
            return
        trainer.should_stop = True
        if trainer.is_global_zero:
            dest = self.snapshot_dir / 'checkpoints' / 'resume_stopped.ckpt'
            dest.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(dest))
            LOG.info('stop_at_global_step=%s reached at %s', self.step, step)


class SaveCkptCallback(Callback):
    """Save a loadable checkpoint after each epoch."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        if self.epoch_interval <= 0:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def _loader_kwargs(cfg) -> dict:
    kwargs = {
        'num_workers': cfg.loader.num_workers,
        'pin_memory': cfg.loader.pin_memory,
        'collate_fn': collate_by_env,
    }
    if cfg.loader.num_workers > 0:
        kwargs['persistent_workers'] = cfg.loader.persistent_workers
        kwargs['prefetch_factor'] = cfg.loader.prefetch_factor
    return kwargs


def _fit_action_normalizer(dataset, frameskip: int):
    """Z-score the raw per-step action, as ``scripts/train/lewm.py`` does.

    The dataset reshape that follows packs ``frameskip`` normalized
    actions into one dynamics step. Stats are fit on the full column,
    before the clip split.
    """
    normalizer = get_column_normalizer(dataset, 'action', 'action')
    # WrapTorchTransform stores the fitted scaler on ``lambd``.
    scaler = normalizer.lambd
    raw_dim = int(dataset.get_dim('action'))
    return normalizer, {
        'raw_action_dim': raw_dim,
        'frameskip': int(frameskip),
        'blocked_action_dim': int(frameskip) * raw_dim,
        'mean': scaler.mean.reshape(-1).tolist(),
        'std': scaler.std.reshape(-1).tolist(),
    }


class _CachedClipSet(torch.utils.data.Dataset):
    """Repeated synthetic clips. The bench measures the model, not disk."""

    def __init__(self, n: int, pixels: torch.Tensor, actions: torch.Tensor):
        self.n = int(n)
        self.pixels = pixels
        self.actions = actions

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        slot = int(idx) % self.pixels.shape[0]
        return {
            'pixels': self.pixels[slot],
            'action': self.actions[slot],
        }


def _build_synthetic(cfg):
    steps = int(cfg.wm.num_preds) + int(cfg.wm.history_size)
    clips = int(cfg.data.synthetic_clips)
    pool = int(cfg.data.synthetic_pool)
    train_n = int(clips * float(cfg.train_split))
    val_n = clips - train_n
    if train_n < 1 or val_n < 1:
        raise ValueError(f'synthetic split is empty: {train_n} / {val_n}')
    trains = {}
    vals = {}
    stats = {}
    img = int(cfg.img_size)
    for index, name in enumerate(cfg.data.env_order):
        width = int(cfg.model.action_adapters[name].input_dim)
        generator = torch.Generator().manual_seed(int(cfg.seed) + index)
        pixels = torch.rand(pool, steps, 3, img, img, generator=generator)
        actions = torch.randn(pool, steps, width, generator=generator)
        trains[name] = EnvTaggedDataset(
            _CachedClipSet(train_n, pixels, actions), name
        )
        vals[name] = EnvTaggedDataset(
            _CachedClipSet(val_n, pixels, actions), name
        )
        stats[name] = {
            'blocked_action_dim': width,
            'note': 'synthetic cached clips, actions are not normalized',
        }
        if _is_main():
            print(
                f'  synthetic {name}: blocked action {width}, '
                f'train {train_n}, val {val_n}, pool {pool}'
            )
    return ConcatEnvDataset(trains), ConcatEnvDataset(vals), stats


def _build_splits(cfg):
    if str(cfg.data.get('source', 'lance')) == 'synthetic':
        return _build_synthetic(cfg)
    trains = {}
    vals = {}
    stats = {}
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    for name in cfg.data.env_order:
        env_cfg = OmegaConf.to_container(cfg.data.envs[name], resolve=True)
        dataset_name = env_cfg.pop('name')
        frameskip = int(env_cfg['frameskip'])
        where = (
            'local cache: ' + cache_dir if cache_dir else 'default location'
        )
        if _is_main():
            print(f'Loading {name} "{dataset_name}" from {where}')
        dataset = swm.data.load_dataset(
            dataset_name,
            transform=None,
            cache_dir=cache_dir,
            **env_cfg,
        )
        normalizer, info = _fit_action_normalizer(dataset, frameskip)
        stats[name] = info
        blocked = info['blocked_action_dim']
        with open_dict(cfg):
            cfg.model.action_adapters[name].input_dim = blocked
        if _is_main():
            print(
                f'  {name}: raw action {info["raw_action_dim"]}, '
                f'frameskip {frameskip}, blocked action {blocked}, '
                f'clips {len(dataset)}'
            )
        dataset.transform = spt.data.transforms.Compose(
            get_img_preprocessor('pixels', 'pixels', cfg.img_size),
            normalizer,
        )
        generator = torch.Generator().manual_seed(cfg.seed)
        train_set, val_set = spt.data.random_split(
            dataset,
            lengths=[cfg.train_split, 1 - cfg.train_split],
            generator=generator,
        )
        trains[name] = EnvTaggedDataset(train_set, name)
        vals[name] = EnvTaggedDataset(val_set, name)
    return ConcatEnvDataset(trains), ConcatEnvDataset(vals), stats


def _profile_comm(prof) -> tuple[int, int]:
    comm = 0
    total = 0
    for evt in prof.key_averages():
        cuda = int(getattr(evt, 'self_device_time_total', 0) or 0)
        if cuda <= 0:
            cuda = int(getattr(evt, 'self_cuda_time_total', 0) or 0)
        total += cuda
        name = evt.key.lower()
        if (
            'nccl' in name
            or 'allreduce' in name
            or 'all_reduce' in name
            or 'allgather' in name
            or 'all_gather' in name
        ):
            comm += cuda
    return comm, total


def _metric(trainer, key: str):
    value = trainer.callback_metrics.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    try:
        return float(value)
    except TypeError:
        return None


def _rewind_interrupted_epoch(checkpoint: dict) -> bool:
    """Make a checkpoint that stopped inside an epoch resumable.

    Lightning increments ``epoch_progress.processed`` before the
    epoch-end checkpoint, including when ``max_steps`` or a signal
    stopped the epoch with batches still left. Those counters then
    match a finished epoch, so reloading the last epoch returns
    without taking another step. A finished epoch has
    ``is_last_batch`` set and is left alone.
    """
    fit = (checkpoint.get('loops') or {}).get('fit_loop') or {}
    epoch = fit.get('epoch_progress')
    batch = fit.get('epoch_loop.batch_progress')
    if not isinstance(epoch, dict) or not isinstance(batch, dict):
        return False
    current = batch.get('current') or {}
    if int(current.get('completed', 0)) <= 0:
        return False
    if batch.get('is_last_batch', False):
        return False
    changed = False
    for side in ('current', 'total'):
        progress = epoch.get(side)
        if not isinstance(progress, dict):
            continue
        processed = int(progress.get('processed', 0))
        completed = int(progress.get('completed', 0))
        started = int(progress.get('started', 0))
        if (
            processed == completed + 1
            and processed == started
            and processed > 0
        ):
            progress['processed'] = completed
            changed = True
    return changed


def _align_scheduler_horizon(trainer, max_steps, warmup_steps) -> None:
    """Keep the cosine horizon on the plan this process was launched with.

    Loading a checkpoint replaces ``max_steps`` with the value from the
    run that wrote it. A short ``max_steps`` interrupt would otherwise
    freeze that shorter horizon, and the cosine keeps going past it.
    ``last_epoch`` stays as saved.
    """
    if max_steps is None:
        return
    configs = getattr(trainer, 'lr_scheduler_configs', None) or []
    for config in configs:
        sched = config.scheduler
        changed = False
        if hasattr(sched, 'max_steps') and int(sched.max_steps) != int(
            max_steps
        ):
            if _is_main():
                LOG.info(
                    'scheduler horizon %s -> %s for this launch',
                    int(sched.max_steps),
                    int(max_steps),
                )
            sched.max_steps = int(max_steps)
            changed = True
        if (
            warmup_steps is not None
            and hasattr(sched, 'warmup_steps')
            and int(sched.warmup_steps) != int(warmup_steps)
        ):
            if _is_main():
                LOG.info(
                    'scheduler warmup %s -> %s for this launch',
                    int(sched.warmup_steps),
                    int(warmup_steps),
                )
            sched.warmup_steps = int(warmup_steps)
            changed = True
        if not changed:
            continue
        # The checkpoint restored the learning rate from the old horizon.
        # Recompute it at the saved last_epoch so the next update uses
        # this launch's cosine, without consuming a step.
        refreshed = sched.get_lr()
        for group, lr in zip(sched.optimizer.param_groups, refreshed):
            group['lr'] = lr
        sched._last_lr = list(refreshed)


class JointTrainProgress(Callback):
    """tmux tqdm bar plus a CSV that does not contain the bar itself."""

    def __init__(
        self,
        exp_dir: Path,
        env_names: list[str],
        log_every: int = 50,
        per_env: int = 1,
        world_size: int = 1,
        clip_frames: int = 4,
        schedule_max_steps: int | None = None,
        warmup_steps: int | None = None,
        bar_epochs: int | None = None,
        bar_total: int | None = None,
    ):
        super().__init__()
        self.exp_dir = Path(exp_dir)
        self.env_names = list(env_names)
        self.log_every = log_every
        self.per_env = int(per_env)
        self.world_size = int(world_size)
        self.clip_frames = int(clip_frames)
        self.schedule_max_steps = schedule_max_steps
        self.warmup_steps = warmup_steps
        self.bar_epochs = None if not bar_epochs else int(bar_epochs)
        self.bar_total = None if not bar_total else int(bar_total)
        self.enabled = False
        self.bar = None
        self.val_bar = None
        self._csv = None
        self._started = None
        self._step_t0 = None
        self._prev_end = None
        self._data_s = None
        self._step_times: list[float] = []
        self._data_times: list[float] = []
        self._comm_times: list[float] = []
        self._seen = 0
        self._grad_norm = None
        self._gpu_samples: list[str] = []
        self._gpu_stop = None
        self._gpu_thread = None
        self._prof = None
        self._nccl_us = None
        self._profile_cuda_us = None
        self._profile = os.environ.get('STAGE2A_PROFILE') == '1'

    def on_fit_start(self, trainer, pl_module):
        self.enabled = bool(trainer.is_global_zero)
        if not self.enabled:
            return
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        path = self.exp_dir / 'metrics.csv'
        new_file = not path.exists() or path.stat().st_size == 0
        self._csv = path.open('a', newline='')
        self._writer = csv.DictWriter(
            self._csv,
            fieldnames=self._fields(),
            extrasaction='ignore',
        )
        if new_file:
            self._writer.writeheader()
            self._csv.flush()
        self._started = time.time()

    def _fields(self) -> list[str]:
        fields = [
            'stage',
            'global_step',
            'epoch',
            'loss',
            'pred_loss',
            'sigreg_loss',
            'variance_loss',
            'lr',
            'it_per_s',
            'elapsed_s',
            'gpu_mem_mb',
            'gpu_peak_mb',
            'data_s',
            'step_s',
            'sigreg_comm_s',
        ]
        for name in self.env_names:
            fields.extend(
                [
                    f'{name}_loss',
                    f'{name}_pred_loss',
                    f'{name}_sigreg_loss',
                    f'{name}_variance_loss',
                    f'{name}_latent_std',
                    f'{name}_latent_dim_std_min',
                ]
            )
        return fields

    def _gpu(self) -> tuple[float | None, float | None]:
        if not torch.cuda.is_available():
            return None, None
        scale = 1024**2
        return (
            torch.cuda.memory_allocated() / scale,
            torch.cuda.max_memory_allocated() / scale,
        )

    def _row(self, trainer, stage: str) -> dict:
        lr = None
        opts = trainer.optimizers or []
        if opts:
            lr = opts[0].param_groups[0]['lr']
        mem, peak = self._gpu()
        rate = None
        if self.bar is not None:
            rate = self.bar.format_dict.get('rate')
        elapsed = None
        if self._started is not None:
            elapsed = time.time() - self._started
        row = {
            'stage': stage,
            'global_step': int(trainer.global_step),
            'epoch': int(trainer.current_epoch),
            'loss': _metric(trainer, f'{stage}/loss'),
            'pred_loss': _metric(trainer, f'{stage}/pred_loss'),
            'sigreg_loss': _metric(trainer, f'{stage}/sigreg_loss'),
            'variance_loss': _metric(trainer, f'{stage}/variance_loss'),
            'lr': lr,
            'it_per_s': rate,
            'elapsed_s': elapsed,
            'gpu_mem_mb': mem,
            'gpu_peak_mb': peak,
            'data_s': self._data_s,
            'step_s': None,
            'sigreg_comm_s': None,
        }
        for name in self.env_names:
            row[f'{name}_loss'] = _metric(trainer, f'{stage}/{name}/loss')
            row[f'{name}_pred_loss'] = _metric(
                trainer, f'{stage}/{name}/pred_loss'
            )
            row[f'{name}_sigreg_loss'] = _metric(
                trainer, f'{stage}/{name}/sigreg_loss'
            )
            row[f'{name}_variance_loss'] = _metric(
                trainer, f'{stage}/{name}/variance_loss'
            )
            row[f'{name}_latent_std'] = _metric(
                trainer, f'{stage}/{name}/latent_std'
            )
            row[f'{name}_latent_dim_std_min'] = _metric(
                trainer, f'{stage}/{name}/latent_dim_std_min'
            )
        return row

    def _overlay_step_metrics(self, row: dict, pl_module) -> None:
        metrics = getattr(pl_module, '_last_metrics', None)
        if not isinstance(metrics, dict):
            return
        for key, value in metrics.items():
            if key in ('loss', 'pred_loss', 'sigreg_loss', 'variance_loss'):
                row[key] = value
                continue
            if '/' not in key:
                continue
            name, suffix = key.split('/', 1)
            dest = f'{name}_{suffix}'
            if dest in row:
                row[dest] = value

    def _write(self, row: dict) -> None:
        if self._csv is None:
            return
        self._writer.writerow(row)
        self._csv.flush()

    def _gpu_watch(self, stop: threading.Event, sink: list) -> None:
        while not stop.is_set():
            proc = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits',
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                sink.append(proc.stdout.strip())
            stop.wait(0.4)

    def _stop_watch(self) -> None:
        if self._gpu_stop is not None:
            self._gpu_stop.set()
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=2)
            self._gpu_thread = None

    def _gpu_stats(self) -> dict:
        utils: dict[int, list[float]] = {}
        mems: dict[int, list[float]] = {}
        for blob in self._gpu_samples:
            for line in blob.splitlines():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) != 3:
                    continue
                index, util, mem = int(parts[0]), float(parts[1]), float(parts[2])
                utils.setdefault(index, []).append(util)
                mems.setdefault(index, []).append(mem)
        return {
            'gpu_util_mean': {
                str(index): sum(values) / len(values)
                for index, values in sorted(utils.items())
            },
            'gpu_mem_mib_max': {
                str(index): max(values)
                for index, values in sorted(mems.items())
            },
        }

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.enabled or self._grad_norm is not None:
            return
        opt = getattr(optimizer, 'optimizer', optimizer)
        square = 0.0
        for group in opt.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                square += float(param.grad.detach().float().pow(2).sum().cpu())
        self._grad_norm = square**0.5

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._prev_end is not None:
            self._data_s = now - self._prev_end
        if (
            self._profile
            and self._seen == 12
            and self._prof is None
            and int(trainer.max_steps or 0) >= 16
        ):
            self._prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            )
            self._prof.__enter__()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_t0 = time.perf_counter()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if _rewind_interrupted_epoch(checkpoint) and trainer.is_global_zero:
            LOG.info(
                'rewound epoch progress in the checkpoint so this '
                'interrupted epoch can resume'
            )

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if _rewind_interrupted_epoch(checkpoint) and trainer.is_global_zero:
            LOG.info(
                'rewound epoch progress loaded from an interrupted epoch'
            )

    def on_train_start(self, trainer, pl_module):
        _align_scheduler_horizon(
            trainer, self.schedule_max_steps, self.warmup_steps
        )
        if not self.enabled:
            return
        self._prev_end = time.perf_counter()
        self._gpu_stop = threading.Event()
        self._gpu_thread = threading.Thread(
            target=self._gpu_watch,
            args=(self._gpu_stop, self._gpu_samples),
            daemon=True,
        )
        self._gpu_thread.start()
        total = self.bar_total or trainer.max_steps
        if total is None or int(total) < 1:
            batches = trainer.num_training_batches
            if isinstance(batches, (list, tuple)):
                batches = batches[0]
            epochs = self.bar_epochs or trainer.max_epochs or 1
            total = int(batches) * int(epochs)
        self.bar = tqdm(
            total=int(total),
            initial=int(trainer.global_step),
            desc='Stage 2-A Joint Training',
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.1,
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_s = None
        profiling = self._prof is not None
        if self._step_t0 is not None:
            step_s = time.perf_counter() - self._step_t0
            if not profiling:
                self._step_times.append(step_s)
        if self._data_s is not None and not profiling:
            self._data_times.append(self._data_s)
        self._prev_end = time.perf_counter()
        if profiling and self._seen >= 13:
            self._prof.__exit__(None, None, None)
            self._nccl_us, self._profile_cuda_us = _profile_comm(self._prof)
            self._prof = None
        self._seen += 1
        sig = getattr(pl_module, 'sigreg', None)
        comm_s = sig.read_comm_s() if sig is not None else 0.0
        if not profiling:
            self._comm_times.append(comm_s)
        if self.bar is not None:
            self.bar.update(1)
        row = self._row(trainer, 'train')
        row['step_s'] = step_s
        row['sigreg_comm_s'] = comm_s
        self._overlay_step_metrics(row, pl_module)
        self._write(row)
        if self.bar is not None and row['loss'] is not None:
            pred = row['pred_loss']
            sig = row['sigreg_loss']
            lr = row['lr']
            self.bar.set_postfix(
                loss=f'{row["loss"]:.4f}',
                pred='?' if pred is None else f'{pred:.4f}',
                sigreg='?' if sig is None else f'{sig:.4f}',
                lr='?' if lr is None else f'{lr:.2e}',
                refresh=False,
            )
        step = int(trainer.global_step)
        if step % self.log_every == 0:
            LOG.info(
                'step %s loss=%s pred=%s sigreg=%s lr=%s '
                'pusht=%s tworoom=%s cube=%s gpu_peak_mb=%s',
                step,
                row['loss'],
                row['pred_loss'],
                row['sigreg_loss'],
                row['lr'],
                row.get('pusht_loss'),
                row.get('tworoom_loss'),
                row.get('cube_loss'),
                row['gpu_peak_mb'],
            )
        _promote_resume(trainer, self.exp_dir)

    def on_validation_start(self, trainer, pl_module):
        if not self.enabled:
            return
        total = trainer.num_val_batches
        if isinstance(total, (list, tuple)):
            total = total[0] if total else None
        if total is not None and int(total) < 0:
            total = None
        self.val_bar = tqdm(
            total=None if total is None else int(total),
            desc='Stage 2-A validation',
            file=sys.stderr,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.2,
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not self.enabled or self.val_bar is None:
            return
        self.val_bar.update(1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return
        if self.val_bar is not None:
            self.val_bar.close()
            self.val_bar = None
        self._write(self._row(trainer, 'validate'))

    def on_train_end(self, trainer, pl_module):
        if not self.enabled:
            return
        self._stop_watch()
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        if self._csv is not None:
            self._csv.close()
            self._csv = None

    def _summary(self) -> dict:
        def median(values: list[float]):
            usable = values[3:] if len(values) > 3 else values
            if not usable:
                return None
            ordered = sorted(usable)
            return ordered[len(ordered) // 2]

        step_s = median(self._step_times)
        clips = self.per_env * self.world_size * len(self.env_names)
        frames = clips * self.clip_frames
        clips_per_s = None if not step_s else clips / step_s
        fraction = None
        if self._profile_cuda_us:
            fraction = self._nccl_us / self._profile_cuda_us
        summary = {
            'world_size': self.world_size,
            'per_env_per_gpu': self.per_env,
            'effective_clips_per_env': self.per_env * self.world_size,
            'effective_clips_per_step': clips,
            'frames_per_step': frames,
            'median_step_s': step_s,
            'median_data_s': median(self._data_times),
            'median_sigreg_comm_s': median(self._comm_times),
            'steps_per_s': None if not step_s else 1.0 / step_s,
            'clips_per_s': clips_per_s,
            'frames_per_s': None if not step_s else frames / step_s,
            'timed_steps': max(0, len(self._step_times) - 3),
            'grad_norm': self._grad_norm,
            'nccl_cuda_us': self._nccl_us,
            'profile_cuda_us': self._profile_cuda_us,
            'nccl_fraction': fraction,
        }
        summary.update(self._gpu_stats())
        return summary

    def on_fit_end(self, trainer, pl_module):
        if not self.enabled:
            return
        mem, peak = self._gpu()
        status = {
            'global_step': int(trainer.global_step),
            'current_epoch': int(trainer.current_epoch),
            'gpu_mem_mb': mem,
            'gpu_peak_mb': peak,
        }
        (self.exp_dir / 'status.json').write_text(json.dumps(status, indent=2))
        summary = self._summary()
        summary['global_step'] = status['global_step']
        summary['gpu_peak_mb'] = peak
        (self.exp_dir / 'benchmark.json').write_text(
            json.dumps(summary, indent=2)
        )
        LOG.info('fit end %s', status)
        LOG.info('benchmark %s', summary)


def generalist_forward(self, batch, stage, cfg):
    loss, logs = generalist_loss(
        self.model,
        self.sigreg,
        batch,
        history_size=cfg.wm.history_size,
        num_preds=cfg.wm.num_preds,
        sigreg_weight=cfg.loss.sigreg.weight,
        detach_pred_target=bool(cfg.loss.get('detach_pred_target', False)),
        variance_weight=float(cfg.loss.variance.weight),
        variance_gamma=float(cfg.loss.variance.gamma),
        variance_eps=float(cfg.loss.variance.eps),
    )
    prefixed = {f'{stage}/{key}': value for key, value in logs.items()}
    self.log_dict(prefixed, on_step=True, sync_dist=True)
    metrics = {}
    for key, value in logs.items():
        if torch.is_tensor(value):
            metrics[key] = float(value.detach().float().cpu())
        else:
            metrics[key] = float(value)
    self._last_metrics = metrics
    return {'loss': loss}


def _promote_resume(trainer, exp_dir: Path) -> None:
    """Copy the full Lightning state to a stable path.

    stable_pretraining redirects ModelCheckpoint into a fresh cache run
    directory on every launch, so the next process cannot find that file.
    """
    dest = exp_dir / 'checkpoints' / 'resume.ckpt'
    dest.parent.mkdir(parents=True, exist_ok=True)
    found = []
    for callback in trainer.callbacks:
        if not isinstance(callback, ModelCheckpoint) or not callback.dirpath:
            continue
        folder = Path(callback.dirpath)
        for name in ('last.ckpt', 'resume.ckpt'):
            path = folder / name
            if path.is_file():
                found.append(path)
    if not found:
        if dest.is_file():
            return
        LOG.warning('no optimizer checkpoint to promote')
        return
    src = max(found, key=lambda path: path.stat().st_mtime)
    if src.resolve() == dest.resolve():
        return
    if dest.is_file() and src.stat().st_mtime <= dest.stat().st_mtime:
        return
    shutil.copy2(src, dest)
    LOG.info('resume checkpoint %s', dest)


def _save_normalizer(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic = any(isinstance(info, dict) and 'note' in info for info in stats.values())
    payload = {
        'method': 'none' if synthetic else 'zscore',
        'fit_on': (
            'not fit; synthetic benchmark clips'
            if synthetic
            else (
                'full action column before the clip split, '
                'matching scripts/train/lewm.py'
            )
        ),
        'envs': stats,
    }
    path.write_text(json.dumps(payload, indent=2))
    if _is_main():
        print(f'Wrote action normalizer to {path}')


def _experiment_dir(name: str) -> Path:
    path = REPO_ROOT / 'results' / 'stage2a' / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_log(exp_dir: Path) -> None:
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    if any(isinstance(h, logging.FileHandler) for h in LOG.handlers):
        return
    handle = logging.FileHandler(exp_dir / 'train.log')
    handle.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    )
    LOG.addHandler(handle)


@hydra.main(
    version_base=None, config_path='./config', config_name='lewm_generalist'
)
def run(cfg):
    # Rich/simple bars from stable_pretraining fight a manual tqdm.
    spt.set(progress_bar='none')
    world = _planned_world(cfg)
    rank = _rank()
    exp_dir = _experiment_dir(cfg.output_model_name)
    if _is_main():
        _file_log(exp_dir)
        LOG.info('experiment dir %s', exp_dir)
        LOG.info('rank %s world %s', rank, world)

    train_set, val_set, train_stats = _build_splits(cfg)

    per_env = int(cfg.per_env_batch_size)
    train_sampler = BalancedEnvBatchSampler(
        train_set.env_lengths(),
        per_env=per_env,
        seed=cfg.seed,
        rank=rank,
        world_size=world,
    )
    val_per_env = _fitting_per_env(val_set.env_lengths(), per_env, world)
    if val_per_env != per_env and _is_main():
        LOG.info(
            'val per-GPU batch reduced from %s to %s to fit the split',
            per_env,
            val_per_env,
        )
    val_sampler = BalancedEnvBatchSampler(
        val_set.env_lengths(),
        per_env=val_per_env,
        seed=cfg.seed + 1,
        rank=rank,
        world_size=world,
    )
    loader_kwargs = _loader_kwargs(cfg)
    train = torch.utils.data.DataLoader(
        train_set, batch_sampler=train_sampler, **loader_kwargs
    )
    val = torch.utils.data.DataLoader(
        val_set, batch_sampler=val_sampler, **loader_kwargs
    )
    shortest = min(train_set.env_lengths().values())
    covered = len(train_sampler) * per_env * world
    dropped = int(shortest) - int(covered)
    if _is_main():
        print(
            f'Balanced batches: {len(train)} train / {len(val)} val, '
            f'{per_env} clips/env/gpu, world {world}, '
            f'effective {per_env * world} clips/env/step, '
            f'dropped {dropped} clips from the shortest split'
        )

    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    world_model = hydra.utils.instantiate(cfg.model)
    shared = {
        'encoder',
        'predictor',
        'projector',
        'pred_proj',
        'env_embed',
    }
    adapter_names = set(world_model.action_encoder.adapters)
    if _is_main():
        print(
            f'Shared modules: {sorted(shared)}; '
            f'action adapters: {sorted(adapter_names)}'
        )

    total_steps = cfg.trainer.max_epochs * len(train)
    if cfg.trainer.get('max_steps', -1) not in (None, -1):
        total_steps = min(total_steps, int(cfg.trainer.max_steps))
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': max(1, total_steps),
            },
            'interval': 'epoch',
        },
    }

    module = spt.Module(
        model=world_model,
        sigreg=swm.wm.SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(generalist_forward, cfg=cfg),
        optim=optimizers,
    )

    weight_dir = (
        Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
        / cfg.output_model_name
    )
    if _is_main():
        _save_normalizer(weight_dir / 'action_normalizer.json', train_stats)
        _save_normalizer(exp_dir / 'action_normalizer.json', train_stats)
        with open(exp_dir / 'config.yaml', 'w') as handle:
            OmegaConf.save(cfg, handle)
        LOG.info(
            'schedule_steps=%s lr=%s effective_per_env=%s dropped=%s '
            'detach_pred_target=%s variance_weight=%s variance_gamma=%s',
            total_steps,
            cfg.optimizer.lr,
            per_env * world,
            dropped,
            bool(cfg.loss.get('detach_pred_target', False)),
            float(cfg.loss.variance.weight),
            float(cfg.loss.variance.gamma),
        )
        topology = {
            'data_source': str(cfg.data.get('source', 'lance')),
            'world_size': world,
            'per_env_per_gpu': per_env,
            'effective_per_env': per_env * world,
            'local_steps_per_epoch': len(train),
            'schedule_max_steps': int(total_steps),
            'lr': float(cfg.optimizer.lr),
            'lr_scaled_with_world_size': False,
            'sync_batchnorm': world > 1,
            'shortest_env_clips': int(shortest),
            'dropped_clips_shortest_env': dropped,
        }
        (exp_dir / 'topology.json').write_text(json.dumps(topology, indent=2))

    logger = None
    if cfg.wandb.enabled and _is_main():
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    resume_path = exp_dir / 'checkpoints' / 'resume.ckpt'
    every = int(cfg.checkpoint_every_n_steps)
    epoch_interval = int(cfg.weights_every_n_epochs)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.setdefault('num_sanity_val_steps', 1)
    trainer_cfg['use_distributed_sampler'] = False
    if world > 1:
        # DDP averages gradients. SIGReg reduces the global sums first,
        # so that average matches the single-process global statistic.
        trainer_cfg['strategy'] = 'ddp'
        trainer_cfg['sync_batchnorm'] = True
    stop_after = cfg.get('stop_after_epoch', None)
    stop_at = cfg.get('stop_at_global_step', None)
    raw_snapshots = cfg.get('snapshot_at_global_steps', None) or []
    snapshot_steps = [int(step) for step in raw_snapshots]
    callbacks = [
        SaveCkptCallback(
            run_name=cfg.output_model_name,
            cfg=cfg.model,
            epoch_interval=epoch_interval,
        ),
        ModelCheckpoint(
            dirpath=str(exp_dir / 'checkpoints'),
            filename='resume',
            save_last=False,
            every_n_train_steps=every,
            save_on_train_epoch_end=epoch_interval > 0,
            enable_version_counter=False,
        ),
        JointTrainProgress(
            exp_dir,
            env_names=list(cfg.data.env_order),
            log_every=int(cfg.log_every_n_steps),
            per_env=per_env,
            world_size=world,
            clip_frames=int(cfg.wm.num_preds) + int(cfg.wm.history_size),
            schedule_max_steps=int(total_steps),
            warmup_steps=max(1, int(0.01 * total_steps)),
            bar_epochs=None if stop_after in (None, 0) else int(stop_after),
            bar_total=None if stop_at in (None, 0) else int(stop_at),
        ),
    ]
    if stop_at not in (None, 0):
        callbacks.append(
            StopAtGlobalStep(int(stop_at), snapshot_steps, exp_dir)
        )
        if _is_main():
            LOG.info(
                'stop_at_global_step=%s with schedule_steps=%s',
                int(stop_at),
                int(total_steps),
            )
    if stop_after not in (None, 0):
        callbacks.append(StopAfterEpoch(int(stop_after)))
        if _is_main():
            LOG.info(
                'stop_after_epoch=%s with schedule_steps=%s',
                int(stop_after),
                int(total_steps),
            )
    trainer = pl.Trainer(
        **trainer_cfg,
        default_root_dir=str(exp_dir),
        enable_progress_bar=False,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        seed=int(cfg.seed),
        ckpt_path=str(resume_path.resolve()) if resume_path.exists() else None,
        weights_only=False,
    )
    manager()
    if trainer.is_global_zero:
        _promote_resume(trainer, exp_dir)
        save_pretrained(
            module.model,
            run_name=cfg.output_model_name,
            config=cfg.model,
            filename='weights_last.pt',
        )
    return


if __name__ == '__main__':
    run()
