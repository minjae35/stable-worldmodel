"""Train and evaluate official PreJEPA with RGB + action only.

The predictor, loss, optimizer, and schedule match scripts/train/prejepa.yaml.
The proprio encoder is omitted. Training clips are the experiment's PushT
train episodes, so the fixed 10 val pairs stay out of the fit.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void':
    del os.environ['NVIDIA_VISIBLE_DEVICES']

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightning as pl  # noqa: E402
import numpy as np  # noqa: E402
import stable_pretraining as spt  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from lightning.pytorch.callbacks import Callback  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from rdwm.exp1.config import DEFAULT_CONFIG, env_path, load_config  # noqa: E402
from rdwm.exp1.data import split_episodes  # noqa: E402
from rdwm.exp1.progress import Progress  # noqa: E402

RUN = Path('/workspace/rdwm_runs/exp1/dinowm_rgb')
LOG = Path('/workspace/rdwm_runs/exp1/logs/dinowm_rgb/train.log')
WEIGHTS = RUN / 'weights_epoch_10.pt'


def _official():
    path = ROOT / 'scripts' / 'train' / 'prejepa.py'
    spec = importlib.util.spec_from_file_location('prejepa_train', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'scripts' / 'train' / 'config')):
        return compose(config_name='prejepa', overrides=['~wm.encoding.proprio'])


class _Bar(Callback):
    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self.bar = None

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        total = int(trainer.estimated_stepping_batches)
        self.bar = Progress(total, 0, 'DINO-WM RGB | PushT | train', self.log_path, unit='it/s')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.bar is None:
            return
        loss = outputs.get('loss') if isinstance(outputs, dict) else None
        pixels = outputs.get('pixels_loss') if isinstance(outputs, dict) else None
        text = f'epoch {trainer.current_epoch + 1}/{trainer.max_epochs}'
        if torch.is_tensor(loss):
            text += f' | loss {float(loss.detach()):.4f}'
        if torch.is_tensor(pixels):
            text += f' | pixels {float(pixels.detach()):.4f}'
        self.bar.desc = f'DINO-WM RGB | PushT | epoch {trainer.current_epoch + 1}/{trainer.max_epochs}'
        self.bar.set_postfix(text)
        self.bar.update(int(trainer.global_step))

    def on_train_end(self, trainer, pl_module):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class _Save(Callback):
    def __init__(self, run_dir: Path, cfg):
        super().__init__()
        self.run_dir = run_dir
        self.cfg = cfg

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch != trainer.max_epochs and epoch % 5 != 0:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model.state_dict(), self.run_dir / f'weights_epoch_{epoch}.pt')
        config = OmegaConf.to_container(self.cfg, resolve=True)
        (self.run_dir / 'config.json').write_text(json.dumps(config, indent=2))


def _train_indices(dataset, exp_cfg) -> list[int]:
    splits = split_episodes(len(dataset.lengths), exp_cfg['data']['split'])
    train_eps = set(int(ep) for ep in splits['train'])
    return [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in train_eps]


def train() -> None:
    limit = int(os.environ.get('RDWM_LIMIT_BATCHES', '0'))
    run = Path('/tmp/dinowm_rgb_smoke') if limit else RUN
    log = Path('/tmp/dinowm_rgb_smoke.log') if limit else LOG
    weights = run / 'weights_epoch_10.pt'
    if not limit and weights.exists() and (run / 'config.json').exists() and (run / 'norm_stats.json').exists():
        print(f'weights already present: {weights}', flush=True)
        return
    official = _official()
    cfg = _config()
    if list(cfg.wm.encoding.keys()) != ['action']:
        raise RuntimeError(f'expected action-only encoding, got {list(cfg.wm.encoding.keys())}')
    import stable_worldmodel as swm
    from stable_worldmodel.data import column_normalizer as get_column_normalizer

    exp_cfg = load_config(str(DEFAULT_CONFIG))
    lance = env_path(exp_cfg, 'pusht')
    dataset = swm.data.load_dataset(
        lance,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        keys_to_load=['pixels', 'action'],
        keys_to_cache=['action'],
    )
    normalizer = get_column_normalizer(dataset, 'action', 'action')
    dataset.transform = spt.data.transforms.Compose(
        official.get_img_preprocessor('pixels', 'pixels', cfg.image_size),
        normalizer,
    )
    action = np.asarray(dataset.get_col_data('action'), dtype=np.float64)
    action = action[np.isfinite(action).all(axis=1)]
    run.mkdir(parents=True, exist_ok=True)
    (run / 'norm_stats.json').write_text(json.dumps({
        'action': {
            'mean': action.mean(axis=0).astype(float).tolist(),
            'std': action.std(axis=0).astype(float).tolist(),
            'eps': 1e-8,
        }
    }))

    with open_dict(cfg):
        cfg.extra_dims = {}
        for key in cfg.wm.encoding:
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = dim * cfg.frameskip if key == 'action' else dim
        encoder = hydra_encoder(cfg)
        embed_dim = encoder.config.hidden_size + sum(int(v) for v in cfg.wm.encoding.values())
        num_patches = (int(cfg.image_size) // int(cfg.patch_size)) ** 2
        cfg.model.predictor.dim = embed_dim
        cfg.model.predictor.num_patches = num_patches
        cfg.model.extra_encoders = {
            '_target_': 'torch.nn.ModuleDict',
            'modules': {
                key: {
                    '_target_': 'stable_worldmodel.wm.prejepa.module.Embedder',
                    'in_chans': int(cfg.extra_dims[key]),
                    'emb_dim': int(cfg.wm.encoding[key]),
                }
                for key in cfg.wm.encoding
            },
        }

    from hydra.utils import instantiate

    world_model = instantiate(cfg.model, encoder=encoder)
    module = spt.Module(
        model=world_model,
        forward=partial(official.dinowm_forward, cfg=cfg),
        optim={'model_opt': {'modules': 'model', 'optimizer': OmegaConf.to_container(cfg.optimizer, resolve=True)}},
    )
    indices = _train_indices(dataset, exp_cfg)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    visible = 1 if limit else torch.cuda.device_count()
    global_batch = int(cfg.batch_size)
    if global_batch % visible:
        raise RuntimeError(f'batch {global_batch} is not divisible by {visible} GPUs')
    local_batch = global_batch // visible
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=local_batch,
        num_workers=2 if limit else int(cfg.num_workers),
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=generator,
    )
    trainer_kwargs = dict(OmegaConf.to_container(cfg.trainer, resolve=True))
    # Current Lightning rejects Trainer(gradient_clip_val=...) under spt's manual
    # optimization. The clip still runs inside spt from this per-optimizer value.
    clip = float(trainer_kwargs.pop('gradient_clip_val', 1.0))
    module._optimizer_gradient_clip_val['model_opt'] = clip
    module._optimizer_gradient_clip_algorithm['model_opt'] = 'norm'
    if limit:
        trainer_kwargs['max_epochs'] = 1
        trainer_kwargs['devices'] = 1
        trainer_kwargs['strategy'] = 'auto'
        trainer_kwargs['limit_train_batches'] = limit
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[
            _Bar(log),
            _Save(run, cfg.model),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(run), save_last=True, save_top_k=0, every_n_epochs=1
            ),
        ],
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        default_root_dir=str(run),
    )
    (run / 'recipe.json').write_text(json.dumps({
        'source': 'scripts/train/config/prejepa.yaml',
        'encoding': {'action': 10},
        'omitted': 'proprio',
        'dataset': lance,
        'split': 'experiment contiguous train episodes; val and test held out',
        'train_clips': len(indices),
        'epochs': trainer_kwargs['max_epochs'],
        'batch_size': global_batch,
        'local_batch_size': local_batch,
        'devices': visible,
        'optimizer': OmegaConf.to_container(cfg.optimizer, resolve=True),
        'predictor_dim': int(embed_dim),
        'num_patches': int(num_patches),
    }, indent=2))
    last = run / 'last.ckpt'
    trainer.fit(module, train_dataloaders=loader, ckpt_path=str(last) if last.exists() else None)


def hydra_encoder(cfg):
    from hydra.utils import instantiate

    encoder = instantiate(cfg.model.encoder)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def evaluate() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(WEIGHTS)
    from rdwm.exp1.exposure_eval import _notify
    from rdwm.exp1.prejepa_diag import diagnose

    cfg = load_config(str(DEFAULT_CONFIG))
    out = Path('/workspace/rdwm_runs/exp1/logs/dinowm/parity_rgb.json')
    report = diagnose(
        cfg, out, torch.device('cuda'), workers=16, cost='pixels', checkpoint=str(WEIGHTS)
    )
    rank = report['rank_summary']
    plan = report['plan_summary']
    text = (
        f"dinowm rgb | spearman {rank['mean_spearman']:.3f} | "
        f"closer rank {rank['mean_closer_rank_percentile']:.1f} | "
        f"plan {plan['success_count']}/10 | closer {plan['closer_count']}/10 | "
        f"dist {plan['goal_pos_initial']:.0f}->{plan['goal_pos_final']:.0f}"
    )
    print(text, flush=True)
    try:
        _notify(text)
    except Exception as exc:
        print(f'slack failed: {type(exc).__name__}', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode == 'train':
        train()
    elif mode == 'eval':
        evaluate()
    else:
        raise SystemExit(f'unknown mode {mode}')
