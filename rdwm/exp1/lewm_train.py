"""Train LeWM from scratch for a fixed update budget.

The Stable-WM LeWM modules stay unchanged. This wrapper only supplies the
PushT clips, the one-step LeWM loss, and a step-wise warmup plus cosine
whose length is the requested budget. The episode split is the Experiment 1
PushT split, so validation episodes stay out of training. Each budget draws
its own schedule from the same seed; a shorter budget is the prefix of a
longer one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rdwm.exp1.config import env_path
from rdwm.exp1.data import EnvData, decode_jpegs, preprocess
from rdwm.exp1.init import lr_factor, save_checkpoint
from rdwm.exp1.progress import Progress
from rdwm.exp1.util import write_json
from stable_worldmodel.wm.loss import SIGReg

BATCH = 128
LR = 5e-5
WEIGHT_DECAY = 1e-3
WARMUP_FRAC = 0.01
GRAD_CLIP = 1.0
SIGREG_WEIGHT = 0.09
HISTORY = 3
NUM_PREDS = 1
NUM_OBS = HISTORY + NUM_PREDS
STRIDE = 5
SEED_SALT = 7919


def build_lewm() -> nn.Module:
    from hydra.utils import instantiate

    config = {
        '_target_': 'stable_worldmodel.wm.lewm.LeWM',
        'encoder': {
            '_target_': 'stable_pretraining.backbone.utils.vit_hf',
            'size': 'tiny',
            'patch_size': 14,
            'image_size': 224,
            'pretrained': False,
            'use_mask_token': False,
        },
        'predictor': {
            '_target_': 'stable_worldmodel.wm.lewm.module.Predictor',
            'num_frames': HISTORY,
            'input_dim': 192,
            'hidden_dim': 192,
            'output_dim': 192,
            'depth': 6,
            'heads': 16,
            'mlp_dim': 2048,
            'dim_head': 64,
            'dropout': 0.1,
            'emb_dropout': 0.0,
        },
        'action_encoder': {
            '_target_': 'stable_worldmodel.wm.lewm.module.Embedder',
            'input_dim': STRIDE * 2,
            'emb_dim': 192,
        },
        'projector': {
            '_target_': 'stable_worldmodel.wm.lewm.module.MLP',
            'input_dim': 192,
            'output_dim': 192,
            'hidden_dim': 2048,
            'norm_fn': {'_target_': 'torch.nn.BatchNorm1d', '_partial_': True},
        },
        'pred_proj': {
            '_target_': 'stable_worldmodel.wm.lewm.module.MLP',
            'input_dim': 192,
            'output_dim': 192,
            'hidden_dim': 2048,
            'norm_fn': {'_target_': 'torch.nn.BatchNorm1d', '_partial_': True},
        },
    }
    return instantiate(config)


def lewm_loss(model, pixels, actions, sigreg) -> dict[str, torch.Tensor]:
    """One-step CLS MSE plus SIGReg, matching ``lejepa_forward``."""
    info = model.encode({'pixels': pixels, 'action': actions})
    emb = info['emb']
    ctx = HISTORY
    pred = model.predict(emb[:, :ctx], info['act_emb'][:, :ctx])
    target = emb[:, NUM_PREDS:]
    pred_loss = (pred.float() - target.float()).pow(2).mean()
    sig = sigreg(emb.float().transpose(0, 1))
    return {
        'loss': pred_loss + SIGREG_WEIGHT * sig,
        'pred_loss': pred_loss,
        'sigreg': sig,
    }


def _schedule(seed: int, updates: int, n_clips: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, SEED_SALT, BATCH]))
    clip = np.empty((updates, BATCH), dtype=np.int64)
    for i in range(updates):
        clip[i] = rng.choice(n_clips, BATCH, replace=False)
    return {'clip': clip}


class ClipBatches(torch.utils.data.Dataset):
    def __init__(self, lance_path: str, index_dir: str, actions_path: str, schedule_dir: str, mean, std):
        self.lance_path = lance_path
        self.index_dir = index_dir
        self.actions_path = actions_path
        self.schedule_dir = schedule_dir
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.total = int(len(np.load(Path(schedule_dir) / 'clip.npy', mmap_mode='r')))
        self._state = None

    def __len__(self) -> int:
        return self.total

    def _open(self):
        import lance

        self._state = (
            np.load(Path(self.schedule_dir) / 'clip.npy', mmap_mode='r'),
            np.load(Path(self.index_dir) / 'episode.npy'),
            np.load(Path(self.index_dir) / 'start.npy'),
            np.load(Path(self.index_dir) / 'offsets.npy'),
            np.load(self.actions_path, mmap_mode='r'),
            lance.dataset(self.lance_path),
        )

    def __getitem__(self, i: int) -> dict:
        if self._state is None:
            self._open()
        clip, episode, start, offsets, actions, table = self._state
        idx = np.asarray(clip[int(i)], dtype=np.int64)
        eps = episode[idx]
        starts = start[idx]
        base = offsets[eps] + starts
        rows = base[:, None] + STRIDE * np.arange(NUM_OBS)[None]
        frames = decode_jpegs(
            table.take(rows.reshape(-1).tolist(), columns=['pixels']).column('pixels').to_pylist()
        )
        pixels = frames.reshape(len(idx), NUM_OBS, *frames.shape[1:])
        act_rows = base[:, None] + np.arange(NUM_OBS * STRIDE)[None]
        acts = np.asarray(actions[act_rows.reshape(-1)], dtype=np.float32)
        acts = acts.reshape(len(idx), NUM_OBS * STRIDE, -1)
        acts = (acts - self.mean) / self.std
        acts = acts.reshape(len(idx), NUM_OBS, STRIDE * acts.shape[-1])
        return {
            'update': int(i),
            'pixels': pixels,
            'actions': torch.from_numpy(np.ascontiguousarray(acts)),
        }


def _save_index(folder: Path, episode: np.ndarray, start: np.ndarray, offsets: np.ndarray):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / 'episode.npy', episode)
    np.save(folder / 'start.npy', start)
    np.save(folder / 'offsets.npy', offsets)


def train(cfg: dict, *, seed: int, updates: int, stage: str, num_workers: int = 8) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError('training needs a GPU; set CUDA_VISIBLE_DEVICES')
    # Ranking/plan diagnostics call set_grad_enabled(False) for the process.
    # A later budget in the same process must train with grad on.
    torch.set_grad_enabled(True)
    device = torch.device('cuda')
    out = Path(cfg['paths']['run_root']) / stage / 'lewm' / f'seed{seed}' / 'pusht'
    done_path = out / 'done.json'
    if done_path.exists():
        finished = int(json.loads(done_path.read_text()).get('updates', 0))
        if finished >= updates:
            print(f'[skip] {out} already done ({finished} updates)', flush=True)
            return out
    out.mkdir(parents=True, exist_ok=True)

    data = EnvData(cfg, 'pusht', env_path(cfg, 'pusht'))
    index_dir = out / 'clip_index'
    _save_index(index_dir, data.clip_ep, data.clip_start, data.index.offsets)
    sched_dir = out / 'schedule'
    sched_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = sched_dir / 'clip.npy'
    if not schedule_path.exists() or np.load(schedule_path, mmap_mode='r').shape[0] != updates:
        np.save(schedule_path, _schedule(seed, updates, len(data.clip_ep))['clip'])

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = build_lewm().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = max(1, int(round(WARMUP_FRAC * updates)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: lr_factor(step, warmup, updates)
    )
    sigreg = SIGReg(17, 1024).to(device)
    dataset = ClipBatches(
        env_path(cfg, 'pusht'),
        str(index_dir),
        str(data.index.folder / 'actions.npy'),
        str(sched_dir),
        data.action_mean,
        data.action_std,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=cfg['data']['prefetch_factor'] if num_workers else None,
        multiprocessing_context='spawn' if num_workers else None,
        pin_memory=True,
        persistent_workers=False,
    )
    write_json(out / 'meta.json', {
        'model': 'lewm',
        'seed': seed,
        'updates': updates,
        'batch_size': BATCH,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'warmup_steps': warmup,
        'samples': updates * BATCH,
        'train_episodes': int(len(data.splits['train'])),
        'train_clips': int(len(data.clip_ep)),
        'action_mean': data.action_mean.tolist(),
        'action_std': data.action_std.tolist(),
        'split': 'experiment-1 contiguous episode split, train clips only',
    })
    (out / 'config.json').write_text(json.dumps({
        '_target_': 'stable_worldmodel.wm.lewm.LeWM',
        'encoder': {
            '_target_': 'stable_pretraining.backbone.utils.vit_hf',
            'size': 'tiny', 'patch_size': 14, 'image_size': 224,
            'pretrained': False, 'use_mask_token': False,
        },
        'predictor': {
            '_target_': 'stable_worldmodel.wm.lewm.module.Predictor',
            'num_frames': HISTORY, 'input_dim': 192, 'hidden_dim': 192,
            'output_dim': 192, 'depth': 6, 'heads': 16, 'mlp_dim': 2048,
            'dim_head': 64, 'dropout': 0.1, 'emb_dropout': 0.0,
        },
        'action_encoder': {
            '_target_': 'stable_worldmodel.wm.lewm.module.Embedder',
            'input_dim': 10, 'emb_dim': 192,
        },
        'projector': {
            '_target_': 'stable_worldmodel.wm.lewm.module.MLP',
            'input_dim': 192, 'output_dim': 192, 'hidden_dim': 2048,
            'norm_fn': {'_target_': 'torch.nn.BatchNorm1d', '_partial_': True},
        },
        'pred_proj': {
            '_target_': 'stable_worldmodel.wm.lewm.module.MLP',
            'input_dim': 192, 'output_dim': 192, 'hidden_dim': 2048,
            'norm_fn': {'_target_': 'torch.nn.BatchNorm1d', '_partial_': True},
        },
    }, indent=2))

    gpu = os_gpu()
    bar = Progress(updates, 0, f'PushT LeWM | GPU {gpu}', out / 'progress.log')
    log_path = out / 'metrics.jsonl'
    t_last = time.time()
    log_every = int(cfg['train']['log_every'])
    model.train()
    for batch in loader:
        pixels = preprocess(batch['pixels'].to(device, non_blocking=True), cfg)
        actions = batch['actions'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            terms = lewm_loss(model, pixels, actions, sigreg)
        loss = terms['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {int(batch['update'])}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        if not torch.isfinite(torch.tensor(total_norm)):
            raise FloatingPointError(f"non-finite grad at update {int(batch['update'])}")
        opt.step()
        sched.step()
        done = int(batch['update']) + 1
        if done % log_every == 0 or done == updates:
            now = time.time()
            record = {
                'update': done,
                'lr': sched.get_last_lr()[0],
                'loss': float(terms['loss'].detach()),
                'pred_loss': float(terms['pred_loss'].detach()),
                'sigreg': float(terms['sigreg'].detach()),
                'grad_total': total_norm,
                'sec_per_update': (now - t_last) / (log_every if done > log_every else done),
            }
            with open(log_path, 'a') as handle:
                handle.write(json.dumps(record) + '\n')
            t_last = now
        bar.desc = f'PushT LeWM | GPU {gpu}'
        bar.set_postfix(
            f"loss={float(terms['loss'].detach()):.4f} | "
            f"pred={float(terms['pred_loss'].detach()):.4f} | "
            f"sigreg={float(terms['sigreg'].detach()):.4f}"
        )
        bar.update(done)
        if done == updates:
            ckpt_meta = {'total_updates': updates, 'seed': seed, 'model': 'lewm'}
            save_checkpoint(out / 'ckpt_last.pt', model, opt, sched, done, ckpt_meta)
            torch.save(model.state_dict(), out / 'weights.pt')
    bar.close()
    write_json(out / 'done.json', {
        'updates': updates,
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    return out


def os_gpu() -> str:
    import os
    return os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
