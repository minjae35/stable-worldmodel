"""One Experiment 1 training job (any variant, one seed, one env set).

The GPU is whatever ``CUDA_VISIBLE_DEVICES`` exposes as ``cuda:0``; the
launcher sets it per job.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rdwm.exp1.config import (
    check_envs,
    env_order,
    env_path,
    env_tag,
    run_dir,
    shared_dir,
)
from rdwm.exp1.data import (
    EnvData,
    ScheduleBatches,
    build_schedule,
    load_schedule,
    preprocess,
    resolve_schedule,
    restrict_schedule,
    save_schedule,
    schedule_digest,
)
from rdwm.exp1.init import (
    build_model,
    make_optimizer,
    save_checkpoint,
    shared_init,
    shared_subset_hash,
)
from rdwm.exp1.losses import SIGReg, one_step_loss, recursive_loss
from rdwm.exp1.progress import Progress
from rdwm.exp1.util import file_lock, git_state, write_json


def prepare_schedule(
    cfg: dict,
    seed: int,
    schedule_envs: list[str],
    schedule_updates: int,
    data: dict[str, EnvData],
    restrict_to: str | None = None,
) -> tuple[Path, str]:
    """Build (once per seed) the generalist schedule over ``schedule_envs``.

    Specialists pass ``restrict_to`` and get that env's subsequence. B, C,
    and D of one seed read the same folder, so they see the same env order,
    clip IDs, and history lengths.
    """
    base = shared_dir(cfg, seed) / 'schedules'
    tag = f'{env_tag(cfg, schedule_envs)}_u{schedule_updates}'
    folder = base / tag
    with file_lock(base / f'{tag}.lock'):
        if not (folder / 'meta.json').exists():
            raw = build_schedule(
                seed,
                schedule_envs,
                {e: len(data[e].clip_ep) for e in schedule_envs},
                schedule_updates,
                cfg['train']['batch_size'],
                cfg['data']['history_probs'],
            )
            resolved = resolve_schedule(raw, schedule_envs, data)
            save_schedule(
                folder,
                resolved,
                {'seed': seed, 'envs': schedule_envs, 'updates': schedule_updates},
            )
        if restrict_to is None:
            meta = json.loads((folder / 'meta.json').read_text())
            return folder, meta['sha256']
        sub = folder.parent / f'{tag}__only_{restrict_to}'
        if not (sub / 'meta.json').exists():
            full = load_schedule(folder, mmap=False)
            part = restrict_schedule(full, schedule_envs, restrict_to)
            save_schedule(
                sub,
                part,
                {
                    'seed': seed,
                    'envs': [restrict_to],
                    'parent': tag,
                    'updates': int(len(part['env'])),
                },
            )
        meta = json.loads((sub / 'meta.json').read_text())
        return sub, meta['sha256']


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    per_env = {}
    for env in model.envs:
        other_envs = [e for e in model.envs if e != env]
        inactive = 0
        for name, param in model.named_parameters():
            parts = name.split('.')
            if any(o in parts for o in other_envs):
                inactive += param.numel()
        per_env[env] = total - inactive
    return {'total': total, 'active_per_env': per_env}


def _grad_norms(model) -> dict[str, float]:
    groups = defaultdict(float)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name.startswith('dynamics.adapters.'):
            key = 'adapters'
        else:
            key = name.split('.')[0]
        groups[key] += float(param.grad.detach().float().pow(2).sum())
    return {k: math.sqrt(v) for k, v in groups.items()}


def train(
    cfg: dict,
    *,
    variant: str,
    seed: int,
    envs: list[str],
    updates: int,
    stage: str,
    schedule_envs: list[str] | None = None,
    num_workers: int | None = None,
    allow_schedule_rebind: bool = False,
    encoder: str = 'vit',
    objective: str = 'recursive',
    model_overrides: dict | None = None,
    train_overrides: dict | None = None,
    allow_dynamics_reinit: bool = False,
    progress_label: str | None = None,
) -> Path:
    envs = check_envs(cfg, variant, envs)
    out = run_dir(cfg, stage, variant, seed, envs)
    done_path = out / 'done.json'
    if done_path.exists():
        finished = int(json.loads(done_path.read_text()).get('updates', 0))
        if finished >= updates:
            print(f'[skip] {out} already done ({finished} updates)')
            return out
        # A shorter run finished here. Keep its checkpoints and continue.
        done_path.replace(out / f'done_at_{finished}.json')
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('training needs a GPU; set CUDA_VISIBLE_DEVICES')
    # Ranking/plan diagnostics call set_grad_enabled(False) for the process.
    # A later budget in the same process must train with grad on.
    torch.set_grad_enabled(True)
    device = torch.device('cuda')

    if variant == 'A':
        schedule_envs = check_envs(cfg, 'B', schedule_envs or env_order(cfg))
        if envs[0] not in schedule_envs:
            raise ValueError(f'{envs[0]} not in schedule envs {schedule_envs}')
        schedule_updates = updates * len(schedule_envs)
        restrict_to = envs[0]
    else:
        if schedule_envs not in (None, envs):
            raise ValueError('generalists use their own env set as schedule envs')
        schedule_envs = envs
        schedule_updates = updates
        restrict_to = None

    data = {e: EnvData(cfg, e, env_path(cfg, e)) for e in schedule_envs}
    sched_dir, sched_hash = prepare_schedule(
        cfg, seed, schedule_envs, schedule_updates, data, restrict_to
    )
    schedule = load_schedule(sched_dir)
    if len(schedule['env']) != updates:
        raise RuntimeError(
            f"schedule has {len(schedule['env'])} updates, job wants {updates}"
        )

    with file_lock(shared_dir(cfg, seed) / 'shared_init.lock'):
        # Shared init is always the canonical ViT state. Ablation overrides
        # are applied after that, so a wider predictor cannot rewrite the file.
        shared_state, init_hash = shared_init(cfg, seed)
    if model_overrides or train_overrides or objective != 'recursive':
        cfg = copy.deepcopy(cfg)
        if model_overrides:
            cfg['model'].update(model_overrides)
        if train_overrides:
            cfg['train'].update(train_overrides)
        cfg['train']['objective'] = objective
    if encoder == 'vit':
        model, init_report = build_model(cfg, variant, envs, seed, shared_state)
    elif encoder == 'frozen_dinov2_small':
        if variant != 'A' or envs != ['pusht']:
            raise ValueError(
                'frozen DINOv2 is a PushT specialist prototype, not an Experiment 1 variant'
            )
        if allow_dynamics_reinit:
            from rdwm.exp1.dino_encoder import build_dino_ablation

            model, init_report = build_dino_ablation(cfg, envs, seed, shared_state)
        else:
            from rdwm.exp1.dino_encoder import build_dino_specialist

            model, init_report = build_dino_specialist(cfg, envs, seed, shared_state)
    else:
        raise ValueError(f'unknown encoder {encoder}')
    shared_hash = shared_subset_hash(model)
    for env in envs:
        model.action_norm[env].load(data[env].action_mean, data[env].action_std)
    model.to(device)
    opt, lr_sched = make_optimizer(model, cfg, updates)

    first = 0
    last_ckpt = out / 'ckpt_last.pt'
    if last_ckpt.exists():
        state = torch.load(last_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        first = int(state['update'])
        saved_total = state['meta'].get('total_updates', first)
        if saved_total == updates:
            lr_sched.load_state_dict(state['sched'])
        elif allow_schedule_rebind:
            # Diagnostic only. The checkpoint decayed on a shorter budget, so
            # this is not one warmup+cosine over `updates`. Do not use the
            # result as an official Experiment 1 run.
            from rdwm.exp1.init import lr_factor

            factor = lr_factor(first, max(1, int(round(cfg['train']['warmup_frac'] * updates))), updates)
            for group, base in zip(opt.param_groups, lr_sched.base_lrs):
                group['lr'] = base * factor
            lr_sched.last_epoch = first
            print(
                f'[resume] DIAGNOSTIC lr rebind onto the {updates}-update '
                f'schedule at update {first} (factor {factor:.4f})'
            )
        else:
            raise RuntimeError(
                f'{out} checkpoint was trained for {saved_total} updates, '
                f'not {updates}. Official runs need one warmup+cosine over '
                f'the full budget from step 0. Refusing to rebind the schedule.'
            )
        torch.set_rng_state(state['meta']['rng_cpu'])
        torch.cuda.set_rng_state(state['meta']['rng_cuda'])
        print(f'[resume] {out} from update {first}')
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    meta = {
        'variant': variant,
        'variant_name': cfg['variants'][variant]['name'],
        'encoder': encoder,
        'prototype': encoder != 'vit',
        'seed': seed,
        'stage': stage,
        'envs': envs,
        'updates': updates,
        'schedule_dir': str(sched_dir),
        'schedule_sha256': sched_hash,
        'shared_init_sha256': init_hash,
        'shared_params_sha256_at_init': shared_hash,
        'init_report': init_report,
        'params': {
            **count_params(model),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        },
        'data': {e: data[e].summary() for e in envs},
        'git': git_state(),
        'torch': torch.__version__,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_name': torch.cuda.get_device_name(0),
        'config_revision': cfg['revision'],
        'config': cfg,
    }
    write_json(out / 'meta.json', meta)

    dcfg = cfg['data']
    dataset = ScheduleBatches(
        envs=envs,
        lance_paths={e: env_path(cfg, e) for e in envs},
        index_dirs={e: str(data[e].index.folder) for e in envs},
        schedule_dir=str(sched_dir),
        first=first,
        num_obs=dcfg['num_obs'],
        stride=dcfg['obs_stride'],
        action_block=cfg['model']['action_block'],
    )
    workers = dcfg['num_workers'] if num_workers is None else num_workers
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=dcfg['prefetch_factor'] if workers else None,
        multiprocessing_context='spawn' if workers else None,
        pin_memory=True,
        persistent_workers=False,
    )

    tcfg = cfg['train']
    sigreg = SIGReg(tcfg['sigreg']['knots'], tcfg['sigreg']['num_proj']).to(device)
    log_path = out / 'metrics.jsonl'
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    bar = Progress(updates, first, '', out / 'progress.log')
    action_effect = float('nan')
    window: dict[str, list] = defaultdict(list)
    t_last = time.time()
    t_wait = 0.0
    tick = time.time()
    model.train()
    if progress_label:
        bar.desc = f'{_env_label(envs[0])} {progress_label} | GPU {gpu}'
    elif encoder == 'vit':
        bar.desc = (
            f'{_env_label(envs[0])} {variant} '
            f'{"specialist" if variant == "A" else "generalist"} | GPU {gpu}'
        )
    else:
        bar.desc = f'{_env_label(envs[0])} DINOv2 prototype | GPU {gpu}'
    bar.set_postfix('loading the first batch')
    bar.update(first)
    for batch in loader:
        t_wait += time.time() - tick
        u = int(batch['update'])
        env = batch['env']
        pixels = preprocess(batch['pixels'].to(device, non_blocking=True), cfg)
        actions = batch['actions'].to(device, non_blocking=True)
        loss_fn = one_step_loss if cfg['train'].get('objective') == 'one_step' else recursive_loss
        terms = loss_fn(
            model, pixels, actions, env, int(batch['history']), sigreg, cfg
        )
        loss = terms['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at update {u}: {terms}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norms = _grad_norms(model)
        total_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
        )
        if not math.isfinite(total_norm):
            raise FloatingPointError(f'non-finite grad norm at update {u}')
        opt.step()
        lr_sched.step()

        row = {k: float(v.detach()) for k, v in terms.items()}
        row.update({f'grad_{k}': v for k, v in norms.items()})
        row['grad_total'] = total_norm
        row['history'] = int(batch['history'])
        window[env].append(row)
        done = u + 1
        if done % tcfg['log_every'] == 0 or done == updates:
            now = time.time()
            action_effect = _action_effect(model, pixels, actions, env, cfg)
            record = {
                'update': done,
                'lr': lr_sched.get_last_lr()[0],
                'sec_per_update': (now - t_last) / tcfg['log_every'],
                'data_wait_frac': t_wait / max(1e-9, now - t_last),
                'per_env': {
                    e: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                    for e, rows in window.items()
                },
                'action_effect': action_effect,
            }
            with open(log_path, 'a') as handle:
                handle.write(json.dumps(record) + '\n')
            window.clear()
            t_last, t_wait = now, 0.0
        label = _env_label(env)
        kind = 'specialist' if variant == 'A' else 'generalist'
        if progress_label:
            bar.desc = f'{label} {progress_label} | GPU {gpu}'
        elif encoder == 'vit':
            bar.desc = f'{label} {variant} {kind} | GPU {gpu}'
        else:
            bar.desc = f'{label} DINOv2 prototype | GPU {gpu}'
        bar.set_postfix(
            f"loss={row['loss']:.4f} | pred={row['mse_t1']:.4f} | "
            f"sigreg={row['sigreg']:.4f} | action_effect={action_effect:.3e}"
        )
        bar.update(done)
        if (
            done % tcfg['ckpt_every'] == 0
            or done == updates
            or done in tcfg.get('ckpt_steps', ())
        ):
            ckpt_meta = {
                'rng_cpu': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state(),
                'variant': variant,
                'envs': envs,
                'seed': seed,
                'total_updates': updates,
                'encoder': encoder,
            }
            save_checkpoint(last_ckpt, model, opt, lr_sched, done, ckpt_meta)
            save_checkpoint(
                out / f'ckpt_{done:06d}.pt', model, opt, lr_sched, done, ckpt_meta
            )
        tick = time.time()

    bar.close()
    write_json(
        out / 'done.json',
        {'updates': updates, 'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')},
    )
    return out


def _env_label(env: str) -> str:
    return {'pusht': 'PushT'}.get(env, env)


@torch.no_grad()
def _action_effect(model, pixels, actions, env: str, cfg: dict) -> float:
    """t+2 latent MSE between the true action and a batch-shuffled action."""
    was_training = model.training
    model.eval()
    px, ac = pixels[:8], actions[:8]
    shuffled = torch.roll(ac, 1, 0)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=pixels.is_cuda):
        z = model.encode(px)
        true = model.rollout(z[:, :3], ac[:, :2], ac[:, 2:4], env)
        other = model.rollout(z[:, :3], shuffled[:, :2], shuffled[:, 2:4], env)
    diff = (true[:, 1] - other[:, 1]).float().pow(2).mean()
    model.train(was_training)
    return float(diff)
