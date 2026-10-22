"""Train one G5 arm as an exact 500-update continuation of the H1 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from stable_worldmodel.wm.spatial.g2 import (  # noqa: E402
    ENV_NAMES, grad_norm, load_resume_checkpoint, save_checkpoint_no_overwrite,
    schedule_sha256, tensor_state_sha256,
)
from stable_worldmodel.wm.spatial.g2_extension import (  # noqa: E402
    capture_rng_state, restore_rng_state,
)
from stable_worldmodel.wm.spatial.g5 import (  # noqa: E402
    FINAL_STEP, NEW_UPDATES, SOURCE_STEP, run_config, sha256_file, validate_h1_source,
)
from stable_worldmodel.wm.spatial.g5_objective import (  # noqa: E402
    G5_ARMS, objective_components, patch_mse,
)
from stable_worldmodel.wm.spatial.model import SpatialDynamics  # noqa: E402


def _move_optimizer(optimizer):
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(parameter.device)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=G5_ARMS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-update', type=int, default=0)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if int(plan['new_updates_per_arm']) != NEW_UPDATES:
        raise ValueError('G5 update budget differs from preregistered 500')
    if plan['total_optimizer_updates'] > 2000:
        raise ValueError('G5 total optimizer budget exceeds 2000')
    config = run_config(plan, args.arm, output)
    if args.resume is None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    else:
        if not output.is_dir() or args.resume.resolve().parent != output:
            raise ValueError('resume checkpoint must be inside this arm output')
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('existing config differs from requested resume config')
    (output / 'STATUS').write_text('RUNNING\n')

    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = SpatialDynamics().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    source_path = Path(plan['source']['checkpoint'])
    if sha256_file(source_path) != plan['source']['checkpoint_sha256']:
        raise ValueError('source checkpoint hash mismatch')
    source_config = json.loads(Path(plan['source']['config']).read_text())
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    validate_h1_source(source, source_config)

    start_update = 0
    if args.resume is None:
        model.load_state_dict(source['model'], strict=True)
        optimizer.load_state_dict(source['optimizer'])
        _move_optimizer(optimizer)
        restore_rng_state(source['extra']['rng_state'])
    else:
        resumed = load_resume_checkpoint(args.resume, model, optimizer, config)
        if 'rng_state' not in resumed.get('extra', {}):
            raise ValueError('G5 resume checkpoint lacks RNG state')
        start_update = int(resumed['step']) - SOURCE_STEP
        if start_update < 1 or start_update >= NEW_UPDATES:
            raise ValueError('resume update is outside (0, 500)')
        restore_rng_state(resumed['extra']['rng_state'])
    source_model_hash = tensor_state_sha256(source['model'])
    if start_update == 0 and tensor_state_sha256(model.state_dict()) != source_model_hash:
        raise RuntimeError('initial model differs from H1 source')

    cache_root = plan_path.parent / 'shared_train_cache'
    if (cache_root / 'STATUS').read_text().strip() != 'READY':
        raise RuntimeError('shared train cache is not READY')
    schedule_path = Path(plan['schedule_file'])
    if sha256_file(schedule_path) != plan['schedule_file_sha256']:
        raise ValueError('schedule artifact hash mismatch')
    schedules_payload = torch.load(schedule_path, map_location='cpu', weights_only=True)
    schedules = schedules_payload['indices']
    derangements = schedules_payload['derangements']
    bundles = {}
    for env in ENV_NAMES:
        path = cache_root / f'{env}_train.pt'
        if sha256_file(path) != plan['cache_file_sha256'][env]:
            raise ValueError(f'{env} train cache hash mismatch')
        bundles[env] = torch.load(path, map_location='cpu', weights_only=True)
        if schedule_sha256(schedules[env]) != plan['schedule_sha256'][env]:
            raise ValueError(f'{env} schedule hash mismatch')
        if schedule_sha256(derangements[env]) != plan['derangement_sha256'][env]:
            raise ValueError(f'{env} derangement hash mismatch')

    calibration = plan['calibration'][args.arm]
    fields = [
        'new_update', 'cumulative_step', 'objective', 'raw_objective',
        'uniform_mse', 'motion_balanced_mse', 'ranking_loss',
        'ranking_active_fraction', 'correct_mse', 'shuffled_mse', 'action_gap_mse',
        'pusht_mse', 'tworoom_mse', 'cube_mse', 'grad_norm', 'lr',
        'step_time_s', 'samples_per_s', 'peak_vram_gb', 'weight_min', 'weight_max',
    ]
    metrics_path = output / 'metrics.csv'
    mode = 'a' if args.resume else 'x'
    with metrics_path.open(mode, newline='', buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if args.resume is None:
            writer.writeheader()
        params = [p for p in model.parameters() if p.requires_grad]
        model.train()
        torch.cuda.reset_peak_memory_stats()
        stopped = False
        bar = trange(
            start_update, NEW_UPDATES, initial=start_update, total=NEW_UPDATES,
            dynamic_ncols=True, desc=args.arm,
        )
        for update_index in bar:
            torch.cuda.synchronize(); began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            env_parts, shuffled_mses = {}, []
            for env_index, env in enumerate(ENV_NAMES):
                indices = torch.as_tensor(schedules[env][update_index], dtype=torch.long)
                context = bundles[env]['context'].index_select(0, indices).to(device).float()
                target = bundles[env]['target'].index_select(0, indices).to(device).float()[:, None]
                actions = bundles[env]['actions'].index_select(0, indices).to(device).float()
                env_id = torch.full((len(indices),), env_index, device=device, dtype=torch.long)
                prediction = model(context, actions, env_id)[:, -1:]
                permutation = torch.as_tensor(
                    derangements[env][update_index], device=device, dtype=torch.long
                )
                shuffled_actions = actions.index_select(0, permutation)
                if args.arm in ('C_action_sensitive', 'D_combined'):
                    shuffled = model(context, shuffled_actions, env_id)[:, -1:]
                else:
                    with torch.no_grad():
                        shuffled = model(context, shuffled_actions, env_id)[:, -1:]
                env_parts[env] = objective_components(
                    args.arm, prediction, target, context[:, -1:], shuffled,
                    motion_blend=plan['objective']['motion_blend'],
                    relative_cap=plan['objective']['relative_cap'],
                    margin_fraction=plan['objective']['margin_fraction'],
                    ranking_lambda=calibration['ranking_lambda'],
                    objective_scale=calibration['objective_scale'],
                )
                shuffled_mses.append(patch_mse(shuffled, target).mean().detach())
            objective = torch.stack([env_parts[e]['objective'] for e in ENV_NAMES]).mean()
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(f'non-finite objective at update {update_index + 1}')
            objective.backward()
            norm = grad_norm(params)
            if not np.isfinite(norm):
                raise FloatingPointError(f'non-finite gradient at update {update_index + 1}')
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            torch.cuda.synchronize(); elapsed = time.perf_counter() - began
            done = update_index + 1
            mean = lambda key: float(torch.stack([env_parts[e][key].detach() for e in ENV_NAMES]).mean())
            correct_mse = mean('uniform_mse')
            shuffled_mse = float(torch.stack(shuffled_mses).mean())
            row = {
                'new_update': done, 'cumulative_step': SOURCE_STEP + done,
                'objective': float(objective.detach()), 'raw_objective': mean('raw_objective'),
                'uniform_mse': correct_mse, 'motion_balanced_mse': mean('motion_balanced_mse'),
                'ranking_loss': mean('ranking_loss'),
                'ranking_active_fraction': mean('ranking_active_fraction'),
                'correct_mse': correct_mse, 'shuffled_mse': shuffled_mse,
                'action_gap_mse': shuffled_mse - correct_mse,
                **{f'{e}_mse': float(env_parts[e]['uniform_mse'].detach()) for e in ENV_NAMES},
                'grad_norm': norm, 'lr': optimizer.param_groups[0]['lr'],
                'step_time_s': elapsed,
                'samples_per_s': len(ENV_NAMES) * len(indices) / elapsed,
                'peak_vram_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
                'weight_min': min(float(env_parts[e]['weight_min']) for e in ENV_NAMES),
                'weight_max': max(float(env_parts[e]['weight_max']) for e in ENV_NAMES),
            }
            writer.writerow(row); handle.flush()
            bar.set_postfix(
                mse=f'{correct_mse:.4f}', gap=f'{row["action_gap_mse"]:.2e}',
                lr=f'{row["lr"]:.1e}', vram=f'{row["peak_vram_gb"]:.1f}G',
                sps=f'{row["samples_per_s"]:.1f}',
            )
            cumulative = SOURCE_STEP + done
            if done == 10 or done % 100 == 0 or done == NEW_UPDATES:
                save_checkpoint_no_overwrite(
                    output / f'checkpoint_step{cumulative}.pt', model=model,
                    optimizer=optimizer, step=cumulative, run_config=config,
                    extra={
                        'rng_state': capture_rng_state(), 'new_update': done,
                        'source_model_sha256': source_model_hash,
                        'schedule_set_sha256': plan['schedule_set_sha256'],
                        'derangement_set_sha256': plan['derangement_set_sha256'],
                        'train_cache_key': plan['train_cache_key'],
                    },
                )
            if args.stop_after_update and done == args.stop_after_update:
                stopped = True
                break
        bar.close()
    if stopped:
        (output / 'STATUS').write_text('PAUSED_AFTER_VALIDATED_SMOKE\n')
        return
    summary = {
        'arm': args.arm, 'source_step': SOURCE_STEP, 'new_updates_completed': NEW_UPDATES,
        'final_step': FINAL_STEP, 'source_model_sha256': source_model_hash,
        'final_model_sha256': tensor_state_sha256(model.state_dict()),
        'evaluation_deferred': True,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'STATUS').write_text('TRAINING_DONE_EVALUATION_PENDING\n')


if __name__ == '__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        try: Path(sys.argv[sys.argv.index('--output-dir') + 1], 'STATUS').write_text('FAILED_OOM\n')
        except Exception: pass
        raise SystemExit(2)
    except KeyboardInterrupt:
        traceback.print_exc()
        try: Path(sys.argv[sys.argv.index('--output-dir') + 1], 'STATUS').write_text('INTERRUPTED\n')
        except Exception: pass
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        try: Path(sys.argv[sys.argv.index('--output-dir') + 1], 'STATUS').write_text('FAILED\n')
        except Exception: pass
        raise SystemExit(1)
