"""Strictly extend one G2 Fresh A/B arm from step 500 to step 2,000."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_spatial import (  # noqa: E402
    DATASETS,
    DINO_WEIGHTS,
    MANIFEST,
    cap_items,
    load_encoded_bundle,
    model_batch,
    sha256_file,
)
from stable_worldmodel.wm.spatial.g2 import (  # noqa: E402
    ENV_NAMES,
    calibrate_loss_scales,
    config_sha256,
    grad_norm,
    make_index_schedule,
    save_checkpoint_no_overwrite,
    schedule_sha256,
    spatial_prediction_loss,
    tensor_state_sha256,
    validate_manifest,
)
from stable_worldmodel.wm.spatial.g2_extension import (  # noqa: E402
    capture_rng_state,
    optimizer_steps,
    restore_rng_state,
    validate_schedule_prefix,
    validate_source_checkpoint,
)
from stable_worldmodel.wm.spatial.g2_fresh import (  # noqa: E402
    prepare_action_normalizers,
    torch_normalizers,
    validate_sampling_episode_alignment,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics  # noqa: E402


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--arm', required=True, choices=('A', 'B'))
    result.add_argument('--source-dir', type=Path, required=True)
    result.add_argument('--source-checkpoint', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--normalizer-file', type=Path, required=True)
    result.add_argument('--loss-calibration-file', type=Path, required=True)
    result.add_argument('--total-steps', type=int, default=2000)
    result.add_argument('--checkpoint-steps', type=int, nargs='+', default=(1000, 1500, 2000))
    result.add_argument('--eval-batch', type=int, default=32)
    result.add_argument('--encode-batch', type=int, default=32)
    result.add_argument('--resume', type=Path)
    result.add_argument('--verify-only', action='store_true')
    return result


def move_optimizer(optimizer):
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(parameter.device)


def main(argv=None):
    args = parser().parse_args(argv)
    source_dir = args.source_dir.resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    output = args.output_dir.resolve()
    source_config = json.loads((source_dir / 'config.json').read_text())
    expected_sampling = 's0' if args.arm == 'A' else 's1'
    if source_config['name'] != args.arm or source_config['sampling'] != expected_sampling:
        raise ValueError('source arm identity does not match requested extension arm')
    if source_config['loss'] != 'mse' or source_config['initial_checkpoint'] is not None:
        raise ValueError('source is not the expected fresh-init MSE run')
    if int(source_config['steps']) != 500:
        raise ValueError('source run is not the 500-step pilot')
    source = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
    validate_source_checkpoint(source, source_config, 500)
    if args.total_steps != 2000:
        raise ValueError('this controlled extension requires total_steps=2000')
    if sorted(set(args.checkpoint_steps)) != [1000, 1500, 2000]:
        raise ValueError('required checkpoint steps are 1000, 1500, and 2000')

    manifest = json.loads(MANIFEST.read_text())
    validate_manifest(manifest)
    validate_sampling_episode_alignment(manifest)
    if sha256_file(MANIFEST) != source_config['manifest_sha256']:
        raise ValueError('manifest changed since the source run')
    if sha256_file(DINO_WEIGHTS / 'model.safetensors') != source_config['dino_sha256']:
        raise ValueError('DINO weights changed since the source run')
    if sha256_file(args.normalizer_file) != source_config['normalizer_sha256']:
        raise ValueError('action normalizer changed since the source run')

    seed = int(source_config['seed'])
    batch = int(source_config['batch_per_env'])
    schedule = make_index_schedule(1024, batch, args.total_steps, seed + 101)
    prefix_hash = validate_schedule_prefix(
        schedule, 500, source['extra']['schedule_sha256']
    )
    if args.verify_only:
        print(json.dumps({
            'arm': args.arm,
            'source_checkpoint_sha256': sha256_file(source_checkpoint),
            'source_step': source['step'],
            'optimizer_steps': sorted(optimizer_steps(source['optimizer'])),
            'lr': source['optimizer']['param_groups'][0]['lr'],
            'scheduler': None,
            'schedule_prefix_sha256': prefix_hash,
            'full_schedule_sha256': schedule_sha256(schedule),
            'rng_snapshot_in_source': False,
            'rng_reconstruction': 'safe: dropout=0 and sampler uses an isolated deterministic generator',
        }, indent=2))
        return

    if args.resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir() or args.resume.resolve().parent != output:
        raise ValueError('extension resume checkpoint must be inside its output directory')
    status = output / 'STATUS'
    status.write_text('RUNNING\n')

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SpatialDynamics().to(device)
    initial_hash = tensor_state_sha256(model.state_dict())
    if initial_hash != source_config['initial_model_sha256']:
        raise ValueError('fresh model initialization no longer reproduces source hash')
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)

    import lance
    datasets = {env: lance.dataset(str(DATASETS[env])) for env in ENV_NAMES}
    normalizer_artifact = prepare_action_normalizers(
        args.normalizer_file.resolve(), manifest, sha256_file(MANIFEST),
        datasets, DATASETS, int(manifest['frameskip']),
    )
    normalizers = torch_normalizers(normalizer_artifact)
    bundles = {}
    for env in ENV_NAMES:
        items = manifest[env][expected_sampling]['train']
        bundles[env] = load_encoded_bundle(
            datasets[env], items, env, int(manifest['frameskip']), normalizers[env],
            encoder, device, args.encode_batch,
        )
    sizes = {env: len(value['context']) for env, value in bundles.items()}
    if sizes != {env: 1024 for env in ENV_NAMES}:
        raise ValueError(f'unexpected training cohort sizes: {sizes}')

    calibration_batches = []
    for env_index, env in enumerate(ENV_NAMES):
        calibration = bundles[env]
        if expected_sampling == 's1':
            calibration = load_encoded_bundle(
                datasets[env], manifest[env]['s0']['train'][:batch], env,
                int(manifest['frameskip']), normalizers[env], encoder, device,
                args.encode_batch,
            )
        context, actions, env_id, target, _ = model_batch(
            model, calibration, np.arange(batch), env_index, device
        )
        calibration_batches.append((context, actions, env_id, target))
    measured = calibrate_loss_scales(model, calibration_batches)
    artifact = json.loads(args.loss_calibration_file.read_text())
    if artifact['key']['initial_model_sha256'] != initial_hash:
        raise ValueError('loss calibration initial model mismatch')
    if artifact['calibration'] != source['extra']['loss_calibration']:
        raise ValueError('loss calibration differs from source checkpoint')
    if tensor_state_sha256(model.state_dict()) != initial_hash:
        raise RuntimeError('calibration changed the model')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(source_config['lr']),
        weight_decay=float(source_config['weight_decay']),
    )
    extension_config = {
        **source_config,
        'steps': args.total_steps,
        'output_dir': str(output),
        'extension': {
            'source_dir': str(source_dir),
            'source_checkpoint': str(source_checkpoint),
            'source_checkpoint_sha256': sha256_file(source_checkpoint),
            'source_config_sha256': config_sha256(source_config),
            'source_step': 500,
            'checkpoint_steps': [1000, 1500, 2000],
            'lr_schedule': 'none_constant_lr',
            'rng_reconstruction': (
                'source omitted snapshots; exact because dropout=0, no augmentation, '
                'and sampler is a deterministic isolated schedule'
            ),
        },
    }
    config_path = output / 'config.json'
    if args.resume is None:
        config_path.write_text(json.dumps(extension_config, indent=2) + '\n')
        model.load_state_dict(source['model'], strict=True)
        optimizer.load_state_dict(source['optimizer'])
        move_optimizer(optimizer)
        start_step = 500
    else:
        if json.loads(config_path.read_text()) != extension_config:
            raise ValueError('extension config changed before resume')
        payload = torch.load(args.resume, map_location='cpu', weights_only=True)
        if payload['config_sha256'] != config_sha256(extension_config):
            raise ValueError('extension resume config hash mismatch')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        move_optimizer(optimizer)
        restore_rng_state(payload['extra']['rng_state'])
        start_step = int(payload['step'])
    if optimizer_steps(optimizer.state_dict()) != {start_step}:
        raise ValueError('restored optimizer step is inconsistent')
    if optimizer.param_groups[0]['lr'] != float(source_config['lr']):
        raise ValueError('restored optimizer LR is inconsistent')
    if start_step == 500:
        source_model_hash = tensor_state_sha256(source['model'])
        if tensor_state_sha256(model.state_dict()) != source_model_hash:
            raise ValueError('restored model differs from source checkpoint')

    validation = {
        'source_checkpoint_sha256': sha256_file(source_checkpoint),
        'source_model_sha256': tensor_state_sha256(source['model']),
        'restored_model_sha256': tensor_state_sha256(model.state_dict()),
        'source_optimizer_steps': sorted(optimizer_steps(source['optimizer'])),
        'restored_optimizer_steps': sorted(optimizer_steps(optimizer.state_dict())),
        'lr': optimizer.param_groups[0]['lr'],
        'scheduler': None,
        'schedule_prefix_sha256': prefix_hash,
        'full_schedule_sha256': schedule_sha256(schedule),
        'source_rng_snapshot_present': False,
        'stochastic_training_layers': False,
        'sampler_state': 'derived from seed and current step; no mutable cursor',
    }
    (output / 'resume_validation.json').write_text(json.dumps(validation, indent=2) + '\n')

    fields = [
        'step', 'objective', 'objective_scaled', 'pusht_loss', 'tworoom_loss',
        'cube_loss', 'mse_common', 'l1_common', 'grad_norm', 'lr',
        'step_time_s', 'samples_per_s', 'peak_vram_gb',
    ]
    metrics = output / 'metrics_extension.csv'
    mode = 'a' if args.resume is not None else 'x'
    with metrics.open(mode, newline='', buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if args.resume is None:
            writer.writeheader()
        parameters = [p for p in model.parameters() if p.requires_grad]
        model.train()
        torch.cuda.reset_peak_memory_stats()
        bar = trange(start_step, args.total_steps, initial=start_step, total=args.total_steps, dynamic_ncols=True)
        for step_index in bar:
            torch.cuda.synchronize()
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            common_l1 = []
            for env_index, env in enumerate(ENV_NAMES):
                _, _, _, target, prediction = model_batch(
                    model, bundles[env], schedule[step_index], env_index, device
                )
                losses[env] = spatial_prediction_loss(prediction, target, 'mse')
                common_l1.append(spatial_prediction_loss(prediction, target, 'l1'))
            objective = torch.stack([losses[env] for env in ENV_NAMES]).mean()
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(f'non-finite loss at step {step_index + 1}')
            objective.backward()
            norm = grad_norm(parameters)
            if not np.isfinite(norm):
                raise FloatingPointError(f'non-finite gradient at step {step_index + 1}')
            torch.nn.utils.clip_grad_norm_(parameters, float(source_config['grad_clip']))
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            done = step_index + 1
            row = {
                'step': done,
                'objective': float(objective.detach()),
                'objective_scaled': float(objective.detach()),
                'pusht_loss': float(losses['pusht'].detach()),
                'tworoom_loss': float(losses['tworoom'].detach()),
                'cube_loss': float(losses['cube'].detach()),
                'mse_common': float(objective.detach()),
                'l1_common': float(torch.stack(common_l1).mean().detach()),
                'grad_norm': norm,
                'lr': optimizer.param_groups[0]['lr'],
                'step_time_s': elapsed,
                'samples_per_s': len(ENV_NAMES) * batch / elapsed,
                'peak_vram_gb': torch.cuda.max_memory_allocated() / (1024**3),
            }
            writer.writerow(row)
            handle.flush()
            bar.set_postfix(
                arm=args.arm, p=f'{row["pusht_loss"]:.3f}',
                t=f'{row["tworoom_loss"]:.3f}', c=f'{row["cube_loss"]:.3f}',
                lr=f'{row["lr"]:.1e}', vram=f'{row["peak_vram_gb"]:.2f}G',
                sps=f'{row["samples_per_s"]:.1f}',
            )
            if done in args.checkpoint_steps:
                extra = {
                    **validation,
                    'rng_state': capture_rng_state(),
                    'normalizer_key': normalizer_artifact['key'],
                }
                save_checkpoint_no_overwrite(
                    output / f'checkpoint_step{done}.pt', model=model,
                    optimizer=optimizer, step=done,
                    run_config=extension_config, extra=extra,
                )
        bar.close()
    summary = {
        'arm': args.arm,
        'source_step': 500,
        'steps_completed': args.total_steps,
        'updates_added': args.total_steps - 500,
        'final_model_sha256': tensor_state_sha256(model.state_dict()),
        'full_schedule_sha256': schedule_sha256(schedule),
        'evaluation_deferred': True,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    status.write_text('TRAINING_DONE_EVALUATION_DEFERRED\n')
    print(f'G2_FRESH_EXTENSION_DONE arm={args.arm} steps={args.total_steps}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        try:
            Path(sys.argv[sys.argv.index('--output-dir') + 1], 'STATUS').write_text('FAILED_OOM\n')
        except Exception:
            pass
        raise SystemExit(2)
    except KeyboardInterrupt:
        traceback.print_exc()
        try:
            Path(sys.argv[sys.argv.index('--output-dir') + 1], 'STATUS').write_text('INTERRUPTED\n')
        except Exception:
            pass
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        try:
            Path(sys.argv[sys.argv.index('--output-dir') + 1], 'STATUS').write_text('FAILED\n')
        except Exception:
            pass
        raise SystemExit(1)
