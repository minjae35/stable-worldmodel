"""Fresh-initialization Stage-3 G2 A/B controlled training."""

from __future__ import annotations

import argparse
import csv
import json
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
    arm_config,
    arm_spec,
    calibrate_loss_scales,
    grad_norm,
    make_index_schedule,
    save_checkpoint_no_overwrite,
    schedule_sha256,
    spatial_prediction_loss,
    tensor_state_sha256,
    validate_manifest,
)
from stable_worldmodel.wm.spatial.g2_fresh import (  # noqa: E402
    prepare_action_normalizers,
    torch_normalizers,
    validate_sampling_episode_alignment,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=('A', 'B'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--normalizer-file', type=Path, required=True)
    parser.add_argument('--loss-calibration-file', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-per-env', type=int, default=16)
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--encode-batch', type=int, default=32)
    parser.add_argument('--max-train-items', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--defer-evaluation', action='store_true', required=True)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    spec = arm_spec(args.arm)
    if spec.loss != 'mse':
        raise ValueError('fresh A/B pilot is restricted to the MSE objective')
    output = args.output_dir.resolve()
    if args.steps < 1 or args.batch_per_env < 1:
        raise ValueError('steps and batch size must be positive')
    if args.resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir() or args.resume.parent.resolve() != output:
        raise ValueError('resume checkpoint must be inside the existing output directory')
    status = output / 'STATUS'
    status.write_text('RUNNING\n')

    manifest = json.loads(MANIFEST.read_text())
    manifest_summary = validate_manifest(manifest)
    validate_sampling_episode_alignment(manifest)
    frameskip = int(manifest['frameskip'])

    import lance

    datasets = {env: lance.dataset(str(DATASETS[env])) for env in ENV_NAMES}
    normalizer_path = args.normalizer_file.resolve()
    normalizer_artifact = prepare_action_normalizers(
        normalizer_path,
        manifest,
        sha256_file(MANIFEST),
        datasets,
        DATASETS,
        frameskip,
    )
    normalizers = torch_normalizers(normalizer_artifact)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')

    # Reset immediately before construction so encoder loading cannot influence
    # the shared fresh initialization.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = SpatialDynamics().to(device)
    initial_hash = tensor_state_sha256(model.state_dict())
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)

    base = {
        'seed': args.seed,
        'steps': args.steps,
        'batch_per_env': args.batch_per_env,
        'optimizer': 'AdamW',
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'balanced_environments': list(ENV_NAMES),
        'model': 'Frozen DINOv2-S/14 + shared SpatialDynamics',
        'initialization': 'fresh_spatial_dynamics_seeded_before_construction',
        'initial_checkpoint': None,
        'initial_model_sha256': initial_hash,
        'normalizer_source': 'all raw action rows from manifest train episodes',
        'normalizer_file': str(normalizer_path),
        'normalizer_key': normalizer_artifact['key'],
        'normalizer_sha256': sha256_file(normalizer_path),
        'manifest': str(MANIFEST),
        'manifest_sha256': sha256_file(MANIFEST),
        'dino_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
        'frameskip': frameskip,
        'max_train_items': args.max_train_items,
        'loss_calibration_file': str(args.loss_calibration_file.resolve()),
        'evaluation_deferred': True,
    }
    config = arm_config(base, args.arm)
    config['output_dir'] = str(output)
    config['manifest_summary'] = manifest_summary
    config_path = output / 'config.json'
    if args.resume is None:
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    elif json.loads(config_path.read_text()) != config:
        raise ValueError('existing run config differs from resume config')

    bundles = {}
    for env in ENV_NAMES:
        items = cap_items(manifest[env][spec.sampling]['train'], args.max_train_items)
        bundles[env] = load_encoded_bundle(
            datasets[env], items, env, frameskip, normalizers[env], encoder,
            device, args.encode_batch,
        )
    sizes = {env: len(bundle['context']) for env, bundle in bundles.items()}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f'unbalanced training clips: {sizes}')
    n_items = next(iter(sizes.values()))
    schedule = make_index_schedule(n_items, args.batch_per_env, args.steps, args.seed + 101)

    calibration_batches = []
    for env_index, env in enumerate(ENV_NAMES):
        if spec.sampling == 's0':
            calibration = bundles[env]
        else:
            calibration_items = manifest[env]['s0']['train'][: args.batch_per_env]
            calibration = load_encoded_bundle(
                datasets[env], calibration_items, env, frameskip,
                normalizers[env], encoder, device, args.encode_batch,
            )
        context, actions, env_id, target, _ = model_batch(
            model, calibration, np.arange(args.batch_per_env), env_index, device
        )
        calibration_batches.append((context, actions, env_id, target))
    measured_calibration = calibrate_loss_scales(model, calibration_batches)
    calibration_key = {
        'initial_model_sha256': initial_hash,
        'normalizer_key': normalizer_artifact['key'],
        'seed': args.seed,
        'batch_per_env': args.batch_per_env,
        'reference_sampling': 's0',
    }
    calibration_path = args.loss_calibration_file.resolve()
    if calibration_path.exists():
        artifact = json.loads(calibration_path.read_text())
        if artifact['key'] != calibration_key:
            raise ValueError('shared fresh loss calibration key mismatch')
        calibration = artifact['calibration']
    else:
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {'key': calibration_key, 'calibration': measured_calibration}
        with calibration_path.open('x') as handle:
            json.dump(artifact, handle, indent=2)
            handle.write('\n')
        calibration = measured_calibration
    if tensor_state_sha256(model.state_dict()) != initial_hash:
        raise RuntimeError('loss calibration changed model parameters')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_step = 0
    if args.resume is not None:
        from stable_worldmodel.wm.spatial.g2 import load_resume_checkpoint
        resumed = load_resume_checkpoint(args.resume, model, optimizer, config)
        start_step = int(resumed['step'])
        if start_step >= args.steps:
            raise ValueError('resume checkpoint is already at or beyond requested steps')

    metadata = {
        'initial_model_sha256': initial_hash,
        'schedule_sha256': schedule_sha256(schedule),
        'loss_calibration': calibration,
        'locally_measured_loss_calibration': measured_calibration,
        'loss_scale': 1.0,
        'normalizer_key': normalizer_artifact['key'],
        'normalizer_sha256': sha256_file(normalizer_path),
        'training_columns': ['episode_idx', 'pixels', 'action'],
        'forbidden_training_inputs': manifest['forbidden_in_sampling_and_loss'],
    }
    (output / 'reproducibility.json').write_text(json.dumps(metadata, indent=2) + '\n')

    fields = [
        'step', 'objective', 'objective_scaled', 'pusht_loss', 'tworoom_loss',
        'cube_loss', 'mse_common', 'l1_common', 'grad_norm', 'lr',
        'step_time_s', 'samples_per_s', 'peak_vram_gb',
    ]
    metrics_path = output / 'metrics.csv'
    mode = 'a' if args.resume is not None else 'x'
    with metrics_path.open(mode, newline='', buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if args.resume is None:
            writer.writeheader()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        model.train()
        torch.cuda.reset_peak_memory_stats()
        bar = trange(start_step, args.steps, initial=start_step, total=args.steps, dynamic_ncols=True)
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
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
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
                'samples_per_s': len(ENV_NAMES) * args.batch_per_env / elapsed,
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
            if done % args.checkpoint_every == 0 or done == args.steps:
                save_checkpoint_no_overwrite(
                    output / f'checkpoint_step{done}.pt', model=model,
                    optimizer=optimizer, step=done, run_config=config, extra=metadata,
                )
        bar.close()

    summary = {
        'arm': args.arm,
        'sampling': spec.sampling,
        'loss': 'mse',
        'steps_completed': args.steps,
        'initial_model_sha256': initial_hash,
        'final_model_sha256': tensor_state_sha256(model.state_dict()),
        'schedule_sha256': metadata['schedule_sha256'],
        'normalizer_key': normalizer_artifact['key'],
        'evaluation': None,
        'evaluation_deferred': True,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    status.write_text('TRAINING_DONE_EVALUATION_DEFERRED\n')
    print(f'G2_FRESH_DONE arm={args.arm} steps={args.steps}', flush=True)


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
