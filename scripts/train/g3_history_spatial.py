"""Stage-3 G3 fresh H1/H3 controlled training on common target anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    encode_images,
    sha256_file,
)
from stable_worldmodel.wm.spatial.g2 import (  # noqa: E402
    ENV_NAMES,
    grad_norm,
    make_index_schedule,
    save_checkpoint_no_overwrite,
    schedule_sha256,
    spatial_prediction_loss,
    tensor_state_sha256,
)
from stable_worldmodel.wm.spatial.g2_extension import (  # noqa: E402
    capture_rng_state,
    restore_rng_state,
)
from stable_worldmodel.wm.spatial.g2_fresh import torch_normalizers  # noqa: E402
from stable_worldmodel.wm.spatial.g3_history import (  # noqa: E402
    history_model_batch,
    load_label_free_history_clips,
    normalize_action_history,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics  # noqa: E402

ARM_HISTORY = {'H1': 1, 'H3': 3}


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(ARM_HISTORY), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--normalizer-file', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-per-env', type=int, default=16)
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--encode-batch', type=int, default=32)
    parser.add_argument('--max-train-items', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument(
        '--temporal-encoding', choices=('none', 'sinusoidal'), default='none'
    )
    parser.add_argument('--temporal-encoding-scale', type=float, default=1.0)
    parser.add_argument('--experiment-name', default='g3_history_h1_vs_h3')
    parser.add_argument('--stop-after-step', type=int, default=0)
    return parser


def encode_bundle(raw, normalizer, encoder, device, encode_batch):
    images = raw['images_context']
    n, history = images.shape[:2]
    context = encode_images(
        encoder, images.reshape(n * history, *images.shape[2:]), device, encode_batch
    ).reshape(n, history, 256, 384)
    target = encode_images(encoder, raw['images_target'], device, encode_batch)
    return {
        'context': context,
        'target': target,
        'actions': normalize_action_history(raw['actions'], normalizer),
        'episode': raw['episode'],
        'start': raw['start'],
    }


def main(argv=None) -> None:
    args = make_parser().parse_args(argv)
    history = ARM_HISTORY[args.arm]
    output = args.output_dir.resolve()
    if args.steps < 1 or args.batch_per_env < 1:
        raise ValueError('steps and batch size must be positive')
    if args.resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir() or args.resume.parent.resolve() != output:
        raise ValueError('resume checkpoint must be inside the output directory')
    status = output / 'STATUS'
    status.write_text('RUNNING\n')

    common_path = args.common_manifest.resolve()
    common = json.loads(common_path.read_text())
    if int(common['max_history']) != 3 or int(common['frameskip']) != 5:
        raise ValueError('G3 requires max_history=3 and frameskip=5')
    normalizer_path = args.normalizer_file.resolve()
    normalizer_artifact = json.loads(normalizer_path.read_text())
    normalizers = torch_normalizers(normalizer_artifact)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = SpatialDynamics(
        temporal_encoding=args.temporal_encoding,
        temporal_encoding_scale=args.temporal_encoding_scale,
    ).to(device)
    initial_hash = tensor_state_sha256(model.state_dict())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)

    config = {
        'experiment': args.experiment_name,
        'arm': args.arm,
        'history': history,
        'seed': args.seed,
        'steps': args.steps,
        'batch_per_env': args.batch_per_env,
        'optimizer': 'AdamW',
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'sampling': 'uniform_s0_common_history_valid_anchors',
        'loss': 'mse',
        'frameskip': 5,
        'balanced_environment_batches': list(ENV_NAMES),
        'initialization': 'fresh_seeded_spatial_dynamics',
        'initial_model_sha256': initial_hash,
        'model_parameter_count': parameter_count,
        'architecture_changed': args.temporal_encoding != 'none',
        'target_definition': 'anchor_t_plus_5',
        'common_manifest': str(common_path),
        'common_manifest_sha256': sha256_file(common_path),
        'normalizer_file': str(normalizer_path),
        'normalizer_key': normalizer_artifact['key'],
        'normalizer_sha256': sha256_file(normalizer_path),
        'dino_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
        'max_train_items': args.max_train_items,
        'output_dir': str(output),
    }
    if args.temporal_encoding != 'none':
        config.update({
            'temporal_encoding': args.temporal_encoding,
            'temporal_encoding_scale': args.temporal_encoding_scale,
            'temporal_encoding_positions': [0, 1, 2],
            'temporal_encoding_application': (
                'added to projected spatial tokens after aligned action and environment '
                'embedding, once before the factorized spatiotemporal block stack'
            ),
            'temporal_encoding_trainable_parameters': 0,
        })
    config_path = output / 'config.json'
    if args.resume is None:
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    elif json.loads(config_path.read_text()) != config:
        raise ValueError('existing run config differs from resume config')

    import lance

    datasets = {env: lance.dataset(str(DATASETS[env])) for env in ENV_NAMES}
    bundles = {}
    schedules = {}
    schedule_hashes = {}
    for env_index, env in enumerate(ENV_NAMES):
        items = list(common['environments'][env]['train'])
        if args.max_train_items > 0:
            items = items[: args.max_train_items]
        if len(items) < args.batch_per_env:
            raise ValueError(f'{env} has too few common anchors: {len(items)}')
        raw = load_label_free_history_clips(
            datasets[env], items, env, 5, history
        )
        bundles[env] = encode_bundle(
            raw, normalizers[env], encoder, device, args.encode_batch
        )
        schedule = make_index_schedule(
            len(items), args.batch_per_env, args.steps, args.seed + 101 + env_index
        )
        schedules[env] = schedule
        schedule_hashes[env] = schedule_sha256(schedule)

    metadata = {
        'initial_model_sha256': initial_hash,
        'schedule_sha256_by_environment': schedule_hashes,
        'schedule_set_sha256': canonical_hash(schedule_hashes),
        'normalizer_key': normalizer_artifact['key'],
        'normalizer_sha256': sha256_file(normalizer_path),
        'training_columns': ['episode_idx', 'pixels', 'action'],
        'forbidden_training_inputs': common['forbidden_training_inputs'],
        'history_alignment': common['alignment'],
        'model_parameter_count': parameter_count,
    }
    if args.temporal_encoding != 'none':
        metadata.update({
            'temporal_encoding': args.temporal_encoding,
            'temporal_encoding_scale': args.temporal_encoding_scale,
            'temporal_encoding_positions': [0, 1, 2],
        })
    (output / 'reproducibility.json').write_text(json.dumps(metadata, indent=2) + '\n')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_step = 0
    if args.resume is not None:
        from stable_worldmodel.wm.spatial.g2 import load_resume_checkpoint

        resumed = load_resume_checkpoint(args.resume, model, optimizer, config)
        if 'rng_state' not in resumed.get('extra', {}):
            raise ValueError('resume checkpoint lacks required RNG state')
        restore_rng_state(resumed['extra']['rng_state'])
        start_step = int(resumed['step'])
        if start_step >= args.steps:
            raise ValueError('resume checkpoint is already at or beyond requested steps')

    fields = [
        'step', 'objective', 'pusht_loss', 'tworoom_loss', 'cube_loss',
        'mse_common', 'l1_common', 'grad_norm', 'lr', 'step_time_s',
        'samples_per_s', 'peak_vram_gb',
    ]
    metrics_path = output / 'metrics.csv'
    mode = 'a' if args.resume is not None else 'x'
    with metrics_path.open(mode, newline='', buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if args.resume is None:
            writer.writeheader()
        parameters = [p for p in model.parameters() if p.requires_grad]
        model.train()
        torch.cuda.reset_peak_memory_stats()
        bar = trange(start_step, args.steps, initial=start_step, total=args.steps, dynamic_ncols=True)
        stopped_for_resume_test = False
        for step_index in bar:
            torch.cuda.synchronize()
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            l1s = []
            for env_index, env in enumerate(ENV_NAMES):
                _, _, _, target, prediction = history_model_batch(
                    model, bundles[env], schedules[env][step_index], env_index, device
                )
                losses[env] = spatial_prediction_loss(prediction, target, 'mse')
                l1s.append(spatial_prediction_loss(prediction, target, 'l1'))
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
                'pusht_loss': float(losses['pusht'].detach()),
                'tworoom_loss': float(losses['tworoom'].detach()),
                'cube_loss': float(losses['cube'].detach()),
                'mse_common': float(objective.detach()),
                'l1_common': float(torch.stack(l1s).mean().detach()),
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
                    optimizer=optimizer, step=done, run_config=config,
                    extra={**metadata, 'rng_state': capture_rng_state()},
                )
            if args.stop_after_step and done == args.stop_after_step:
                stopped_for_resume_test = True
                break
        bar.close()

    if stopped_for_resume_test:
        status.write_text('PAUSED_FOR_RESUME_TEST\n')
        return

    summary = {
        'arm': args.arm,
        'history': history,
        'steps_completed': args.steps,
        'initial_model_sha256': initial_hash,
        'final_model_sha256': tensor_state_sha256(model.state_dict()),
        'schedule_sha256_by_environment': schedule_hashes,
        'model_parameter_count': parameter_count,
        'evaluation_deferred': True,
    }
    if args.temporal_encoding != 'none':
        summary.update({
            'temporal_encoding': args.temporal_encoding,
            'temporal_encoding_scale': args.temporal_encoding_scale,
        })
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    status.write_text('TRAINING_DONE_EVALUATION_DEFERRED\n')


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
