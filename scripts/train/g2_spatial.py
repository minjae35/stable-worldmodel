"""Stage-3 G2 controlled 2x2 spatial-dynamics experiment.

Training reads only episode ids, pixels, and actions. Environment-specific state
is loaded only after optimizer updates finish, inside the evaluation phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.pop('HF_ENDPOINT', None)

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from stable_worldmodel.wm.spatial.g2 import (  # noqa: E402
    ENV_NAMES,
    arm_config,
    arm_spec,
    calibrate_loss_scales,
    grad_norm,
    load_label_free_clips,
    make_index_schedule,
    save_checkpoint_no_overwrite,
    schedule_sha256,
    spatial_prediction_loss,
    tensor_state_sha256,
    validate_manifest,
)
from stable_worldmodel.wm.spatial.g2_eval import (  # noqa: E402
    challenge_mask,
    finite_report,
    fit_probe,
    load_state_labels,
    pool4,
    probe_metric_report,
    token_error_metrics,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics  # noqa: E402

MANIFEST = ROOT / 'results/stage3/g2_c1_followup/manifest.json'
G1_CHECKPOINT = ROOT / 'results/stage3/g1_spatial/overfit500/checkpoint_step500.pt'
DINO_WEIGHTS = ROOT / 'results/stage3/g0_dinov2/weights'
DATASETS = {
    'pusht': Path('/workspace/datasets/lewm-pusht/train.lance'),
    'tworoom': Path('/workspace/datasets/lewm-tworooms/train.lance'),
    'cube': Path('/workspace/datasets/lewm-cube/train.lance'),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(images: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    value = value.to(device=device, dtype=torch.float32).div_(255)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (value - mean) / std


@torch.inference_mode()
def encode_images(encoder, images: np.ndarray, device, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, len(images), batch_size):
        pixels = preprocess(images[start : start + batch_size], device)
        tokens = encoder.encode(pixels)[:, 0].float().cpu()
        if tokens.shape[1:] != (256, 384):
            raise RuntimeError(f'unexpected DINO shape {tuple(tokens.shape)}')
        parts.append(tokens)
    return torch.cat(parts)


def normalize_actions(actions: np.ndarray, normalizer: dict) -> torch.Tensor:
    mean = normalizer['mean'].cpu().numpy()
    std = normalizer['std'].cpu().numpy()
    value = (actions.astype(np.float32) - mean) / std
    if not np.isfinite(value).all():
        raise RuntimeError('non-finite normalized action')
    return torch.from_numpy(np.ascontiguousarray(value))


def cap_items(items: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(items):
        return list(items)
    return list(items[:limit])


def load_encoded_bundle(
    dataset,
    items,
    env,
    frameskip,
    normalizer,
    encoder,
    device,
    encode_batch,
):
    raw = load_label_free_clips(dataset, items, env, frameskip)
    context = encode_images(encoder, raw['images_context'], device, encode_batch)
    target = encode_images(encoder, raw['images_target'], device, encode_batch)
    actions = normalize_actions(raw['actions'], normalizer)
    return {
        'context': context,
        'target': target,
        'actions': actions,
        'episode': raw['episode'],
        'start': raw['start'],
        'items': list(items),
    }


def model_batch(model, bundle, indices, env_index, device, actions_override=None):
    take = torch.as_tensor(indices, dtype=torch.long)
    context = bundle['context'].index_select(0, take).to(device)[:, None]
    target = bundle['target'].index_select(0, take).to(device)[:, None]
    actions = bundle['actions'].index_select(0, take).to(device)[:, None]
    if actions_override is not None:
        actions = actions_override
    env_id = torch.full(
        (len(take),), env_index, device=device, dtype=torch.long
    )
    prediction = model(context, actions, env_id)
    return context, actions, env_id, target, prediction


@torch.inference_mode()
def predict_bundle(model, bundle, env_index, device, batch_size, shuffle_seed):
    correct = []
    shuffled = []
    permutation = np.random.default_rng(shuffle_seed).permutation(len(bundle['context']))
    for start in range(0, len(permutation), batch_size):
        stop = min(start + batch_size, len(permutation))
        indices = np.arange(start, stop)
        take = torch.as_tensor(indices, dtype=torch.long)
        context = bundle['context'].index_select(0, take).to(device)[:, None]
        action = bundle['actions'].index_select(0, take).to(device)[:, None]
        shuffled_take = torch.as_tensor(permutation[start:stop], dtype=torch.long)
        shuffled_action = bundle['actions'].index_select(0, shuffled_take).to(device)[:, None]
        env_id = torch.full((len(indices),), env_index, device=device, dtype=torch.long)
        correct.append(model(context, action, env_id)[:, 0].float().cpu())
        shuffled.append(model(context, shuffled_action, env_id)[:, 0].float().cpu())
    return torch.cat(correct), torch.cat(shuffled), permutation


def evaluate_after_training(
    model,
    encoder,
    manifest,
    normalizers,
    device,
    args,
):
    """Common held-out set plus disjoint natural/challenge state subsets."""
    import lance

    model.eval()
    report = {
        'natural_sampling': 'manifest s0 test for every arm',
        'challenge_definition': (
            'target-specific state-labelled object-motion clips; each target reports '
            'challenge and its disjoint complement from held-out test episodes; labels '
            'are loaded only after all optimizer updates'
        ),
        'environments': {},
    }
    split_limits = {
        'train': args.eval_probe_train_items,
        'val': args.eval_items,
        'test': args.eval_items,
    }
    for env_index, env in enumerate(ENV_NAMES):
        dataset = lance.dataset(str(DATASETS[env]))
        bundles = {}
        labels = {}
        for split in ('train', 'val', 'test'):
            items = cap_items(manifest[env]['s0'][split], split_limits[split])
            bundles[split] = load_encoded_bundle(
                dataset,
                items,
                env,
                int(manifest['frameskip']),
                normalizers[env],
                encoder,
                device,
                args.encode_batch,
            )
            labels[split] = load_state_labels(
                dataset, items, env, int(manifest['frameskip'])
            )
        del dataset

        test = bundles['test']
        correct, shuffled, _ = predict_bundle(
            model,
            test,
            env_index,
            device,
            args.eval_batch,
            args.seed + 17,
        )
        env_report = {
            'n_natural': len(test['context']),
            'token_metrics': token_error_metrics(
                correct,
                test['target'],
                test['context'],
                shuffled,
            ),
            'state_metrics': {},
        }
        for target_name, (test_current, test_future) in labels['test'].items():
            kind = 'angle' if target_name == 'block_angle' else 'vector'
            train_current, train_future = labels['train'][target_name]
            val_current, val_future = labels['val'][target_name]
            train_features = np.concatenate(
                [pool4(bundles['train']['context']), pool4(bundles['train']['target'])]
            )
            val_features = np.concatenate(
                [pool4(bundles['val']['context']), pool4(bundles['val']['target'])]
            )
            train_target = np.concatenate([train_current, train_future])
            val_target = np.concatenate([val_current, val_future])
            probe, fit = fit_probe(
                train_features,
                train_target,
                val_features,
                val_target,
                kind,
            )
            features = {
                'true_current_features': pool4(test['context']),
                'true_future_features': pool4(test['target']),
                'predicted_features': pool4(correct),
                'shuffled_features': pool4(shuffled),
                'state_current': test_current,
                'state_future': test_future,
            }
            challenge = challenge_mask(
                env, target_name, test_current, test_future
            )
            natural = ~challenge

            def token_subset(mask):
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
                if not bool(mask_tensor.any()):
                    return {'n': 0, 'reason': 'empty subset'}
                return {
                    'n': int(mask_tensor.sum()),
                    **token_error_metrics(
                        correct[mask_tensor],
                        test['target'][mask_tensor],
                        test['context'][mask_tensor],
                        shuffled[mask_tensor],
                    ),
                }

            env_report['state_metrics'][target_name] = {
                'probe_fit': fit,
                'natural_non_challenge': probe_metric_report(
                    probe, mask=natural, **features
                ),
                'object_motion_challenge': probe_metric_report(
                    probe, mask=challenge, **features
                ),
                'token_subsets': {
                    'natural_non_challenge': token_subset(natural),
                    'object_motion_challenge': token_subset(challenge),
                },
            }
        report['environments'][env] = env_report
    if not finite_report(report):
        raise RuntimeError('evaluation produced non-finite values')
    return report


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=tuple('ABCD'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-per-env', type=int, default=16)
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--encode-batch', type=int, default=32)
    parser.add_argument('--eval-batch', type=int, default=32)
    parser.add_argument('--max-train-items', type=int, default=0)
    parser.add_argument('--eval-items', type=int, default=0)
    parser.add_argument('--eval-probe-train-items', type=int, default=0)
    parser.add_argument('--loss-calibration-file', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--defer-evaluation', action='store_true')
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    spec = arm_spec(args.arm)
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
        'initial_checkpoint': str(G1_CHECKPOINT),
        'normalizer_source': 'G1 checkpoint, identical for all warm-start arms',
        'manifest': str(MANIFEST),
        'manifest_sha256': sha256_file(MANIFEST),
        'checkpoint_sha256': sha256_file(G1_CHECKPOINT),
        'dino_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
        'frameskip': int(manifest['frameskip']),
        'max_train_items': args.max_train_items,
        'eval_items': args.eval_items,
        'eval_probe_train_items': args.eval_probe_train_items,
        'loss_calibration_file': (
            str(args.loss_calibration_file.resolve())
            if args.loss_calibration_file is not None
            else None
        ),
    }
    config = arm_config(base, args.arm)
    config['output_dir'] = str(output)
    config['manifest_summary'] = manifest_summary
    config_path = output / 'config.json'
    if args.resume is None:
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    elif json.loads(config_path.read_text()) != config:
        raise ValueError('existing run config differs from resume config')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')

    checkpoint = torch.load(G1_CHECKPOINT, map_location='cpu', weights_only=True)
    normalizers = checkpoint['normalizer']
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)
    model = SpatialDynamics().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    initial_hash = tensor_state_sha256(model.state_dict())

    import lance

    bundles = {}
    datasets = {}
    for env in ENV_NAMES:
        datasets[env] = lance.dataset(str(DATASETS[env]))
        items = cap_items(
            manifest[env][spec.sampling]['train'], args.max_train_items
        )
        bundles[env] = load_encoded_bundle(
            datasets[env],
            items,
            env,
            int(manifest['frameskip']),
            normalizers[env],
            encoder,
            device,
            args.encode_batch,
        )
    sizes = {env: len(bundle['context']) for env, bundle in bundles.items()}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f'unbalanced training clips: {sizes}')
    n_items = next(iter(sizes.values()))
    schedule = make_index_schedule(
        n_items, args.batch_per_env, args.steps, args.seed + 101
    )

    calibration_batches = []
    for env_index, env in enumerate(ENV_NAMES):
        if spec.sampling == 's0':
            calibration = bundles[env]
        else:
            calibration_items = manifest[env]['s0']['train'][: args.batch_per_env]
            calibration = load_encoded_bundle(
                datasets[env],
                calibration_items,
                env,
                int(manifest['frameskip']),
                normalizers[env],
                encoder,
                device,
                args.encode_batch,
            )
        indices = np.arange(args.batch_per_env)
        context, actions, env_id, target, _ = model_batch(
            model, calibration, indices, env_index, device
        )
        calibration_batches.append((context, actions, env_id, target))
    measured_calibration = calibrate_loss_scales(model, calibration_batches)
    calibration_key = {
        'checkpoint_sha256': base['checkpoint_sha256'],
        'seed': args.seed,
        'batch_per_env': args.batch_per_env,
        'reference_sampling': 's0',
    }
    if args.loss_calibration_file is None:
        calibration = measured_calibration
    else:
        calibration_path = args.loss_calibration_file.resolve()
        if calibration_path.exists():
            artifact = json.loads(calibration_path.read_text())
            if artifact['key'] != calibration_key:
                raise ValueError('shared loss calibration key does not match this run')
            calibration = artifact['calibration']
        else:
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            artifact = {
                'key': calibration_key,
                'calibration': measured_calibration,
            }
            with calibration_path.open('x') as handle:
                json.dump(artifact, handle, indent=2)
                handle.write('\n')
            calibration = measured_calibration
    loss_scale = calibration['scale'][spec.loss]
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
        'loss_scale': loss_scale,
        'training_columns': ['episode_idx', 'pixels', 'action'],
        'forbidden_training_inputs': manifest['forbidden_in_sampling_and_loss'],
    }
    (output / 'reproducibility.json').write_text(json.dumps(metadata, indent=2) + '\n')

    fields = [
        'step',
        'objective',
        'objective_scaled',
        'pusht_loss',
        'tworoom_loss',
        'cube_loss',
        'mse_common',
        'l1_common',
        'grad_norm',
        'lr',
        'step_time_s',
        'samples_per_s',
        'peak_vram_gb',
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
            common_mse = []
            common_l1 = []
            for env_index, env in enumerate(ENV_NAMES):
                _, _, _, target, prediction = model_batch(
                    model, bundles[env], schedule[step_index], env_index, device
                )
                raw = spatial_prediction_loss(prediction, target, spec.loss)
                losses[env] = raw
                common_mse.append(spatial_prediction_loss(prediction, target, 'mse'))
                common_l1.append(spatial_prediction_loss(prediction, target, 'l1'))
            objective = torch.stack([losses[env] for env in ENV_NAMES]).mean()
            scaled = objective * loss_scale
            if not bool(torch.isfinite(scaled)):
                raise FloatingPointError(f'non-finite loss at step {step_index + 1}')
            scaled.backward()
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
                'objective_scaled': float(scaled.detach()),
                'pusht_loss': float(losses['pusht'].detach()),
                'tworoom_loss': float(losses['tworoom'].detach()),
                'cube_loss': float(losses['cube'].detach()),
                'mse_common': float(torch.stack(common_mse).mean().detach()),
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
                arm=args.arm,
                p=f'{row["pusht_loss"]:.3f}',
                t=f'{row["tworoom_loss"]:.3f}',
                c=f'{row["cube_loss"]:.3f}',
                lr=f'{row["lr"]:.1e}',
                vram=f'{row["peak_vram_gb"]:.2f}G',
                sps=f'{row["samples_per_s"]:.1f}',
            )
            if done % args.checkpoint_every == 0 or done == args.steps:
                save_checkpoint_no_overwrite(
                    output / f'checkpoint_step{done}.pt',
                    model=model,
                    optimizer=optimizer,
                    step=done,
                    run_config=config,
                    extra=metadata,
                )
        bar.close()

    evaluation_path = None
    if not args.defer_evaluation:
        # Labels first become reachable here, after all optimizer steps.
        evaluation = evaluate_after_training(
            model, encoder, manifest, normalizers, device, args
        )
        evaluation_path = output / 'evaluation.json'
        evaluation_path.write_text(json.dumps(evaluation, indent=2) + '\n')
    summary = {
        'arm': args.arm,
        'sampling': spec.sampling,
        'loss': spec.loss,
        'steps_completed': args.steps,
        'initial_model_sha256': initial_hash,
        'final_model_sha256': tensor_state_sha256(model.state_dict()),
        'schedule_sha256': metadata['schedule_sha256'],
        'loss_calibration': calibration,
        'evaluation': None if evaluation_path is None else str(evaluation_path),
        'evaluation_deferred': bool(args.defer_evaluation),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    status.write_text(
        'TRAINING_DONE_EVALUATION_DEFERRED\n'
        if args.defer_evaluation
        else 'DONE\n'
    )
    print(f'G2_DONE arm={args.arm} steps={args.steps}', flush=True)


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
            output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
            final_checkpoint = output / 'checkpoint_step500.pt'
            status = (
                'TRAINING_DONE_EVALUATION_INTERRUPTED\n'
                if final_checkpoint.is_file()
                else 'INTERRUPTED\n'
            )
            (output / 'STATUS').write_text(status)
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
