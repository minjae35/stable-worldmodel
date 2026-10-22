"""Evaluate all G2 Fresh multi-seed A/B checkpoints on one frozen cohort."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from scripts.train.compare_g2_fresh_ab import compare_checkpoints
from scripts.train.g2_evaluate_shared import configure_determinism, evaluate_checkpoint
from scripts.train.g2_spatial import MANIFEST, sha256_file

ROOT = Path('/workspace/stable-worldmodel')
CACHE = ROOT / 'results/stage3/g2_2x2/shared_eval_cache'
NORMALIZERS = ROOT / 'results/stage3/g2_fresh_ab/action_normalizers.json'
ARM_DIRS = {
    'A': 'A_fresh_uniform_mse',
    'B': 'B_fresh_temporal_mse',
}


def checkpoint_path(output: Path, seed: int, arm: str, step: int) -> Path:
    return output / f'seed_{seed}' / ARM_DIRS[arm] / f'checkpoint_step{step}.pt'


def evaluation_path(output: Path, seed: int, arm: str, step: int) -> Path:
    return output / f'seed_{seed}' / ARM_DIRS[arm] / f'evaluation_development_step{step}.json'


def verify_existing_evaluation(path: Path, checkpoint: Path, step: int) -> None:
    report = json.loads(path.read_text())
    if int(report['step']) != step:
        raise ValueError(f'existing evaluation has wrong step: {path}')
    if report['checkpoint_sha256'] != sha256_file(checkpoint):
        raise ValueError(f'existing evaluation checkpoint mismatch: {path}')


def verify_existing_bootstrap(path: Path, checkpoint_a: Path, checkpoint_b: Path) -> None:
    report = json.loads(path.read_text())
    if report['checkpoint_a_sha256'] != sha256_file(checkpoint_a):
        raise ValueError(f'existing bootstrap A mismatch: {path}')
    if report['checkpoint_b_sha256'] != sha256_file(checkpoint_b):
        raise ValueError(f'existing bootstrap B mismatch: {path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=(3073, 3074, 3075))
    parser.add_argument('--steps', type=int, nargs='+', default=(500, 1000, 1500, 2000))
    parser.add_argument('--eval-batch', type=int, default=32)
    parser.add_argument('--evaluation-seed', type=int, default=3072)
    parser.add_argument('--bootstrap-seed', type=int, default=43072)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    if (CACHE / 'STATUS').read_text().strip() != 'READY':
        raise RuntimeError(f'evaluation cache is not READY: {CACHE}')
    if not NORMALIZERS.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError('normalizer or development cohort manifest is missing')
    checkpoints = {
        (seed, arm, step): checkpoint_path(output, seed, arm, step)
        for seed in args.seeds
        for arm in ARM_DIRS
        for step in args.steps
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing checkpoints: {missing}')

    status_path = output / 'EVALUATION_STATUS'
    config_path = output / 'evaluation_launcher_config.json'
    if status_path.exists() and status_path.read_text().strip() == 'DONE':
        raise FileExistsError('evaluation is already DONE; refusing to overwrite')
    report = {
        'seeds': list(args.seeds),
        'steps': list(args.steps),
        'development_cohort_only': True,
        'cache': str(CACHE),
        'cache_status': 'READY',
        'manifest': str(MANIFEST),
        'normalizers': str(NORMALIZERS),
        'evaluation_seed': args.evaluation_seed,
        'bootstrap_seed': args.bootstrap_seed,
        'bootstrap_draws': args.bootstrap_draws,
        'jobs': [],
    }
    status_path.write_text('RUNNING\n')
    config_path.write_text(json.dumps(report, indent=2) + '\n')
    configure_determinism(args.evaluation_seed)
    try:
        for seed in args.seeds:
            for step in args.steps:
                for arm in ARM_DIRS:
                    checkpoint = checkpoints[(seed, arm, step)]
                    destination = evaluation_path(output, seed, arm, step)
                    began = time.perf_counter()
                    if destination.exists():
                        verify_existing_evaluation(destination, checkpoint, step)
                        disposition = 'verified_existing'
                    else:
                        evaluate_checkpoint(
                            checkpoint,
                            CACHE,
                            destination,
                            args.evaluation_seed,
                            args.eval_batch,
                            NORMALIZERS,
                            MANIFEST,
                        )
                        disposition = 'created'
                    report['jobs'].append({
                        'kind': 'evaluation',
                        'seed': seed,
                        'step': step,
                        'arm': arm,
                        'output': str(destination),
                        'disposition': disposition,
                        'seconds': time.perf_counter() - began,
                    })
                    config_path.write_text(json.dumps(report, indent=2) + '\n')
                    gc.collect()
                    torch.cuda.empty_cache()

                checkpoint_a = checkpoints[(seed, 'A', step)]
                checkpoint_b = checkpoints[(seed, 'B', step)]
                destination = output / f'seed_{seed}' / f'development_paired_bootstrap_step{step}.json'
                began = time.perf_counter()
                if destination.exists():
                    verify_existing_bootstrap(destination, checkpoint_a, checkpoint_b)
                    disposition = 'verified_existing'
                else:
                    compare_checkpoints(
                        CACHE,
                        MANIFEST,
                        NORMALIZERS,
                        checkpoint_a,
                        checkpoint_b,
                        destination,
                        args.bootstrap_seed,
                        args.bootstrap_draws,
                        args.eval_batch,
                    )
                    disposition = 'created'
                report['jobs'].append({
                    'kind': 'paired_bootstrap',
                    'seed': seed,
                    'step': step,
                    'output': str(destination),
                    'disposition': disposition,
                    'seconds': time.perf_counter() - began,
                })
                config_path.write_text(json.dumps(report, indent=2) + '\n')
                gc.collect()
                torch.cuda.empty_cache()
        status_path.write_text('DONE\n')
    except BaseException as error:
        status_path.write_text(f'FAILED {type(error).__name__}: {error}\n')
        raise


if __name__ == '__main__':
    main()
