"""Launch the single preregistered G4 H3 + fixed temporal encoding arm."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
G3_ROOT = ROOT / 'results/stage3/g3_history'
NORMALIZERS = ROOT / 'results/stage3/g2_fresh_ab/action_normalizers.json'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--max-train-items', type=int, default=0)
    args = parser.parse_args()
    root = args.output_root.resolve()
    run = root / 'H3_sinusoidal'
    if run.exists():
        raise FileExistsError(f'refusing to overwrite {run}')
    root.mkdir(parents=True, exist_ok=True)
    common = G3_ROOT / 'common_windows.json'
    g3_repro = json.loads((G3_ROOT / 'H3_uniform_mse/reproducibility.json').read_text())
    preregistration = {
        'experiment': 'g4_h3_fixed_sinusoidal_temporal_encoding',
        'baseline': str(G3_ROOT / 'H3_uniform_mse'),
        'history': 3,
        'temporal_encoding': 'fixed_sinusoidal',
        'positions': [0, 1, 2],
        'scale': 1.0,
        'scale_selection': (
            'fixed before training; canonical amplitude 1.0. Fresh projected-token RMS '
            '2.5565 and encoding RMS 0.7071, so encoding RMS is 27.7% of input RMS.'
        ),
        'application': 'after spatial/action/environment embedding, before transformer blocks',
        'trainable_parameters_added': 0,
        'seed': 3072,
        'updates': args.steps,
        'batch_per_environment': 16,
        'optimizer': {'name': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-4},
        'gradient_clip': 1.0,
        'sampling': 'same G3 common uniform schedules',
        'expected_initial_model_sha256': g3_repro['initial_model_sha256'],
        'expected_schedule_sha256_by_environment': g3_repro['schedule_sha256_by_environment'],
        'development_only': True,
        'final_holdout_used': False,
    }
    prereg_path = root / 'temporal_encoding_preregistration.json'
    if prereg_path.exists():
        if json.loads(prereg_path.read_text()) != preregistration:
            raise ValueError('existing preregistration differs')
    else:
        prereg_path.write_text(json.dumps(preregistration, indent=2) + '\n')
    command = [
        '/venv/stable-wm/bin/python', str(ROOT / 'scripts/train/g3_history_spatial.py'),
        '--arm', 'H3', '--output-dir', str(run),
        '--common-manifest', str(common), '--normalizer-file', str(NORMALIZERS),
        '--steps', str(args.steps), '--checkpoint-every', str(args.checkpoint_every),
        '--temporal-encoding', 'sinusoidal', '--temporal-encoding-scale', '1.0',
        '--experiment-name', 'g4_h3_fixed_sinusoidal_temporal_encoding',
    ]
    if args.max_train_items:
        command.extend(['--max-train-items', str(args.max_train_items)])
    launcher = {'command': command, 'status': 'RUNNING'}
    launcher_path = root / 'temporal_launcher.json'
    launcher_path.write_text(json.dumps(launcher, indent=2) + '\n')
    began = time.perf_counter()
    log_path = root / 'H3_sinusoidal_training.log'
    with log_path.open('xb') as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for chunk in iter(process.stdout.readline, b''):
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            handle.write(chunk); handle.flush()
        returncode = process.wait()
    launcher.update({
        'status': 'DONE' if returncode == 0 else 'FAILED',
        'returncode': returncode,
        'seconds': time.perf_counter() - began,
        'log': str(log_path),
    })
    launcher_path.write_text(json.dumps(launcher, indent=2) + '\n')
    if returncode:
        raise SystemExit(returncode)
    observed = json.loads((run / 'reproducibility.json').read_text())
    full_schedule_check = args.steps == 2000 and args.max_train_items == 0
    checks = {
        'same_initial_model': observed['initial_model_sha256'] == preregistration['expected_initial_model_sha256'],
        'same_schedules': (
            observed['schedule_sha256_by_environment']
            == preregistration['expected_schedule_sha256_by_environment']
        ) if full_schedule_check else True,
        'schedule_hash_check_scope': 'required_full_run' if full_schedule_check else 'not_applicable_reduced_smoke',
        'same_parameter_count': observed['model_parameter_count'] == g3_repro['model_parameter_count'],
    }
    checks['status'] = 'PASS' if all(
        value for key, value in checks.items() if key != 'schedule_hash_check_scope'
    ) else 'FAIL'
    (root / 'temporal_fairness_validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    if checks['status'] != 'PASS':
        raise RuntimeError(f'G4 fairness validation failed: {checks}')


if __name__ == '__main__':
    main()
