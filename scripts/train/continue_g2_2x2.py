"""Safely continue the Stage-3 G2 pilot after completed Arm A training.

This launcher intentionally accepts only the absent B/C/D run directories.  It
never resumes or reruns Arm A and delegates each new run to ``g2_spatial.py``
with evaluation deferred to the shared-cache evaluator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.launch_g2_2x2 import (  # noqa: E402
    RUN_NAMES,
    build_commands,
    run_command_tee,
)

DEFAULT_OUTPUT = ROOT / 'results/stage3/g2_2x2'
REMAINING_ARMS = ('B', 'C', 'D')


def verify_continuation(output_root: Path) -> None:
    if not output_root.is_dir():
        raise FileNotFoundError(f'missing existing output root {output_root}')
    arm_a = output_root / RUN_NAMES['A']
    required = (
        arm_a / 'checkpoint_step500.pt',
        arm_a / 'metrics.csv',
        arm_a / 'evaluation_optimized.json',
        output_root / 'gradient_calibration.json',
        output_root / 'shared_eval_cache' / 'metadata.json',
        output_root / 'shared_eval_cache' / 'probes.json',
        output_root / 'shared_eval_cache' / 'probes.pt',
        output_root / 'shared_eval_cache' / 'STATUS',
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'continuation prerequisites are missing: {missing}')
    if (output_root / 'shared_eval_cache' / 'STATUS').read_text().strip() != 'READY':
        raise RuntimeError('shared evaluation cache is not READY')
    for arm in REMAINING_ARMS:
        run_dir = output_root / RUN_NAMES[arm]
        log_path = output_root / f'{RUN_NAMES[arm]}.log'
        if run_dir.exists() or log_path.exists():
            raise FileExistsError(f'refusing existing output for Arm {arm}')


def write_document(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + '\n')


def run_continuation(output_root: Path, commands: list[list[str]]) -> int:
    record_path = output_root / 'continuation_bcd.json'
    if record_path.exists():
        raise FileExistsError(f'refusing to overwrite {record_path}')
    status = output_root / 'STATUS'
    document = {
        'session_name': 'stablewm_stage3_g2',
        'sequential': True,
        'arms': list(REMAINING_ARMS),
        'commands': commands,
        'runs': [],
    }
    write_document(record_path, document)
    status.write_text('TRAINING_B_C_D_RUNNING\n')
    for arm, command in zip(REMAINING_ARMS, commands):
        name = RUN_NAMES[arm]
        log_path = output_root / f'{name}.log'
        began = time.time()
        print(f'G2_CONTINUE arm={arm} log={log_path}', flush=True)
        returncode = run_command_tee(command, log_path)
        run_dir = output_root / name
        run_log = run_dir / 'train.log'
        if run_dir.is_dir() and not run_log.exists():
            shutil.copy2(log_path, run_log)
        row = {
            'arm': arm,
            'name': name,
            'returncode': returncode,
            'seconds': time.time() - began,
            'log': str(log_path),
            'run_log': str(run_log) if run_log.is_file() else None,
        }
        document['runs'].append(row)
        write_document(record_path, document)
        if returncode != 0:
            status.write_text(f'FAILED arm={arm} returncode={returncode}\n')
            return returncode
    status.write_text('TRAINING_DONE_B_C_D_EVALUATION_PENDING\n')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    verify_continuation(output_root)
    all_commands = build_commands(
        sys.executable, output_root, smoke=False, defer_evaluation=True
    )
    commands = all_commands[1:]
    if args.dry_run:
        print(json.dumps({'commands': commands}, indent=2))
        return 0
    return run_continuation(output_root, commands)


if __name__ == '__main__':
    raise SystemExit(main())
