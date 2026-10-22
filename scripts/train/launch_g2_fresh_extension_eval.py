"""Evaluate G2 Fresh A/B at steps 1,000, 1,500, and 2,000."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
RUN = ROOT / 'results/stage3/g2_fresh_ab_extension_2000'
SOURCE = ROOT / 'results/stage3/g2_fresh_ab'
CACHE = ROOT / 'results/stage3/g2_2x2/shared_eval_cache'
MANIFEST = ROOT / 'results/stage3/g2_c1_followup/manifest.json'
PYTHON = '/venv/stable-wm/bin/python'


def run(command: list[str], log: Path) -> tuple[int, float]:
    began = time.perf_counter()
    with log.open('xb') as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for chunk in iter(process.stdout.readline, b''):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            handle.write(chunk)
            handle.flush()
        return process.wait(), time.perf_counter() - began


def main():
    status = RUN / 'EVALUATION_STATUS'
    if status.exists():
        raise FileExistsError(f'refusing existing evaluation status {status}')
    status.write_text('RUNNING\n')
    records = []
    for step in (1000, 1500, 2000):
        paths = {}
        for arm, directory in (
            ('A', RUN / 'A_fresh_uniform_mse_extension'),
            ('B', RUN / 'B_fresh_temporal_mse_extension'),
        ):
            checkpoint = directory / f'checkpoint_step{step}.pt'
            output = directory / f'evaluation_development_step{step}.json'
            command = [
                PYTHON, str(ROOT / 'scripts/train/g2_evaluate_shared.py'),
                '--cache-dir', str(CACHE),
                '--checkpoint', str(checkpoint),
                '--output', str(output),
                '--action-normalizers', str(SOURCE / 'action_normalizers.json'),
                '--cohort-manifest', str(MANIFEST),
            ]
            returncode, seconds = run(
                command, directory / f'evaluation_development_step{step}.log'
            )
            records.append({'step': step, 'arm': arm, 'kind': 'evaluation', 'returncode': returncode, 'seconds': seconds})
            if returncode:
                status.write_text(f'FAILED step={step} arm={arm} returncode={returncode}\n')
                raise SystemExit(returncode)
            paths[arm] = checkpoint
        comparison = RUN / f'development_paired_bootstrap_step{step}.json'
        command = [
            PYTHON, str(ROOT / 'scripts/train/compare_g2_fresh_ab.py'),
            '--cache-dir', str(CACHE),
            '--cohort-manifest', str(MANIFEST),
            '--action-normalizers', str(SOURCE / 'action_normalizers.json'),
            '--checkpoint-a', str(paths['A']),
            '--checkpoint-b', str(paths['B']),
            '--output', str(comparison),
        ]
        returncode, seconds = run(command, RUN / f'development_paired_bootstrap_step{step}.log')
        records.append({'step': step, 'kind': 'paired_bootstrap', 'returncode': returncode, 'seconds': seconds})
        if returncode:
            status.write_text(f'FAILED step={step} bootstrap returncode={returncode}\n')
            raise SystemExit(returncode)
    (RUN / 'development_evaluation_runs.json').write_text(json.dumps(records, indent=2) + '\n')
    status.write_text('DONE\n')


if __name__ == '__main__':
    main()
