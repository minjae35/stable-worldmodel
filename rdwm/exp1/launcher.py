"""GPU job queue for Experiment 1.

Jobs sit in one priority-ordered queue. One worker thread per GPU pulls the
next pending job when its GPU is free and runs it as a subprocess with
``CUDA_VISIBLE_DEVICES`` set to that GPU. GPU ids come from the launcher
config or the caller's ``CUDA_VISIBLE_DEVICES``; nothing is hardcoded.

The full matrix queues B/C/D (70k each) ahead of the seven A specialists
(10k each), so GPUs that finish a generalist pick up remaining specialists.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdwm.exp1.config import REPO_ROOT, env_order, env_tag, load_config, run_dir

TRAIN_SCRIPT = REPO_ROOT / 'scripts' / 'rdwm' / 'exp1_train.py'


@dataclass
class Job:
    variant: str
    seed: int
    envs: list[str]
    updates: int
    stage: str
    extra_args: list[str] = field(default_factory=list)

    def job_id(self, cfg: dict) -> str:
        return f'{self.stage}/{self.variant}/seed{self.seed}/{env_tag(cfg, self.envs)}'


def resolve_gpus(spec) -> list[str]:
    """``auto`` -> CUDA_VISIBLE_DEVICES of the launcher, else every GPU."""
    if spec in (None, 'auto'):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible:
            return [g.strip() for g in visible.split(',') if g.strip()]
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    return [str(g) for g in spec]


def expand_jobs(cfg: dict, lcfg: dict) -> list[Job]:
    """Explicit ``jobs`` list, or the full ``matrix`` in priority order."""
    stage = lcfg['stage']
    extra = list(lcfg.get('extra_args', []))
    if 'jobs' in lcfg:
        return [
            Job(j['variant'], int(j['seed']), list(j['envs']), int(j['updates']),
                stage, extra + list(j.get('extra_args', [])))
            for j in lcfg['jobs']
        ]
    matrix = lcfg['matrix']
    budgets = cfg['train']['budgets']
    envs = matrix.get('envs') or env_order(cfg)
    jobs = []
    for seed in matrix['seeds']:
        for variant in matrix.get('generalists', ['B', 'C', 'D']):
            jobs.append(Job(variant, seed, list(envs), budgets[variant], stage, extra))
        if matrix.get('specialists', True):
            for env in envs:
                jobs.append(Job('A', seed, [env], budgets['A'], stage, extra))
    return jobs


def train_command(job: Job, config_path: str) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        '--config', config_path,
        '--stage', job.stage,
        '--variant', job.variant,
        '--seed', str(job.seed),
        '--envs', ','.join(job.envs),
        '--updates', str(job.updates),
        *job.extra_args,
    ]


class Launcher:
    def __init__(self, cfg: dict, config_path: str, jobs: list[Job], gpus: list[str],
                 log_dir: Path, make_command=None, poll: float = 1.0):
        self.cfg = cfg
        self.config_path = config_path
        self.jobs = jobs
        self.gpus = gpus
        self.log_dir = Path(log_dir)
        self.make_command = make_command or (lambda job: train_command(job, config_path))
        self.status: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.poll = poll

    def is_done(self, job: Job) -> bool:
        return (run_dir(self.cfg, job.stage, job.variant, job.seed, job.envs)
                / 'done.json').exists()

    def _write_status(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.log_dir / 'status.tmp'
        tmp.write_text(json.dumps(self.status, indent=2))
        tmp.replace(self.log_dir / 'status.json')

    def _set(self, job: Job, **fields) -> None:
        with self.lock:
            self.status.setdefault(job.job_id(self.cfg), {'job': asdict(job)}).update(fields)
            self._write_status()

    def _worker(self, gpu: str, pending: queue.Queue) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            job_id = job.job_id(self.cfg)
            log_path = self.log_dir / (job_id.replace('/', '__') + '.log')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
            cmd = self.make_command(job)
            self._set(job, state='running', gpu=gpu, started=time.time(),
                      log=str(log_path), cmd=cmd)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a') as log:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      cwd=REPO_ROOT).returncode
            self._set(job, state='done' if code == 0 else 'failed',
                      returncode=code, finished=time.time())

    def run(self) -> dict[str, dict]:
        pending: queue.Queue = queue.Queue()
        for job in self.jobs:
            if self.is_done(job):
                self._set(job, state='skipped_done')
            else:
                self._set(job, state='pending')
                pending.put(job)
        threads = [threading.Thread(target=self._worker, args=(gpu, pending), daemon=True)
                   for gpu in self.gpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return self.status


def load_launcher_config(path: str | Path) -> dict:
    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


def plan(lcfg_path: str | Path) -> tuple[dict, dict, list[Job], list[str]]:
    lcfg = load_launcher_config(lcfg_path)
    cfg = load_config(REPO_ROOT / lcfg['config'])
    return lcfg, cfg, expand_jobs(cfg, lcfg), resolve_gpus(lcfg.get('gpus'))
