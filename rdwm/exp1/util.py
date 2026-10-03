"""Small process helpers shared by training, eval, and the launcher."""

from __future__ import annotations

import contextlib
import fcntl
import json
import subprocess
from pathlib import Path

from rdwm.exp1.config import REPO_ROOT


@contextlib.contextmanager
def file_lock(path: Path):
    """Exclusive advisory lock so concurrent jobs build shared artifacts once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def git_state() -> dict:
    def run(*args):
        return subprocess.run(
            ['git', *args], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        'commit': run('rev-parse', 'HEAD'),
        'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(run('status', '--porcelain', '--', 'rdwm', 'scripts/rdwm')),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)
