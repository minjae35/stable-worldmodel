"""Regression tests for collect_cube.py MUJOCO_GL handling."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

COLLECT_CUBE = (
    Path(__file__).resolve().parents[2]
    / 'scripts'
    / 'data'
    / 'collect_cube.py'
)

_IMPORT_SNIPPET = """
import importlib.util
import os
import sys
print('BEFORE', os.environ.get('MUJOCO_GL'))
spec = importlib.util.spec_from_file_location(
    'collect_cube',
    sys.argv[1],
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('AFTER', os.environ.get('MUJOCO_GL'))
"""


def _mujoco_gl_after_import(
    *,
    mujoco_gl: str | None,
) -> tuple[str | None, str | None]:
    env = os.environ.copy()
    if mujoco_gl is None:
        env.pop('MUJOCO_GL', None)
    else:
        env['MUJOCO_GL'] = mujoco_gl
    result = subprocess.run(
        [sys.executable, '-c', _IMPORT_SNIPPET, str(COLLECT_CUBE)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(COLLECT_CUBE.parents[2]),
    )
    before = after = None
    for line in result.stdout.splitlines():
        if line.startswith('BEFORE '):
            value = line.split(' ', 1)[1]
            before = None if value == 'None' else value
        elif line.startswith('AFTER '):
            value = line.split(' ', 1)[1]
            after = None if value == 'None' else value
    return before, after


def test_collect_cube_uses_setdefault_not_overwrite():
    source = COLLECT_CUBE.read_text()
    assert "os.environ.setdefault('MUJOCO_GL', 'glfw')" in source
    assert "os.environ['MUJOCO_GL'] = 'glfw'" not in source


def test_collect_cube_defaults_mujoco_gl_to_glfw_when_unset():
    before, after = _mujoco_gl_after_import(mujoco_gl=None)
    assert before is None
    assert after == 'glfw'


def test_collect_cube_preserves_mujoco_gl_egl():
    before, after = _mujoco_gl_after_import(mujoco_gl='egl')
    assert before == 'egl'
    assert after == 'egl'
