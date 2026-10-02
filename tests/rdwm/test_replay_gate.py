"""Phase 1 replay gate.

This hits the Hub and the live simulators. It is the same check as
``scripts/rdwm/replay_validate.py``.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void':
    del os.environ['NVIDIA_VISIBLE_DEVICES']

from rdwm.data.manifest import ENVS
from rdwm.replay import run_env


@pytest.mark.parametrize('name', list(ENVS))
def test_replay_gate(name):
    result = run_env(name)
    assert result['status'] == 'PASS', result.get('error') or result
