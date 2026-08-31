"""Isolated 3-episode schema dry-run for TwoRoom and OGBCube collectors.

Writes under a temporary directory by default. Canonical and collector
default artifact paths are rejected so LanceWriter append cannot pollute
them. This script never runs a 10,000-episode collection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.data.design_v0.spec import (  # noqa: E402
    COLLECT_SEED,
    CONFIG_SEED,
    DRYRUN_EPISODES,
    DRYRUN_NUM_ENVS,
    IMAGE_SHAPE,
    OGBCUBE_ENV_ID,
    OGBCUBE_MAX_EPISODE_STEPS,
    TWOROOM_ENV_ID,
    TWOROOM_MAX_EPISODE_STEPS,
    assert_destination_absent,
    assert_isolated_dryrun_dest,
)

ENV_CHOICES = ('tworoom', 'ogbcube', 'both')


def dryrun_dest(
    output_root: str | Path,
    env_name: str,
    cache_dir: str | Path | None = None,
) -> Path:
    names = {
        'tworoom': 'tworoom_expert.lance',
        'ogbcube': 'ogbcube_multiview_expert.lance',
    }
    dest = Path(output_root) / names[env_name]
    assert_isolated_dryrun_dest(dest, cache_dir=cache_dir)
    return assert_destination_absent(dest)


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    import numpy as np

    from stable_worldmodel.data import load_dataset

    dataset = load_dataset(str(path))
    columns = list(dataset.column_names)
    image_columns = [
        name
        for name in columns
        if name == 'pixels' or 'pixel' in name.lower()
    ]
    action_shape: list[int] | None = None
    if 'action' in columns and len(dataset.lengths):
        action = dataset.load_episode(0)['action']
        if hasattr(action, 'detach'):
            action = action.detach().cpu().numpy()
        action_shape = [int(dim) for dim in np.asarray(action).shape]
    return {
        'path': str(path),
        'num_episodes': int(len(dataset.lengths)),
        'num_transitions': int(dataset.lengths.sum()),
        'episode_lengths': [int(x) for x in dataset.lengths],
        'columns': columns,
        'image_columns': image_columns,
        'action_shape': action_shape,
        'config_seed': CONFIG_SEED,
        'collect_seed': COLLECT_SEED,
    }


def _collect_tworoom(dest: Path) -> None:
    import stable_worldmodel as swm
    from stable_worldmodel.envs.two_room import ExpertPolicy

    dest.parent.mkdir(parents=True, exist_ok=True)
    world = swm.World(
        TWOROOM_ENV_ID,
        num_envs=DRYRUN_NUM_ENVS,
        max_episode_steps=TWOROOM_MAX_EPISODE_STEPS,
        image_shape=IMAGE_SHAPE,
        render_mode='rgb_array',
    )
    world.set_policy(
        ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)
    )
    world.collect(
        dest,
        episodes=DRYRUN_EPISODES,
        seed=COLLECT_SEED,
        format='lance',
    )


def _collect_ogbcube(dest: Path) -> None:
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    import stable_worldmodel as swm
    from stable_worldmodel.envs.ogbench import ExpertPolicy

    dest.parent.mkdir(parents=True, exist_ok=True)
    world = swm.World(
        OGBCUBE_ENV_ID,
        num_envs=DRYRUN_NUM_ENVS,
        max_episode_steps=OGBCUBE_MAX_EPISODE_STEPS,
        image_shape=IMAGE_SHAPE,
        env_type='single',
        multiview=True,
        width=IMAGE_SHAPE[0],
        height=IMAGE_SHAPE[1],
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
    )
    world.set_policy(ExpertPolicy())
    world.collect(
        dest,
        episodes=DRYRUN_EPISODES,
        seed=COLLECT_SEED,
        format='lance',
    )


_COLLECTORS = {
    'tworoom': _collect_tworoom,
    'ogbcube': _collect_ogbcube,
}


def run_dryrun(
    env_name: str,
    output_root: str | Path,
    *,
    cache_dir: str | Path | None = None,
    collect: bool = True,
) -> dict[str, Any]:
    if env_name not in _COLLECTORS:
        raise ValueError(f'unknown env {env_name!r}')
    dest = dryrun_dest(output_root, env_name, cache_dir=cache_dir)
    report = {
        'environment': env_name,
        'dest': str(dest),
        'episodes': DRYRUN_EPISODES,
        'num_envs': DRYRUN_NUM_ENVS,
        'config_seed': CONFIG_SEED,
        'collect_seed': COLLECT_SEED,
    }
    if collect:
        _COLLECTORS[env_name](dest)
        report['inspect'] = inspect_dataset(dest)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--env',
        choices=ENV_CHOICES,
        default='both',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Isolated directory. Defaults to a new temp directory.',
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Cache root used only to detect protected artifact paths.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    output_root = args.output_root
    if output_root is None:
        output_root = Path(
            tempfile.mkdtemp(prefix='design_v0_dryrun_')
        )
    envs = ('tworoom', 'ogbcube') if args.env == 'both' else (args.env,)
    reports = [
        run_dryrun(name, output_root, cache_dir=args.cache_dir)
        for name in envs
    ]
    print(json.dumps(reports, indent=2))
    return reports


if __name__ == '__main__':
    main()
