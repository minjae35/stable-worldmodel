"""Canonical Design v0 dataset constants and path policy.

Collector scripts are not modified. TwoRoom / OGBCube collection still
uses ``cfg.seed = 3072``, which the existing collectors turn into
``World.collect(seed=724895)``. Provenance records both values.
"""

from __future__ import annotations

import os
from pathlib import Path

from stable_worldmodel.utils import DEFAULT_CACHE_DIR

SCHEMA_VERSION = 1
CONFIG_SEED = 3072
COLLECT_SEED = 724895
SPLIT_SEED = 3072
TRAIN_FRACTION = 0.9

DRYRUN_EPISODES = 3
DRYRUN_NUM_ENVS = 1

CANONICAL_RELATIVE_ROOT = Path('datasets') / 'design_v0'

TWOROOM_CANONICAL_NAME = 'tworoom_expert.lance'
PUSHT_CANONICAL_NAME = 'pusht_expert_train.h5'
OGBCUBE_RAW_RELATIVE = Path('ogbcube') / 'cube_single_multiview_expert.lance'
OGBCUBE_CANONICAL_RELATIVE = Path('ogbcube') / 'cube_single_front_expert.lance'

COLLECTOR_TWOROOM_RELATIVE = Path('datasets') / 'tworoom_expert.lance'
COLLECTOR_OGBCUBE_RELATIVE = (
    Path('datasets') / 'ogbench' / 'cube_single_multiview_expert.lance'
)

PUSHT_HF_REPO = 'quentinll/lewm-pusht'
PUSHT_HF_REVISION = '655cd446b9929369d7d406001da85c15d1457850'
PUSHT_HF_FILENAME = 'pusht_expert_train.h5.zst'
PUSHT_LFS_SHA256 = (
    '7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318'
)
PUSHT_ZST_SIZE_BYTES = 13_136_247_974
PUSHT_EXPECTED_EPISODES = 2000
PUSHT_EXPECTED_TRANSITIONS = 297_806

TWOROOM_ENV_ID = 'swm/TwoRoom-v1'
OGBCUBE_ENV_ID = 'swm/OGBCube-v0'
TWOROOM_MAX_EPISODE_STEPS = 100
OGBCUBE_MAX_EPISODE_STEPS = 200
IMAGE_SHAPE = (224, 224)


def cache_root(cache_dir: str | Path | None = None) -> Path:
    """Return the SWM cache root without creating directories."""
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.getenv('STABLEWM_HOME', DEFAULT_CACHE_DIR))


def canonical_root(cache_dir: str | Path | None = None) -> Path:
    return cache_root(cache_dir) / CANONICAL_RELATIVE_ROOT


def canonical_artifacts(
    cache_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = canonical_root(cache_dir)
    return {
        'tworoom': root / TWOROOM_CANONICAL_NAME,
        'pusht': root / PUSHT_CANONICAL_NAME,
        'ogbcube_raw': root / OGBCUBE_RAW_RELATIVE,
        'ogbcube': root / OGBCUBE_CANONICAL_RELATIVE,
        'manifests': root / 'manifests',
        'splits': root / 'splits',
    }


def collector_default_artifacts(
    cache_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    root = cache_root(cache_dir)
    return (
        root / COLLECTOR_TWOROOM_RELATIVE,
        root / COLLECTOR_OGBCUBE_RELATIVE,
    )


def _is_inside(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def assert_destination_absent(path: str | Path) -> Path:
    dest = Path(path)
    if dest.exists():
        raise FileExistsError(
            f'Destination already exists (refusing append/overwrite): {dest}'
        )
    return dest


def assert_isolated_dryrun_dest(
    dest: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """Reject dest paths that would pollute canonical or collector data."""
    dest = Path(dest).resolve()
    canon = canonical_root(cache_dir)
    if _is_inside(dest, canon):
        raise ValueError(
            f'dry-run dest {dest} is inside canonical root {canon.resolve()}'
        )
    for forbidden in (
        *collector_default_artifacts(cache_dir),
        *canonical_artifacts(cache_dir).values(),
    ):
        if dest == Path(forbidden).resolve():
            raise ValueError(
                f'dry-run dest {dest} collides with protected path {forbidden}'
            )
    return dest
