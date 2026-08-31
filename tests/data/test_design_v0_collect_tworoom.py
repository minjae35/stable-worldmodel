"""Fixture tests for the Design v0 TwoRoom canonical collection wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.design_v0.collect_tworoom import (
    COLLECTOR_SCRIPT,
    collect_tworoom,
    collector_command,
    staging_artifact_path,
    validate_tworoom_artifact,
    verify_seeds,
)
from scripts.data.design_v0.manifest import (
    episode_split,
    indices_sha256,
    sha256_artifact,
)
from scripts.data.design_v0.spec import (
    COLLECT_SEED,
    COLLECTOR_TWOROOM_RELATIVE,
    CONFIG_SEED,
    TWOROOM_ACTION_DIM,
    TWOROOM_EXPECTED_EPISODES,
    canonical_artifacts,
    collector_default_artifacts,
)


N_EPISODES = 10
STEPS = 3


def _write_tworoom_lance(
    path: Path,
    *,
    num_episodes: int = N_EPISODES,
    action_dim: int = TWOROOM_ACTION_DIM,
    include_pixels: bool = True,
) -> None:
    pytest.importorskip('lancedb')
    from stable_worldmodel.data import LanceWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    episode: dict[str, list] = {
        'action': [
            np.arange(action_dim, dtype=np.float32) for _ in range(STEPS)
        ],
    }
    if include_pixels:
        episode['pixels'] = [
            np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(STEPS)
        ]
    with LanceWriter(path) as writer:
        for _ in range(num_episodes):
            writer.write_episode(
                {
                    key: [item.copy() for item in values]
                    for key, values in episode.items()
                }
            )


def _ok_collector(staging: Path, **kwargs) -> None:
    _write_tworoom_lance(staging_artifact_path(staging), **kwargs)


def test_existing_canonical_dest_fails_fast(tmp_path):
    dest = canonical_artifacts(tmp_path)['tworoom']
    dest.parent.mkdir(parents=True)
    dest.mkdir()
    (dest / 'marker.txt').write_text('existing')

    def boom(staging: Path) -> None:
        raise AssertionError(f'collector should not run: {staging}')

    with pytest.raises(FileExistsError, match='already exists'):
        collect_tworoom(
            cache_dir=tmp_path,
            collector=boom,
            expected_episodes=N_EPISODES,
        )
    assert (dest / 'marker.txt').read_text() == 'existing'


def test_staging_is_isolated_from_canonical_and_collector_paths(tmp_path):
    seen: dict[str, Path] = {}

    def collector(staging: Path) -> None:
        seen['staging'] = Path(staging)
        seen['artifact'] = staging_artifact_path(staging)
        artifacts = canonical_artifacts(tmp_path)
        baseline = collector_default_artifacts(tmp_path)[0]
        assert staging.resolve() != artifacts['tworoom'].resolve()
        assert seen['artifact'].resolve() != artifacts['tworoom'].resolve()
        assert seen['artifact'].resolve() != Path(baseline).resolve()
        _ok_collector(staging)

    report = collect_tworoom(
        cache_dir=tmp_path,
        collector=collector,
        expected_episodes=N_EPISODES,
    )
    dest = Path(report['dest'])
    assert dest == canonical_artifacts(tmp_path)['tworoom']
    assert dest.exists()
    assert seen['staging'].name.startswith('tworoom_staging_')
    assert seen['staging'].parent == dest.parent
    assert not seen['staging'].exists()
    assert not seen['artifact'].exists()
    assert not (tmp_path / COLLECTOR_TWOROOM_RELATIVE).exists()


def test_collector_failure_leaves_canonical_dest_absent(tmp_path):
    dest = canonical_artifacts(tmp_path)['tworoom']

    def boom(staging: Path) -> None:
        raise RuntimeError('collector exploded')

    with pytest.raises(RuntimeError, match='collector exploded'):
        collect_tworoom(
            cache_dir=tmp_path,
            collector=boom,
            expected_episodes=N_EPISODES,
        )
    assert not dest.exists()
    assert not list(dest.parent.glob('tworoom_staging_*'))


def test_validation_failure_does_not_promote(tmp_path):
    dest = canonical_artifacts(tmp_path)['tworoom']

    def bad_count(staging: Path) -> None:
        _ok_collector(staging, num_episodes=N_EPISODES - 1)

    with pytest.raises(ValueError, match='episode count'):
        collect_tworoom(
            cache_dir=tmp_path,
            collector=bad_count,
            expected_episodes=N_EPISODES,
        )
    assert not dest.exists()

    def missing_pixels(staging: Path) -> None:
        _ok_collector(staging, include_pixels=False)

    with pytest.raises(ValueError, match='missing pixels'):
        collect_tworoom(
            cache_dir=tmp_path,
            collector=missing_pixels,
            expected_episodes=N_EPISODES,
        )
    assert not dest.exists()

    def wrong_action(staging: Path) -> None:
        _ok_collector(staging, action_dim=5)

    with pytest.raises(ValueError, match='action dim'):
        collect_tworoom(
            cache_dir=tmp_path,
            collector=wrong_action,
            expected_episodes=N_EPISODES,
        )
    assert not dest.exists()
    assert not (tmp_path / COLLECTOR_TWOROOM_RELATIVE).exists()


def test_successful_staging_promotes_and_writes_manifests(tmp_path):
    report = collect_tworoom(
        cache_dir=tmp_path,
        collector=_ok_collector,
        expected_episodes=N_EPISODES,
    )
    dest = Path(report['dest'])
    artifacts = canonical_artifacts(tmp_path)
    assert dest.exists()
    assert dest == artifacts['tworoom']
    assert report['config_seed'] == CONFIG_SEED
    assert report['collect_seed'] == COLLECT_SEED
    assert report['inspect']['num_episodes'] == N_EPISODES
    assert report['inspect']['action_dim'] == TWOROOM_ACTION_DIM
    assert 'pixels' in report['inspect']['columns']
    assert report['train_episodes'] == 9
    assert report['val_episodes'] == 1

    inspect = validate_tworoom_artifact(
        dest,
        expected_episodes=N_EPISODES,
    )
    assert inspect['num_transitions'] == N_EPISODES * STEPS
    assert sha256_artifact(dest) == report['artifact_sha256']

    provenance = json.loads(
        Path(report['provenance_path']).read_text()
    )
    split = json.loads(Path(report['split_path']).read_text())
    assert provenance['source']['script'] == (
        'scripts/data/collect_tworooms.py'
    )
    assert provenance['config_seed'] == CONFIG_SEED
    assert provenance['collect_seed'] == COLLECT_SEED
    assert split['train_indices_sha256'] == indices_sha256(
        split['train_episode_indices']
    )
    assert split['val_indices_sha256'] == indices_sha256(
        split['val_episode_indices']
    )
    assert not (tmp_path / COLLECTOR_TWOROOM_RELATIVE).exists()
    assert not list(dest.parent.glob('tworoom_staging_*'))


def test_episode_split_10000_is_9000_1000():
    train, val = episode_split(TWOROOM_EXPECTED_EPISODES)
    assert len(train) == 9000
    assert len(val) == 1000
    assert sorted(train + val) == list(range(TWOROOM_EXPECTED_EPISODES))
    assert set(train).isdisjoint(val)


def test_collector_command_uses_baseline_script_and_seeds():
    command = collector_command('/tmp/staging')
    assert command[1] == str(COLLECTOR_SCRIPT)
    assert COLLECTOR_SCRIPT.name == 'collect_tworooms.py'
    assert 'cache_dir=/tmp/staging' in command
    assert f'seed={CONFIG_SEED}' in command
    assert f'num_traj={TWOROOM_EXPECTED_EPISODES}' in command
    verify_seeds()
    with pytest.raises(ValueError, match='config_seed'):
        verify_seeds(config_seed=0)
