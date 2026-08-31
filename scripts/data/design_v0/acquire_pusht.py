"""Pinned HuggingFace acquisition plan for the Design v0 PushT dataset.

Default mode is ``--plan``: print the pinned download/decompress commands
without fetching the 13GB blob. ``--download`` is implemented for later
use and is unit-tested through injectable hooks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.data.design_v0.manifest import sha256_file  # noqa: E402
from scripts.data.design_v0.spec import (  # noqa: E402
    PUSHT_CANONICAL_NAME,
    PUSHT_EXPECTED_EPISODES,
    PUSHT_EXPECTED_TRANSITIONS,
    PUSHT_HF_FILENAME,
    PUSHT_HF_REPO,
    PUSHT_HF_REVISION,
    PUSHT_LFS_SHA256,
    PUSHT_ZST_SIZE_BYTES,
    assert_destination_absent,
    canonical_artifacts,
)

Downloader = Callable[..., str]
Runner = Callable[..., Any]


def hf_download_kwargs() -> dict[str, str]:
    return {
        'repo_id': PUSHT_HF_REPO,
        'filename': PUSHT_HF_FILENAME,
        'repo_type': 'dataset',
        'revision': PUSHT_HF_REVISION,
    }


def hf_download_command(
    dest_dir: str | Path | None = None,
) -> list[str]:
    command = [
        'huggingface-cli',
        'download',
        PUSHT_HF_REPO,
        PUSHT_HF_FILENAME,
        '--repo-type',
        'dataset',
        '--revision',
        PUSHT_HF_REVISION,
    ]
    if dest_dir is not None:
        command.extend(['--local-dir', str(dest_dir)])
    return command


def decompress_command(source: str | Path, dest: str | Path) -> list[str]:
    # No ``-f``: existing HDF5 must fail rather than be overwritten.
    return ['zstd', '-d', '-o', str(dest), str(source)]


def parse_hf_tree_entry(
    tree: list[dict[str, Any]],
    filename: str = PUSHT_HF_FILENAME,
) -> dict[str, Any]:
    for entry in tree:
        if entry.get('path') == filename:
            return entry
    raise FileNotFoundError(
        f'{filename!r} not found in HuggingFace tree for '
        f'{PUSHT_HF_REPO}@{PUSHT_HF_REVISION}'
    )


def lfs_sha256_from_entry(entry: dict[str, Any]) -> str:
    lfs = entry.get('lfs') or {}
    oid = lfs.get('oid')
    if not oid:
        raise ValueError(f'HF tree entry is missing LFS oid: {entry}')
    return str(oid).lower()


def verify_lfs_sha256(
    actual: str,
    expected: str = PUSHT_LFS_SHA256,
) -> str:
    got = actual.lower()
    want = expected.lower()
    if got != want:
        raise ValueError(
            f'PushT LFS SHA-256 mismatch: got {got}, expected {want}'
        )
    return got


def verify_zst_size(
    actual_bytes: int,
    expected_bytes: int = PUSHT_ZST_SIZE_BYTES,
) -> int:
    if int(actual_bytes) != int(expected_bytes):
        raise ValueError(
            f'PushT zst size mismatch: got {actual_bytes}, '
            f'expected {expected_bytes}'
        )
    return int(actual_bytes)


def verify_pusht_counts(
    num_episodes: int,
    num_transitions: int,
    *,
    expected_episodes: int = PUSHT_EXPECTED_EPISODES,
    expected_transitions: int = PUSHT_EXPECTED_TRANSITIONS,
) -> None:
    if int(num_episodes) != int(expected_episodes):
        raise ValueError(
            f'PushT episode count mismatch: got {num_episodes}, '
            f'expected {expected_episodes}'
        )
    if int(num_transitions) != int(expected_transitions):
        raise ValueError(
            f'PushT transition count mismatch: got {num_transitions}, '
            f'expected {expected_transitions}'
        )


def read_h5_episode_stats(path: str | Path) -> tuple[int, int]:
    import h5py
    import numpy as np

    with h5py.File(path, 'r') as handle:
        if 'ep_len' not in handle:
            raise ValueError(f'{path} is missing ep_len')
        lengths = np.asarray(handle['ep_len'][:])
    return int(lengths.size), int(lengths.sum())


def verify_h5_counts(path: str | Path) -> tuple[int, int]:
    num_episodes, num_transitions = read_h5_episode_stats(path)
    verify_pusht_counts(num_episodes, num_transitions)
    return num_episodes, num_transitions


def verify_file_sha256(path: str | Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f'SHA-256 mismatch for {path}: got {actual}, expected {expected}'
        )
    return actual


def build_acquire_plan(
    dest_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = canonical_artifacts(cache_dir)
    dest_h5 = (
        Path(dest_dir) / PUSHT_CANONICAL_NAME
        if dest_dir is not None
        else artifacts['pusht']
    )
    source_zst = dest_h5.with_suffix(dest_h5.suffix + '.zst')
    if dest_dir is not None:
        source_zst = Path(dest_dir) / PUSHT_HF_FILENAME
    return {
        'repo_id': PUSHT_HF_REPO,
        'revision': PUSHT_HF_REVISION,
        'filename': PUSHT_HF_FILENAME,
        'hf_download_kwargs': hf_download_kwargs(),
        'hf_download_command': hf_download_command(
            Path(dest_dir) if dest_dir is not None else dest_h5.parent
        ),
        'expected_lfs_sha256': PUSHT_LFS_SHA256,
        'expected_zst_size_bytes': PUSHT_ZST_SIZE_BYTES,
        'source_zst': str(source_zst),
        'canonical_h5': str(dest_h5),
        'decompress_command': decompress_command(source_zst, dest_h5),
        'expected_episodes': PUSHT_EXPECTED_EPISODES,
        'expected_transitions': PUSHT_EXPECTED_TRANSITIONS,
    }


def decompress_zst(
    source: str | Path,
    dest: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> Path:
    dest_path = assert_destination_absent(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    command = decompress_command(source, dest_path)
    runner(command, check=True)
    return dest_path


def download_pusht_zst(
    dest_zst: str | Path,
    *,
    downloader: Downloader | None = None,
) -> Path:
    dest_path = assert_destination_absent(dest_zst)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        from huggingface_hub import hf_hub_download

        def downloader(**kwargs):
            return hf_hub_download(
                **kwargs,
                local_dir=str(dest_path.parent),
            )

    downloaded = downloader(**hf_download_kwargs())
    return Path(downloaded)


def acquire_pusht(
    *,
    dest_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    download: bool = False,
    downloader: Downloader | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Plan or run pinned PushT acquisition.

    ``download=False`` (default) never touches the network or disk.
    """
    plan = build_acquire_plan(dest_dir=dest_dir, cache_dir=cache_dir)
    if not download:
        plan['executed'] = False
        return plan

    dest_h5 = Path(plan['canonical_h5'])
    source_zst = Path(plan['source_zst'])
    assert_destination_absent(dest_h5)
    downloaded = download_pusht_zst(source_zst, downloader=downloader)
    actual_sha = sha256_file(downloaded)
    verify_lfs_sha256(actual_sha)
    verify_zst_size(downloaded.stat().st_size)
    decompress_zst(downloaded, dest_h5, runner=runner)
    final_sha = sha256_file(dest_h5)
    num_episodes, num_transitions = verify_h5_counts(dest_h5)
    plan['executed'] = True
    plan['source_sha256'] = actual_sha
    plan['final_sha256'] = final_sha
    plan['num_episodes'] = num_episodes
    plan['num_transitions'] = num_transitions
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dest-dir',
        type=Path,
        default=None,
        help='Directory for the zst and decompressed HDF5.',
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Override STABLEWM_HOME when resolving the canonical path.',
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Actually download and decompress. Default is plan-only.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = acquire_pusht(
        dest_dir=args.dest_dir,
        cache_dir=args.cache_dir,
        download=args.download,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
