"""Derive the Design v0 OGBCube canonical Lance table.

The raw multiview artifact is never modified. The destination is created
with ``mode='error'`` so an existing table cannot be appended to or
overwritten. Pass ``--front-column`` after inspecting a dry-run schema;
the front-view name is not guessed silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.data.design_v0.spec import (  # noqa: E402
    assert_destination_absent,
)


def resolve_front_column(
    column_names: list[str],
    front_column: str,
) -> str:
    if front_column not in column_names:
        raise ValueError(
            f'front-view column {front_column!r} is not in the raw schema '
            f'{sorted(column_names)}. Inspect a dry-run collect first.'
        )
    if 'action' not in column_names:
        raise ValueError(
            f'raw dataset is missing action; columns={sorted(column_names)}'
        )
    return front_column


def derive_ogbcube(
    source: str | Path,
    dest: str | Path,
    *,
    front_column: str,
) -> dict[str, object]:
    """Write pixels+action only. ``source`` is left untouched."""
    from stable_worldmodel.data import get_format, load_dataset
    from stable_worldmodel.data.utils import _episode_to_step_lists

    source_path = Path(source)
    dest_path = assert_destination_absent(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    src = load_dataset(
        str(source_path),
        keys_to_load=[front_column, 'action'],
    )
    resolve_front_column(list(src.column_names), front_column)

    writer_cls = get_format('lance')

    def episodes():
        for ep_idx in range(len(src.lengths)):
            episode = src.load_episode(ep_idx)
            steps = _episode_to_step_lists(
                episode, int(src.lengths[ep_idx])
            )
            yield {
                'pixels': steps[front_column],
                'action': steps['action'],
            }

    with writer_cls.open_writer(dest_path, mode='error') as writer:
        writer.write_episodes(episodes())

    derived = load_dataset(str(dest_path))
    report = {
        'source': str(source_path),
        'dest': str(dest_path),
        'front_column': front_column,
        'source_columns': list(src.column_names),
        'dest_columns': list(derived.column_names),
        'num_episodes': int(len(derived.lengths)),
        'num_transitions': int(derived.lengths.sum()),
    }
    if sorted(report['dest_columns']) != ['action', 'pixels']:
        raise ValueError(
            'derived schema must be pixels+action only, '
            f'got {report["dest_columns"]}'
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument(
        '--front-column',
        required=True,
        help='Exact raw image column to rename to pixels (from dry-run).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = derive_ogbcube(
        args.source,
        args.dest,
        front_column=args.front_column,
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
