"""Prepare evaluation-only RGB T-block centroids for common G3/G4 windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_evaluate_shared import canonical_hash  # noqa: E402
from scripts.train.g2_spatial import DINO_WEIGHTS, sha256_file  # noqa: E402
from scripts.train.g4_representation_diagnostic import load_split  # noqa: E402

G3_ROOT = ROOT / 'results/stage3/g3_history'
DEFAULT_REPRESENTATION = (
    ROOT / 'results/stage3/g4_diagnostics/representation_postnorm'
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--representation-dir', type=Path, default=DEFAULT_REPRESENTATION,
        help='completed post-norm representation diagnostic directory',
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'STATUS').write_text('BUILDING\n')
    common_path = G3_ROOT / 'common_windows.json'
    common = json.loads(common_path.read_text())
    items = common['environments']['pusht']['test']
    data = load_split(items)
    centroids = torch.from_numpy(data['centroids'].reshape(-1, 2, 2))
    contact_gap = torch.from_numpy(data['contact_gap_px'].reshape(-1, 2))
    representation = args.representation_dir.resolve()
    probe_source = representation / 'selected_probes.pt'
    probes = torch.load(probe_source, map_location='cpu', weights_only=True)
    selected = probes['readout_selection']['best_final_layer']
    if selected != 'L12_object_crop4':
        raise ValueError(f'expected validation-selected final object crop, got {selected}')
    selected_states = probes['probes'][selected]
    torch.save({
        'centroids_xy': centroids,
        'contact_gap_px': contact_gap,
        'block_angle_probe': selected_states['angle'],
        'block_xy_probe': selected_states['block_xy'],
    }, output / 'pusht_object_readout.pt')
    identity = {
        'schema': 1,
        'purpose': 'G4 evaluation-only object-local final-layer DINO readout',
        'common_manifest_sha256': sha256_file(common_path),
        'dino_weights_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
        'representation_report_sha256': sha256_file(
            representation / 'representation_diagnostic.json'
        ),
        'selected_readout': selected,
        'probe_fit': 'source development train only; alpha/readout selected on validation only',
        'test_used_for_selection': False,
        'oracle_input': 'RGB T-block centroid; evaluation-only',
        'n_test': len(items),
    }
    metadata = {'cache_key': canonical_hash(identity), 'identity': identity}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'STATUS').write_text('READY\n')


if __name__ == '__main__':
    main()
