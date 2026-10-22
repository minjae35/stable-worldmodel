"""Evaluation-only shared-trunk gradient conflict diagnostic for G3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lance
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_spatial import DATASETS, DINO_WEIGHTS  # noqa: E402
from scripts.train.g3_history_spatial import encode_bundle  # noqa: E402
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES, spatial_prediction_loss  # noqa: E402
from stable_worldmodel.wm.spatial.g2_fresh import torch_normalizers  # noqa: E402
from stable_worldmodel.wm.spatial.g3_history import (  # noqa: E402
    history_model_batch,
    load_label_free_history_clips,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics  # noqa: E402

NORMALIZERS = ROOT / 'results/stage3/g2_fresh_ab/action_normalizers.json'


def flattened_shared_gradient(model, bundle, env_index, device) -> tuple[torch.Tensor, float]:
    shared = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith('adapters.') and not name.startswith('env_embed.')
    ]
    indices = torch.arange(min(16, len(bundle['context']))).numpy()
    _, _, _, target, prediction = history_model_batch(
        model, bundle, indices, env_index, device
    )
    loss = spatial_prediction_loss(prediction, target, 'mse')
    gradients = torch.autograd.grad(loss, shared, retain_graph=False)
    vector = torch.cat([gradient.detach().float().flatten().cpu() for gradient in gradients])
    return vector, float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    common = json.loads((args.root / 'common_windows.json').read_text())
    normalizer_artifact = json.loads(NORMALIZERS.read_text())
    normalizers = torch_normalizers(normalizer_artifact)
    device = torch.device('cuda:0')
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)
    datasets = {env: lance.dataset(str(DATASETS[env])) for env in ENV_NAMES}
    report = {
        'purpose': 'evaluation-only shared dynamics gradient conflict diagnostic',
        'batch_size_per_environment': 16,
        'items': 'first 16 common train anchors per environment',
        'excluded_from_gradient_vector': ['environment action adapters', 'environment embedding'],
        'arms': {},
    }
    for arm, history, directory in (
        ('H1', 1, 'H1_uniform_mse'),
        ('H3', 3, 'H3_uniform_mse'),
    ):
        checkpoint_path = args.root / directory / 'checkpoint_step2000.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model = SpatialDynamics().to(device)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.train()
        vectors = {}
        losses = {}
        for env_index, env in enumerate(ENV_NAMES):
            items = common['environments'][env]['train'][:16]
            raw = load_label_free_history_clips(
                datasets[env], items, env, 5, history
            )
            bundle = encode_bundle(
                raw, normalizers[env], encoder, device, encode_batch=32
            )
            vectors[env], losses[env] = flattened_shared_gradient(
                model, bundle, env_index, device
            )
        cosines = {}
        for left_index, left in enumerate(ENV_NAMES):
            for right in ENV_NAMES[left_index + 1:]:
                cosine = torch.nn.functional.cosine_similarity(
                    vectors[left], vectors[right], dim=0
                )
                cosines[f'{left}_vs_{right}'] = float(cosine)
        report['arms'][arm] = {
            'history': history,
            'loss_by_environment': losses,
            'shared_gradient_norm_by_environment': {
                env: float(vectors[env].norm()) for env in ENV_NAMES
            },
            'shared_gradient_cosine': cosines,
            'n_negative_pairs': sum(value < 0 for value in cosines.values()),
        }
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
