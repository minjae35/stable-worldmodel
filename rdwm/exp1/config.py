"""Config loading and run-directory layout for Experiment 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / 'scripts' / 'rdwm' / 'configs' / 'exp1.yaml'
VARIANTS = ('A', 'B', 'C', 'D')


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    cfg['data']['history_probs'] = {
        int(k): float(v) for k, v in cfg['data']['history_probs'].items()
    }
    return cfg


def env_order(cfg: dict) -> list[str]:
    return list(cfg['envs'])


def check_envs(cfg: dict, variant: str, envs: list[str]) -> list[str]:
    """Validate the env list for a variant and return it in canonical order."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant!r}')
    unknown = [e for e in envs if e not in cfg['envs']]
    if unknown:
        raise ValueError(f'unknown envs {unknown}')
    if len(set(envs)) != len(envs):
        raise ValueError(f'duplicate envs in {envs}')
    single = cfg['variants'][variant]['single_env']
    if single and len(envs) != 1:
        raise ValueError(f'variant {variant} trains one env, got {envs}')
    if not single and len(envs) < 2:
        raise ValueError(f'variant {variant} is a generalist, got {envs}')
    order = env_order(cfg)
    return sorted(envs, key=order.index)


def env_tag(cfg: dict, envs: list[str]) -> str:
    envs = sorted(envs, key=env_order(cfg).index)
    if envs == env_order(cfg):
        return 'all7'
    return '+'.join(envs)


def run_dir(
    cfg: dict, stage: str, variant: str, seed: int, envs: list[str]
) -> Path:
    """``<run_root>/<stage>/<variant>/seed<k>/<env_tag>``.

    Variant, seed, and env set each get their own level, so A specialists
    (one env each) and generalists (env set) never share a directory.
    """
    return (
        Path(cfg['paths']['run_root'])
        / stage
        / variant
        / f'seed{seed}'
        / env_tag(cfg, envs)
    )


def shared_dir(cfg: dict, seed: int) -> Path:
    """Per-seed artifacts shared by every variant (init, schedule)."""
    return Path(cfg['paths']['run_root']) / 'shared' / f'seed{seed}'


def pairs_dir(cfg: dict) -> Path:
    """Evaluation pairs are seed-independent and shared by every variant."""
    return Path(cfg['paths']['run_root']) / 'shared' / 'pairs'


def env_path(cfg: dict, env: str) -> str:
    local = Path(cfg['paths']['data_root']) / cfg['envs'][env]['local']
    if not local.exists():
        raise FileNotFoundError(
            f'{env}: local Lance table {local} is missing. Download '
            f"{cfg['envs'][env]['hf_repo']} first."
        )
    return str(local)
