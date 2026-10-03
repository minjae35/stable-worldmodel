import json
import sys

import torch

from rdwm.exp1.launcher import Job, Launcher, expand_jobs, resolve_gpus
from rdwm.exp1.planning import CEMPlanner


class _ToyModel:
    """Latent = cumulative sum of actions; records what the planner feeds it."""

    def __init__(self):
        self.max_seen = 0.0

    def rollout(self, z_hist, a_hist, a_future, env):
        self.max_seen = max(self.max_seen, float(a_future.abs().max()))
        total = a_future.sum(dim=2).cumsum(dim=1)  # [S, K, d]
        return z_hist[:, -1:, :1, :2] + total[:, :, None, :]


def test_cem_clamps_and_reduces_cost(cfg):
    cfg['train']['bf16'] = False
    model = _ToyModel()
    planner = CEMPlanner(model, 'pusht', cfg, [-1, -1], [1, 1], torch.device('cpu'), seed=0)
    z_hist = torch.zeros(1, 1, 1, 2)
    a_hist = torch.zeros(1, 0, 5, 2)
    goal = torch.tensor([[[4.0, -3.0]]])
    plan, info = planner.plan(z_hist, a_hist, goal)
    assert plan.shape == (5, 5, 2)
    assert model.max_seen <= 1.0
    assert plan.abs().max() <= 1.0
    assert info['elite_cost_last'] < info['elite_cost_first']
    reached = plan.sum(dim=(0, 1))
    assert torch.allclose(reached, goal[0, 0], atol=0.5)


def test_full_matrix_queue_order(cfg):
    jobs = expand_jobs(cfg, {'stage': 'full', 'matrix': {'seeds': [0]}})
    assert [j.variant for j in jobs[:3]] == ['B', 'C', 'D']
    assert [j.updates for j in jobs[:3]] == [70000] * 3
    assert [j.envs[0] for j in jobs[3:]] == list(cfg['envs'])
    assert all(j.variant == 'A' and j.updates == 10000 for j in jobs[3:])


def test_gpus_from_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    assert resolve_gpus('auto') == ['2', '3']
    assert resolve_gpus([0, 1]) == ['0', '1']


def test_free_gpu_takes_next_job_and_done_jobs_skip(cfg, tmp_path):
    jobs = [Job('B', 0, ['pusht', 'cube'], 10, 'test'),
            Job('C', 0, ['pusht', 'cube'], 10, 'test'),
            Job('A', 0, ['pusht'], 10, 'test'),
            Job('A', 0, ['cube'], 10, 'test')]
    done_dir = tmp_path / 'runs' / 'test' / 'A' / 'seed0' / 'cube'
    done_dir.mkdir(parents=True)
    (done_dir / 'done.json').write_text('{}')
    out = tmp_path / 'out'
    out.mkdir()

    def command(job):
        name = f'{job.variant}_{job.envs[0]}'
        sleep = 0.6 if job.variant == 'B' else 0.05
        code = ('import os, time, pathlib; time.sleep(%s); '
                'pathlib.Path(%r).write_text(os.environ["CUDA_VISIBLE_DEVICES"])'
                % (sleep, str(out / name)))
        return [sys.executable, '-c', code]

    launcher = Launcher(cfg, 'unused', jobs, ['7', '9'], tmp_path / 'logs', make_command=command)
    status = launcher.run()
    states = {k: v['state'] for k, v in status.items()}
    assert sorted(states.values()) == ['done', 'done', 'done', 'skipped_done']
    gpu_of = {p.name: p.read_text() for p in out.iterdir()}
    # B holds one GPU; the other GPU runs C and then the pusht specialist.
    assert gpu_of['C_pusht'] == gpu_of['A_pusht'] != gpu_of['B_pusht']
    assert json.loads((tmp_path / 'logs' / 'status.json').read_text())
