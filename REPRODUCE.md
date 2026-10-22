# Reproducing Stable-WM Stage 3 (G1-G5)

This document records the environment and commands verified on the US Vast.ai
RTX 5090 host on 2026-10-22. Paths under `results/` are research artifacts;
training commands below use new output directories and must not overwrite them.

## 1. What is verified

Verified on the source host:

- Repository branch: `experiment-2`.
- Python 3.12.15, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128.
- PyTorch CUDA runtime 12.8, CUDA toolkit 12.8, cuDNN 9.10.2, NVIDIA driver
  595.84, RTX 5090 32,607 MiB.
- Frozen DINOv2-S/14 weight SHA-256:
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
- Three Lance datasets load with the paths, row counts, schemas, image shapes,
  and action dimensions below.
- The world-model test suite passes: `151 passed` with four existing warnings.
- G1-G5 production binaries have been read with PyTorch's restricted
  `weights_only=True` loader and indexed in
  `reproducibility/STAGE3_CHECKPOINT_MANIFEST.tsv`.

Not independently verified:

- The original dataset transfer did not leave a full directory SHA-256 manifest
  in this checkout. Hub revisions below are repository-recorded source revisions;
  local row counts/schema were verified, but byte identity with those Hub commits
  must be checked from the external-drive copy if a prior transfer manifest exists.
- Cross-GPU bitwise equality is not promised. Use the recorded metrics and
  numerical tolerances; G1 reproduction previously matched within `1.19e-7`.
- Optional simulator tests need packages such as `pygame`; they are not required
  for frozen-feature Stage 3 training. Full repository collection currently stops
  on two PushT environment tests when `pygame` is absent.

## 2. Environment installation

Clone the backed-up branch into the canonical path because the Stage 3 scripts
use `/workspace/stable-worldmodel` for immutable inputs:

```bash
mkdir -p /workspace
git clone --branch experiment-2 \
  https://github.com/minjae35/stable-worldmodel.git \
  /workspace/stable-worldmodel
cd /workspace/stable-worldmodel
git status --short --branch
```

Create the environment. `uv` is recommended; the Python executable must be 3.12:

```bash
uv venv --python python3.12 --seed /venv/stable-wm

uv pip install --python /venv/stable-wm/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128

grep -vE '^(torch|torchvision)==' requirements_rtx5090_g1_g5.txt \
  > /tmp/stable-wm-nontorch.txt
uv pip install --python /venv/stable-wm/bin/python \
  -r /tmp/stable-wm-nontorch.txt
uv pip install --python /venv/stable-wm/bin/python --no-deps -e .
uv pip check --python /venv/stable-wm/bin/python
```

Verify the critical runtime:

```bash
/venv/stable-wm/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert torch.__version__ == '2.9.1+cu128'
assert torch.version.cuda == '12.8'
assert torch.cuda.is_available()
PY
nvidia-smi
/venv/stable-wm/bin/python -m pytest tests/wm -q
```

The complete verified package set is in
`requirements_rtx5090_g1_g5.txt`. The project itself is installed from the
checked-out Git commit and is intentionally not pinned as an editable Git URL
inside that file.

## 3. Dataset provenance and layout

Repository-recorded Hugging Face sources:

| environment | Hub artifact | recorded revision |
|---|---|---|
| PushT | `hf://datasets/fracapuano/lewm-pusht/train.lance` | `26983123e22d0a871e52c0886c0aaee48f55e77c` |
| TwoRoom | `hf://datasets/fracapuano/lewm-tworooms/train.lance` | `a6dc0810c55cffb74500906735fa6e9ac323810c` |
| Cube | `hf://datasets/fracapuano/lewm-cube/train.lance` | `3838cba3575393ccc4e0cf1571d425b1b9514b86` |

Required local layout:

```text
/workspace/datasets/
├── lewm-pusht/train.lance/
├── lewm-tworooms/train.lance/
└── lewm-cube/train.lance/
```

Observed local identities:

| environment | rows | episodes | raw action | blocked action at frameskip 5 |
|---|---:|---:|---:|---:|
| PushT | 2,336,736 | 18,685 | 2 | 10 |
| TwoRoom | 920,809 | 10,000 | 2 | 10 |
| Cube | 2,010,000 | 10,000 | 5 | 25 |

The Lance tables store RGB pixels, episode/step indices, and actions. State and
privileged columns exist for evaluation but G2-G5 training restricts its inputs
to episode index, RGB pixels, and actions. Training does not write back to Lance.

Preprocessing is deterministic: decode RGB to uint8 224x224, convert to float in
`[0,1]`, then normalize with ImageNet mean `(0.485,0.456,0.406)` and standard
deviation `(0.229,0.224,0.225)`. Frozen DINOv2-S/14 produces a 16x16 grid of
384-dimensional final post-LayerNorm patch tokens. Five outgoing actions from
`[t,t+5)` are concatenated, and the prediction target is frame `t+5`.

Episode-disjoint train/validation/test membership and exact sampled anchors are
committed in:

- `results/stage3/g2_c1_followup/manifest.json`
- `results/stage3/g3_history/common_windows.json`
- G2 sealed/final cohort JSON files under `results/stage3/`

## 4. Shared architecture and experiment matrix

Shared SpatialDynamics has 6,794,368 parameters: DINO token dimension 384,
hidden dimension 256, depth 6, 8 heads, MLP ratio 4, 16x16 spatial grid, and
dropout 0. Frozen DINO is never optimized. Unless noted otherwise, all controlled
runs use AdamW, LR `1e-3`, weight decay `1e-4`, gradient clip `1.0`, seed 3072,
batch 16 per environment (48 samples/update), and frameskip 5.

| stage | controlled condition | initialization | history | updates |
|---|---|---|---:|---:|
| G1 | state-selected one-step uniform MSE feasibility test | fresh | 1 | 500 |
| G2 warm | uniform/temporal sampling x MSE/L1 | G1 step 500 | 1 | 500/arm |
| G2 fresh | uniform vs temporal MSE | same fresh seed | 1 | 500 then 2,000 total |
| G2 multi-seed | uniform vs temporal MSE, seeds 3073-3075 | fresh per seed | 1 | 2,000/arm |
| G3 | H1 versus H3, uniform MSE | same fresh seed | 1 or 3 | 2,000/arm |
| G4 | H3 plus fixed sinusoidal positions `[0,1,2]`, scale 1 | same fresh seed | 3 | 2,000 |
| G5 | uniform, local-motion, action-ranking, combined | H1 step 2,000 | 1 | +500/arm |

G2 temporal coverage is label-free. G5 local-motion weights are computed from
frozen feature change in 4x4 tiles; action ranking uses deterministic in-batch
derangements and a detached negative. Exact per-arm calibration values, hashes,
and schedules are stored in each committed `config.json`, `reproducibility.json`,
and calibration artifact. G4's authoritative representation results are
post-norm; older mixed pre-/post-norm artifacts are marked `SUPERSEDED`.

## 5. Training commands

Always use a new output path. Long commands should run in tmux.

```bash
tmux new -s stablewm_reproduce
cd /workspace/stable-worldmodel
export PYTHON=/venv/stable-wm/bin/python
```

G1 now refuses an existing output directory:

```bash
$PYTHON scripts/train/g1_spatial.py \
  --output-dir /workspace/reproduction/g1_spatial_overfit500
```

G2 warm 2x2 and fresh A/B:

```bash
$PYTHON scripts/train/launch_g2_2x2.py \
  --output-root /workspace/reproduction/g2_2x2 --defer-evaluation
$PYTHON scripts/train/launch_g2_fresh_ab.py \
  --output-root /workspace/reproduction/g2_fresh_ab
```

G2 fresh multi-seed:

```bash
$PYTHON scripts/train/launch_g2_fresh_multiseed.py \
  --output-root /workspace/reproduction/g2_fresh_multiseed \
  --seeds 3073 3074 3075 --steps 2000 --checkpoint-every 500
```

G3 H1/H3 and G4 temporal encoding:

```bash
$PYTHON scripts/train/launch_g3_history.py \
  --output-root /workspace/reproduction/g3_history \
  --steps 2000 --checkpoint-every 500
$PYTHON scripts/train/launch_g4_temporal.py \
  --output-root /workspace/reproduction/g4_diagnostics \
  --steps 2000 --checkpoint-every 500
```

G5 cache/calibration, four-arm training, evaluation, and analysis:

```bash
$PYTHON scripts/train/prepare_g5_object_dynamics.py \
  --output-root /workspace/reproduction/g5_object_dynamics
$PYTHON scripts/train/launch_g5_object_dynamics.py \
  --output-root /workspace/reproduction/g5_object_dynamics
$PYTHON scripts/train/launch_g5_evaluation.py \
  --g5-root /workspace/reproduction/g5_object_dynamics
$PYTHON scripts/train/compare_g5_object_dynamics.py \
  --g5-root /workspace/reproduction/g5_object_dynamics
```

The G5 launcher includes 10 smoke updates inside each arm's 500-update budget,
validates the checkpoint, then resumes to cumulative step 2,500. It does not add
an extra training budget.

## 6. Checkpoint restore and resume

Restore external-drive files to the exact repository-relative paths listed in
`reproducibility/STAGE3_CHECKPOINTS_REQUIRED.txt`. Validate them against
`reproducibility/STAGE3_CHECKPOINT_MANIFEST.tsv` before loading.

The inventory contains:

- 1 frozen DINO backbone;
- 99 production G1-G5 training checkpoints;
- 134 total `checkpoint_step*.pt` files in the workspace, fully indexed in
  `reproducibility/STAGE3_ALL_CHECKPOINTS.tsv`;
- 35 smoke-only checkpoints (2,859,549,309 bytes) classified as regenerable and
  intentionally omitted from the required external-drive list;
- 66 checkpoints with model, optimizer, and RNG state (`exact_model_optimizer_rng`);
- 33 older checkpoints with model and optimizer but no RNG snapshot;
- no LR scheduler state, because these Stage 3 runs used constant LR.

For exact-resume checkpoints, use the original command recorded in the run's
`launcher_config.json`/`launcher.json` and add its `--resume` checkpoint. Keep the
same output directory because config hashes include it. Example for a paused G5
arm:

```bash
$PYTHON scripts/train/g5_object_dynamics.py \
  --arm A_uniform_control \
  --output-dir /workspace/reproduction/g5_object_dynamics/A_uniform_control \
  --plan /workspace/reproduction/g5_object_dynamics/experiment_plan.json \
  --resume /workspace/reproduction/g5_object_dynamics/A_uniform_control/checkpoint_step2010.pt
```

G1 and early G2 checkpoints can restore model and AdamW state, but lack saved RNG
state. They are valid for inference/evaluation and approximate continuation, not
bitwise-exact stochastic resume. Do not silently claim exact resume from those
33 files.

## 7. Evaluation and result verification

Exact G3-G5 evaluation also uses six regenerable binary cache/probe files listed
in `reproducibility/STAGE3_EVALUATION_ARTIFACTS_RECOMMENDED.txt`. Backing them up
avoids repeated DINO encoding and probe fitting. If omitted, regenerate the G3
cache from the committed common manifest and datasets:

```bash
$PYTHON scripts/train/prepare_g3_history_eval_cache.py \
  --common-manifest results/stage3/g3_history/common_windows.json \
  --cache-dir /workspace/reproduction/g3_eval_cache
```

The corrected object readout depends on the validation-selected post-norm probe.
For byte-identical G4/G5 object evaluation, restore the recommended probe/cache
artifacts rather than refitting them.

Compare reproduced JSON/CSV with the committed reports. Key gates are:

1. `pytest tests/wm -q` passes.
2. DINO and checkpoint SHA-256 values match the manifests.
3. Episode split and schedule hashes match `reproducibility.json`.
4. G1 checkpoint loads strictly and the historical evaluation differs by at
   most the recorded numerical tolerance.
5. G2-G5 common feature MSE/MAE, action-shuffle, persistence, natural/challenge,
   and probe-gate fields match committed evaluation JSON.
6. Natural state metrics that fail a probe gate remain `diagnostic_only`.
7. G4/G5 PushT object claims use the corrected post-norm object-local readout.

Primary reports:

- `results/stage3/g4_diagnostics/FINAL_REPORT.md`
- `results/stage3/g5_object_dynamics/FINAL_REPORT.md`
- G2/G3 aggregate JSON and CSV under their respective result directories.

## 8. Mac/WD checkpoint backup

Dataset copying is intentionally not part of this command. First verify the
actual external-drive mount in Mac Terminal; do not assume the volume name:

```zsh
ls -ld /Volumes/*
WD_ROOT='/Volumes/MyPassport_WD_4TB'
test -d "$WD_ROOT" || { echo "WD volume not mounted: $WD_ROOT"; exit 1; }
df -h "$WD_ROOT"
find "$WD_ROOT" -type d -name train.lance -print
```

Create a dated checkpoint backup and fetch the small manifests first:

```zsh
BACKUP_ROOT="$WD_ROOT/stable-worldmodel-stage3-2026-10-22"
mkdir -p "$BACKUP_ROOT/reproducibility" "$BACKUP_ROOT/repository-artifacts"

rsync -avh vast-gpu-5090-1:/workspace/stable-worldmodel/reproducibility/ \
  "$BACKUP_ROOT/reproducibility/"
```

Copy all 100 required G0-G5 binary artifacts (about 7.616 GiB), preserving their
repository-relative paths:

```zsh
rsync -avh --progress --partial --relative \
  --files-from="$BACKUP_ROOT/reproducibility/STAGE3_CHECKPOINTS_REQUIRED.txt" \
  vast-gpu-5090-1:/workspace/stable-worldmodel/ \
  "$BACKUP_ROOT/repository-artifacts/"
```

Recommended: also copy the six exact-evaluation artifacts (about 1.035 GiB):

```zsh
rsync -avh --progress --partial --relative \
  --files-from="$BACKUP_ROOT/reproducibility/STAGE3_EVALUATION_ARTIFACTS_RECOMMENDED.txt" \
  vast-gpu-5090-1:/workspace/stable-worldmodel/ \
  "$BACKUP_ROOT/repository-artifacts/"
```

Verify every required SHA-256 on the Mac:

```zsh
cd "$BACKUP_ROOT/repository-artifacts"
awk -F '\t' 'NR > 1 {print $5 "  " $3}' \
  "$BACKUP_ROOT/reproducibility/STAGE3_CHECKPOINT_MANIFEST.tsv" \
  | shasum -a 256 -c -

awk -F '\t' 'NR > 1 {print $3 "  " $1}' \
  "$BACKUP_ROOT/reproducibility/STAGE3_EVALUATION_ARTIFACTS.tsv" \
  | shasum -a 256 -c -
```

The first command must report 100 `OK` lines. The second reports six `OK` lines
if the recommended evaluation artifacts were copied. Check counts and space:

```zsh
wc -l "$BACKUP_ROOT/reproducibility/STAGE3_CHECKPOINTS_REQUIRED.txt"
du -sh "$BACKUP_ROOT/repository-artifacts"
```

## 9. Restore order on a new instance

1. Clone `experiment-2` and verify the Git commit reported with the backup.
2. Build `/venv/stable-wm` from the exact dependency file.
3. Restore the three Lance directories from the WD drive to `/workspace/datasets`.
4. Validate dataset row counts, schemas, episode counts, and action/image shapes.
5. Restore the 100 required binary artifacts to their relative paths.
6. Optionally restore the six exact-evaluation caches/probes.
7. Run SHA-256 verification and `pytest tests/wm -q`.
8. Load G1 and H1/G3 checkpoints and run evaluation before starting training.
9. Use a new `/workspace/reproduction/...` output root for any rerun.

GitHub contains source, configs, tests, metrics, JSON/CSV evaluations, and final
reports. It intentionally does not contain datasets, checkpoints, DINO weights,
optimizer tensors, feature caches, logs, credentials, or other large binaries.
