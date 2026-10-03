# Experiment 1 archival result

RD-WM v1 architecture development stops here. The 7-env pilot was not run. Raw checkpoints and logs stay outside the repo under `/workspace/rdwm_runs/exp1/`. This note records the PushT horizon-1 diagnostic that closed the branch.

Protocol for every row below: the same first 10 PushT val pairs, candidate seeds 1000+i, horizon 1, CEM 300/30/30. This is not the paper horizon-5 / 50-episode protocol. Spearman and closer-rank compare a cost against true goal distance. Percentile 0 is best. Closer-rank 50 is chance.

## What failed

Trainable-encoder RD-WM does not plan. Oracle C, which replaces the cost with true PushT position distance, succeeds 7/10, so the CEM search itself is not the failure. Mean latent MSE does not rank actions.

| run | Spearman | closer rank | CEM | closer pairs | initial → final |
|---|---:|---:|---:|---:|---:|
| RD-WM ViT 5k | 0.143 | 46.38 | 0/10 | 0/10 | 151.09 → 1735.94 |
| RD-WM ViT 10k | 0.035 | 49.11 | 0/10 | 0/10 | 151.09 → 1402.02 |
| Frozen DINOv2 + RD-WM 5k | 0.474 | 35.92 | 0/10 | 2/10 | 151.09 → 227.63 |
| One-step loss | 0.447 | 36.64 | 0/10 | 0/10 | 151.09 → 1249.92 |
| One-step + wide predictor | 0.407 | 38.12 | 0/10 | 2/10 | 151.10 → 419.82 |
| One-step + wide + action concat | 0.322 | 41.01 | 0/10 | 0/10 | 151.09 → 1324.98 |
| Oracle C, true distance cost | — | — | 7/10 | 7/10 | 150.8 → 280.3 |

Official LeWM (`quentinll/lewm-pusht`) on the same evaluator: Spearman 0.398, closer rank 36.77, CEM 4/10, closer 6/10, distance 150.85 → 150.88. Short LeWM runs at 5k/10k updates and at matched clip counts stay well below that checkpoint. Exposure did not explain the gap.

## DINO-WM proprio split

Official checkpoint `kotmul/dinowm_patch_prop_pusht`. The predictor was not retrained for these rows.

| predictor proprio | planning cost | Spearman | closer rank | CEM | closer pairs | initial → final |
|---|---|---:|---:|---:|---:|---:|
| used | pixels + proprio | 0.953 | 21.00 | 3/10 | 10/10 | 151.08 → 51.57 |
| used | pixels only | 0.428 | 34.75 | 5/10 | 9/10 | 151.06 → 65.87 |
| embedding zero | pixels only | 0.428 | 34.75 | 0/10 | 1/10 | 151.09 → 536.37 |
| train-set mean embedding | pixels only | 0.428 | 34.75 | 0/10 | 1/10 | 151.09 → 564.70 |

Pixels-only cost still plans. Replacing the proprio embedding with zero or with the train-set mean embedding does not. The unchanged Spearman is the visual outcome ranking, which never reads the predictor. Closed-loop success on this checkpoint depends on proprio inside the predictor input.

A proprio-free retrain was started from the official recipe and stopped during epoch 1, before any weight checkpoint.
