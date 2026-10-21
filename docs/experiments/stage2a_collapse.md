# Stage 2-A: 공동 학습과 representation collapse

PushT, TwoRoom, Cube를 한 encoder와 predictor로 같이 학습하는 Stage 2-A에서, PushT와 TwoRoom의 latent가 거의 상수로 무너지는 현상을 확인했다. 데이터 인덱싱 오류는 원인이 아니었다. 예측 target에 stop-gradient를 주거나 variance hinge를 더해도 2000 step 안에서는 이 붕괴가 풀리지 않았다.

원본 결과와 checkpoint는 로컬 `results/stage2a/`에 그대로 있다. 이 문서는 그 파일을 옮기거나 덮어쓰지 않는다. 가중치와 checkpoint의 SHA256은 [checkpoint_manifest.json](checkpoint_manifest.json)에 있다.

## 공통 설정

한 GPU, seed 3072. ViT-tiny (`embed_dim=192`, patch 14, 입력 224). history 3, `num_preds=1`, frameskip 5. AdamW `lr=5e-5`, weight decay `1e-3`, bf16, gradient clip 1.0. 환경당 배치 32. 환경 순서는 pusht, tworoom, cube. SIGReg 가중치 0.09, knots 17, projections 1024.

Cosine 길이는 `max_epochs=100`이다. epoch당 20554 step이므로 schedule horizon은 2,055,400 step이다. 2000 step에서 끊은 비교 실험은 `trainer.max_steps`를 쓰지 않고 `stop_at_global_step`으로 멈춰서, 같은 cosine 길이를 유지한다.

## 실험

| 실험 | 목적 | 학습 목표 | 종료 | 디렉터리 |
| --- | --- | --- | --- | --- |
| `rtx5090_stage2a` | 원래 목표로 1 epoch | next-latent MSE + SIGReg | epoch 1, step 20554 | `results/stage2a/rtx5090_stage2a/` |
| `rtx5090_recover500` | 붕괴된 epoch 1에서 500 step 더 진행 | 위와 같음 | step 21054 | `results/stage2a/rtx5090_recover500/` |
| `rtx5090_base2000` | 2000 step 대조군 | 위와 같음 | step 2000 | `results/stage2a/rtx5090_base2000/` |
| `rtx5090_sg2000` | 예측 target stop-gradient | `detach_pred_target=true` | step 2000 | `results/stage2a/rtx5090_sg2000/` |
| `rtx5090_var2000` | variance hinge | weight 1, gamma 1 | step 2000 | `results/stage2a/rtx5090_var2000/` |
| `collapse_audit` | 데이터와 encoder 표현 점검 | 학습 없음 | — | `results/stage2a/collapse_audit/` |

`base2000`, `sg2000`, `var2000`의 step 1 손실과 latent 통계는 완전히 같다. step 0 가중치 SHA256도 세 실험이 같다 (`b41e8f3dc537…`). `recover500`의 시작 checkpoint는 `rtx5090_stage2a/checkpoints/resume.ckpt`와 바이트 단위로 같다.

## 1 epoch에서 본 붕괴

`rtx5090_stage2a`의 `metrics.csv`. `latent_std`는 손실에 들어가는 embedding의 배치 표준편차다.

| step | PushT pred / latent_std / dim_std_min | TwoRoom pred / latent_std / dim_std_min | Cube pred / latent_std / dim_std_min |
| --- | --- | --- | --- |
| 0 | 0.0879 / 0.211 / 0.00107 | 0.0902 / 0.216 / 0.00091 | 0.0733 / 0.176 / 0.00232 |
| 1000 | 0.00132 / 0.0304 / 0.00817 | 0.00127 / 0.0296 / 0.00598 | 0.143 / 0.492 / 0.0937 |
| 2000 | 0.000757 / 0.0306 / 0.00534 | 0.000727 / 0.0294 / 0.00383 | 0.134 / 0.652 / 0.131 |
| 10000 | 0.000237 / 0.0152 / 0.00105 | 0.000213 / 0.0146 / 0.000969 | 0.0300 / 0.939 / 0.460 |
| 20554 | 0.000250 / 0.0110 / 0.000683 | 0.000252 / 0.0110 / 0.000781 | 0.0201 / 0.944 / 0.533 |

PushT와 TwoRoom은 step 1000 즈음 예측 손실이 0.001 근처로 떨어지고, epoch가 끝날 때는 차원별 표준편차 최솟값이 1e-3 아래다. Cube는 예측 손실이 0.02로 남아 있고 latent 표준편차는 0.94까지 커진다. 전체 손실 0.815의 대부분은 PushT/TwoRoom의 SIGReg(각 12.875)이고, Cube SIGReg는 1.28이다.

같은 checkpoint에서 500 step을 더 간 `recover500`도 되돌리지 못했다. step 21054에서 PushT latent_std 0.0129, dim_std_min 7.3e-4, TwoRoom latent_std 0.0136, dim_std_min 7.9e-4, Cube latent_std 0.921이다.

## 고정 배치 probe

학습 중 로그는 배치마다 흔들린다. `probe_step_*.json`과 `probe_varcmp_step_*.json`은 같은 평가 배치에서 projector의 차원별 표준편차와 gradient를 잰 것이다. step 2000 projector `dim_std_median`:

| 실험 | PushT | TwoRoom | Cube | encoder grad가 Cube와 이루는 cosine |
| --- | --- | --- | --- | --- |
| base | 0.00383 | 0.000717 | 0.548 | 0.996 |
| stop-gradient | 0.00275 | 0.000555 | 0.690 | 0.998 |
| variance hinge | 0.00899 | 0.00142 | 1.026 | 0.995 |

세 조건 모두 공유 encoder gradient는 Cube gradient와 거의 같은 방향이다. PushT/TwoRoom projector 표준편차는 gamma=1에 비해 세 자릿수 작다. variance hinge를 켜도 PushT hinge 손실은 step 2000에 0.984, TwoRoom은 0.991이라 제약이 계속 걸린 상태고, Cube hinge는 0.082로 거의 내려가 있다.

epoch 1 직후 probe(`recover500/probe_step20554.json`)에서는 PushT와 TwoRoom projector median이 9.2e-4, 9.9e-4이고 둘의 cosine은 0.995다. encoder gradient norm은 PushT 0.053, TwoRoom 0.053, Cube 1.43이다. step 21054에서도 projector median은 9.1e-4, 9.5e-4, Cube 0.976이고, 전체 gradient와 Cube gradient의 cosine은 0.999다.

## collapse audit

`collapse_audit/metrics.json`과 `rgb_grid.png`.

데이터 쪽은 정상이다. 세 환경 모두 frameskip이 맞고, episode 첫 프레임이 서로 같은 쌍은 0개다. raw 거리와 전처리 후 거리의 상관은 0.99999보다 크다. episode 안 거리 / episode 사이 거리는 PushT 0.33, Cube 0.50, TwoRoom 0.66이다. `rgb_grid.png`에서도 PushT, TwoRoom, Cube가 서로 다른 장면으로 보인다.

초기 encoder의 projector 차원별 표준편차 median은 이미 작다. PushT 8.4e-4, TwoRoom 1.4e-4, Cube 4.1e-3. `base2000` 이후에도 PushT와 TwoRoom의 backbone은 이 근처고, Cube projector median만 0.446까지 올라간다. 1 epoch checkpoint에서는 PushT projector median이 2.4e-4로 초기보다 더 줄고, Cube는 0.897이다.

## 결론

붕괴는 PushT와 TwoRoom representation에서 실제로 일어난다. 인덱싱이 깨졌거나 프레임이 뒤섞인 데이터 문제는 아니다. 예측 대상이 거의 상수가 되면서 두 환경의 next-latent MSE가 0으로 가고, 공유 encoder의 gradient는 Cube가 차지한다.

2000 step 비교에서 두 가지 수정은 이 패턴을 바꾸지 못했다.

- 예측 target stop-gradient는 학습 로그의 배치 표준편차를 조금 올리지만, 고정 배치 projector 표준편차는 1e-3 근처에 머문다. Cube gradient 지배는 그대로다.
- variance hinge(weight 1, gamma 1)는 PushT/TwoRoom에서 계속 활성화되지만, 2000 step 안에 projector 표준편차를 gamma 근처로 끌어올리지 못한다. Cube는 hinge가 거의 0이다.

500 step을 더 이어서도 회복되지 않았다. G0 DINOv2 실험은 이 기록 이후에 시작한다.

## 코드에서 바뀐 점

기본값이면 예전 목표와 같다.

- `loss.detach_pred_target`: 예측 target에 stop-gradient.
- `loss.variance`: projector 배치 표준편차에 대한 hinge. weight 0이면 빠진다.
- `stop_after_epoch`, `stop_at_global_step`: cosine horizon을 줄이지 않고 이번 실행만 멈춘다.
- `snapshot_at_global_steps`: 지정한 step의 `weights_step_*.pt`.

## Git에 넣은 파일

각 학습 디렉터리의 `config.yaml`, `metrics.csv`, `train.log`, probe JSON, `status.json`, `topology.json`, `benchmark.json`, `action_normalizer.json`, `environment.json`, `requirements_frozen.txt`. `collapse_audit`의 `metrics.json`과 `rgb_grid.png`.

## Git에 넣지 않은 파일

가중치 `.pt`는 개당 약 70 MB로 35개 중 24개다. Lightning checkpoint `.ckpt`는 개당 약 210 MB로 10개다. `fixed_eval_batch.pt`는 220 MB 평가 배치다. GitHub 일반 파일 한도는 100 MB이고, 이 저장소 pre-commit은 1 MB를 넘으면 막는다. 50 MB를 넘는 가중치와 100 MB를 넘는 checkpoint는 승인 없이 올리지 않았다. 로컬 파일은 삭제하지 않았다.

처리량만 보던 smoke와 bench 디렉터리(`rtx5090_lance_bench`, `rtx5090_syn_*`, `rtx5090_local_bench_*`, `ddp_bench_*`)와 `results/stage3/g0_dinov2` 가중치도 이번 커밋에는 없다.
