# Experiment Record 2026-04-21

이 문서는 2026-04-21 시점까지 이 repo에서 수행한 주요 실험을 한 곳에 기록한 로그다.

- 기준 저장소: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge`
- 실험 산출물 루트: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts`
- 공식 baseline 재측정 산출물 루트: `/tmp/elsa_official_baseline`
- 시점 표기는 가능한 경우 `run_name`, 로그 디렉토리명, 결과 디렉토리명 기준이다.
- SR은 기본적으로 live eval 성공률이다.
- same-env는 별도 언급이 없으면 `env0` 기준이다.

## 1. 공통 실행 경로

### 핵심 코드
- 모델/정책: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/agent.py`
- 액션 실행/어댑터: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/utils.py`
- 공용 live rollout/video: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/live_rollout.py`
- same-env 학습/평가: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/train_same_env_bcpolicy_probe.py`
- same-env 단일 실행 wrapper: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/run_same_env_config_with_fallback.sh`
- same-env 진단: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/run_same_env_diagnostics.py`
- federated client: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/client_app.py`
- federated server: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/server_app.py`
- federated task/train loop: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/task.py`
- federated strategy: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/strategies.py`
- official baseline eval script: `/tmp/elsa_official_baseline/scripts/eval_official_bcpolicy_sr.py`

### 자주 쓴 명령어 템플릿

same-env 단일 실험:

```bash
bash scripts/run_same_env_config_with_fallback.sh \
  <task> <gpu_id> <config_path> <run_name> <seed> <eval_episodes> <env_id> <fallback_sr_threshold>
```

same-env 검증:

```bash
python scripts/validate_experiment_config.py \
  --config <config_path> \
  --task <task> \
  --env-id 0 \
  --split train \
  --normalize
```

official baseline rerun:

```bash
source scripts/prepare_live_eval_env.sh
unset ELSA_SIM_HEADLESS QT_QPA_PLATFORM
CUDA_VISIBLE_DEVICES=<gpu> \
/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python \
  /tmp/elsa_official_baseline/scripts/eval_official_bcpolicy_sr.py \
  --task <task> \
  --checkpoint <ckpt_path> \
  --output-dir <output_dir>
```

중요:
- `run_same_env_config_with_fallback.sh`는 외부 GPU를 `CUDA_VISIBLE_DEVICES=<gpu_id>`로 고정한 뒤 내부 train script에는 `--device cuda:0`으로 넘긴다.
- official baseline은 `QT_QPA_PLATFORM=offscreen`에서 `libsimExtCustomUI.so` segfault가 났다.
- official rerun은 `xvfb + software GL + QT_QPA_PLATFORM unset`로 돌려야 안정적이었다.

## 2. 액션 인터페이스 / replay 검증

관련 문서:
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/action_pipeline_presets_kr.md`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/replay_and_sameenv_handoff_kr.md`

### replay upper bound

| Action pipeline | close_box | slide | insert | scoop | 비고 |
|---|---:|---:|---:|---:|---|
| `joint_velocity_direct` | `1.0` | `1.0` | `0.0` | `0.2` | benchmark 기본 인터페이스 |
| `joint_position_direct` | `1.0` | `1.0` | `0.8` | `0.9` | 범용 replay 상한이 가장 안정적 |
| `joint_position -> benchmark JV servo (g20/c1.0/s2)` | `0.9` | `1.0` | `0.9` | `0.9` | strict JV adapter |

정리:
- replay 기준으로는 `joint_position_direct`와 `JP -> benchmark JV servo`가 유효했다.
- learned policy 성능 병목은 replay보다는 prediction quality와 closed-loop 안정성 쪽이었다.

## 3. same-env action sweep with frozen DINO

관련 config:
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_onestep_dinov3_jvdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_onestep_dinov3_jpdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_chunk4_dinov3_jpdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_action_keyframe4_dinov3_jpdirect.yaml`

기준:
- task: `slide_block_to_target`
- env: `env0`
- eval: `20ep`

| 설정 | SR | RMSE | 결과 파일 |
|---|---:|---:|---|
| `JV direct + one-step + DINO` | `0.4` | `0.3969` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_action_onestep_dinov3_jvdirect/env_000/result.json` |
| `JP direct + chunk4 + exec2 + DINO` | `0.2` | `0.0688` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_action_chunk4_dinov3_jpdirect/env_000/result.json` |
| `JP direct + one-step + DINO` | `0.0` | `0.0671` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_action_onestep_dinov3_jpdirect/env_000/result.json` |
| `JP direct + keyframe4 + DINO` | `0.0` | `0.0666` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_action_keyframe4_dinov3_jpdirect/env_000/result.json` |
| `JP -> benchmark JV servo` 계열 | `0.0` | - | servo 실행이 same-env에서 병목 |

결론:
- early same-env winner는 `JV direct + one-step`.
- `JP -> benchmark JV servo`는 same-env 기준으로 탈락했다.

## 4. CNN vs DINO hybrid AdaLN

관련 config:
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_cnn_jvdirect_adaln_hybrid.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_cnn_jpdirect_chunk4exec2_adaln_hybrid.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_jvdirect_adaln_hybrid.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_jpdirect_chunk4exec2_adaln_hybrid.yaml`

기준:
- task: `slide_block_to_target`
- env: `env0`
- eval: `5ep`

| 설정 | 50ep SR | 100ep SR | 100ep RMSE |
|---|---:|---:|---:|
| `CNN + JV direct + hybrid AdaLN` | `0.2` | `0.4` | `0.4004` |
| `CNN + JP chunk4 + exec2 + hybrid AdaLN` | `0.0` | `0.2` | `0.0540` |
| `DINO + JV direct + hybrid AdaLN` | `0.0` | `0.0` | `0.3931` |
| `DINO + JP chunk4 + exec2 + hybrid AdaLN` | `0.0` | `0.0` | `0.0500` |

대표 결과 파일:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_cnn_jvdirect_adaln_hybrid_e100_v1/env_000/result.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_cnn_jpdirect_chunk4exec2_adaln_hybrid_e100_v1/env_000/result.json`

결론:
- 이 라인에서는 `CNN + JV`가 DINO보다 나았다.
- AdaLN 자체가 문제를 해결하지는 못했다.

## 5. deterministic DINO LoRA sweep

관련 config:
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_lora_last4_jvdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_lora_all12_jvdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_lora_last4_jpdirect_chunk4exec2.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_lora_all12_jpdirect_chunk4exec2.yaml`

기준:
- task: `slide_block_to_target`
- env: `env0`
- eval: `20ep`

| 설정 | 50ep SR | 100ep SR |
|---|---:|---:|
| `last4 + JV direct` | `0.25` | `0.15` |
| `all12 + JV direct` | `0.15` | `0.10` |
| `last4 + JP chunk4 + exec2` | `0.15` | `0.15` |
| `all12 + JP chunk4 + exec2` | `0.40` | `0.05` |

대표 결과 파일:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_dinov3_lora_all12_jpdirect_chunk4exec2_e50_s0/env_000/result.json`

결론:
- deterministic LoRA만으로는 `SR 0.4` 근처가 ceiling이었다.
- `all12 + JP chunk4 + exec2`가 deterministic LoRA best였다.

## 6. diffusion head + DINO LoRA sweep on slide

관련 config:
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora2_jvdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora4_jvdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora8_jvdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora12_jvdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora4_jpdirect_chunk4exec2.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora8_jpdirect_chunk4exec2.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_sameenv_dinov3_diffusion_lora12_jpdirect_chunk4exec2.yaml`

기준:
- task: `slide_block_to_target`
- env: `env0`
- eval: `20ep`

### JV direct

| 설정 | SR | RMSE |
|---|---:|---:|
| `LoRA2 + JV` | `0.45` | `0.1324` |
| `LoRA4 + JV` | `0.70` | `0.1028` |
| `LoRA8 + JV` | `0.70` | `0.0967` |
| `LoRA12 + JV` | `0.30` | `0.0982` |

### JP direct chunk4 + exec2

| 설정 | SR | RMSE |
|---|---:|---:|
| `LoRA2 + JP chunk4` | `0.15` | `0.0973` |
| `LoRA4 + JP chunk4` | `0.10` | `0.0964` |
| `LoRA8 + JP chunk4` | `0.25` | `0.0896` |
| `LoRA12 + JP chunk4` | `0.25` | `0.0869` |

대표 결과 파일:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_dinov3_diffusion_lora4_jvdirect_e50_s0/env_000/result.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_dinov3_diffusion_lora8_jvdirect_e50_s0/env_000/result.json`

결론:
- diffusion이 same-env `slide`에선 확실히 효과가 있었다.
- winner family는 `DINO LoRA(4~8) + diffusion + JV direct`.

## 7. 4-task same-env transfer

### 7.1 JV direct + DINO LoRA4 + diffusion, 50ep

| Task | SR | 결과 파일 |
|---|---:|---|
| `slide_block_to_target` | `0.70` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_block_to_target_dinov3_diffusion_lora4_jvdirect_mt_e50_s0/env_000/result.json` |
| `close_box` | `0.20` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_quick_eval/close_box_dinov3_diffusion_lora4_jvdirect_mt_e50_s0_20ep.json` |
| `scoop_with_spatula` | `0.05` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/scoop_with_spatula/scoop_with_spatula_dinov3_diffusion_lora4_jvdirect_mt_e50_s0/env_000/result.json` |
| `insert_onto_square_peg` | `0.00` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/insert_onto_square_peg/insert_onto_square_peg_dinov3_diffusion_lora4_jvdirect_mt_e50_s0/env_000/result.json` |

### 7.2 JV direct + DINO LoRA8 + diffusion, 100ep

| Task | SR | 결과 파일 |
|---|---:|---|
| `slide_block_to_target` | `0.65` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_block_to_target_dinov3_diffusion_lora8_jvdirect_mt_e100_e100_s0/env_000/result.json` |
| `close_box` | `0.05` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/close_box/close_box_dinov3_diffusion_lora8_jvdirect_mt_e100_e100_s0/env_000/result.json` |
| `scoop_with_spatula` | `0.00` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/scoop_with_spatula/scoop_with_spatula_dinov3_diffusion_lora8_jvdirect_mt_e100_e100_s0/env_000/result.json` |
| `insert_onto_square_peg` | `0.00` | `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/insert_onto_square_peg/insert_onto_square_peg_dinov3_diffusion_lora8_jvdirect_mt_e100_e100_s0/env_000/result.json` |

### 7.3 JP ablation

#### JP onestep + LoRA4 + diffusion, 50ep

| Task | SR |
|---|---:|
| `slide_block_to_target` | `0.00` |
| `close_box` | `0.00` |
| `insert_onto_square_peg` | `0.00` |
| `scoop_with_spatula` | `0.00` |

대표 결과 파일:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_block_to_target_dinov3_diffusion_lora4_jpdirect_onestep_mt_e50_s0/env_000/result.json`

#### JP keyframe4 + LoRA4 + diffusion, 50ep

| Task | SR |
|---|---:|
| `slide_block_to_target` | `0.35` |
| `close_box` | `0.20` |
| `insert_onto_square_peg` | `0.00` |
| `scoop_with_spatula` | `0.10` |

대표 결과 파일:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_block_to_target_dinov3_diffusion_lora4_jpkeyframe4_mt_e50_s0/env_000/result.json`

#### JP chunk4 + LoRA8 + diffusion

| Task | 50ep SR | 100ep SR |
|---|---:|---:|
| `slide_block_to_target` | `0.25` | `0.05` |
| `close_box` | `0.05` | `0.00` |
| `insert_onto_square_peg` | `0.00` | `0.00` |
| `scoop_with_spatula` | `0.00` | `0.00` |

결론:
- 4-task 관점에서도 `JV + diffusion`이 우세했다.
- `JP keyframe4`는 일부 task에서 신호가 있지만 범용 winner는 아니다.

## 8. federated pilot: DINO LoRA4 + diffusion + JV direct

관련 config:
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/fl_dinov3_diffusion_lora4_jvdirect_fedavg.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/fl_dinov3_diffusion_lora4_jvdirect_fedprox.yaml`
- launcher: `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/start_fl_diffusion_lora4_jv_pilot_tmux.sh`

기준:
- `rounds=10`
- `local_epochs=5`
- `fraction_fit=0.05`
- eval: `env400..409`, `5ep/env`, 총 `50 rollouts`

| Task | FedAvg | FedProx |
|---|---:|---:|
| `slide_block_to_target` | `0.16` | `0.04` |
| `close_box` | `0.34` | `0.22` |

결과 파일:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/flower_abs_joint_pos_programmatic/slide_block_to_target/dinov3-diffusion-lora4-jv-fedavg-r10e5-v1_round10.eval_env400_409_5ep.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/flower_abs_joint_pos_programmatic/slide_block_to_target/dinov3-diffusion-lora4-jv-fedprox-r10e5-v1_round10.eval_env400_409_5ep.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/flower_abs_joint_pos_programmatic/close_box/dinov3-diffusion-lora4-jv-fedavg-r10e5-v1_round10.eval_env400_409_5ep.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/flower_abs_joint_pos_programmatic/close_box/dinov3-diffusion-lora4-jv-fedprox-r10e5-v1_round10.eval_env400_409_5ep.json`

정리:
- diffusion head는 FL에서도 동작했다.
- 현재 설정에선 `FedProx`가 `FedAvg`보다 나빴다.

## 9. official FLAME baseline CNN + JV direct

### 9.1 원본 archive 결과

경로: `/tmp/elsa_official_baseline/results/official_online_eval_paper_v1_20260409`

| Task | SR | 비고 |
|---|---:|---|
| `close_box` | `0.02` | `round_21` |
| `slide_block_to_target` | `0.00` | `round_30` |
| `scoop_with_spatula` | `0.00` | `round_30` |
| `insert_onto_square_peg` | 미확인 | full eval 파일 부재 |

### 9.2 2026-04-21 xvfb rerun

headless 재측정 메모:
- `offscreen` 경로는 `libsimExtCustomUI.so` segfault
- `xvfb + software GL + QT_QPA_PLATFORM unset` 경로로 rerun 성공

결과 경로: `/tmp/elsa_official_baseline/results/official_online_eval_rerun_20260421_xvfb`

| Task | Checkpoint | SR |
|---|---|---:|
| `close_box` | `round_30` | `0.00` |
| `slide_block_to_target` | `round_30` | `0.00` |
| `scoop_with_spatula` | `round_30` | `0.00` |
| `insert_onto_square_peg` | `round_24` | `0.00` |

결과 파일:
- `/tmp/elsa_official_baseline/results/official_online_eval_rerun_20260421_xvfb/close_box/round_30.eval.sr_only.json`
- `/tmp/elsa_official_baseline/results/official_online_eval_rerun_20260421_xvfb/slide_block_to_target/round_30.eval.sr_only.json`
- `/tmp/elsa_official_baseline/results/official_online_eval_rerun_20260421_xvfb/scoop_with_spatula/round_30.eval.sr_only.json`
- `/tmp/elsa_official_baseline/results/official_online_eval_rerun_20260421_xvfb/insert_onto_square_peg/round_24.eval.sr_only.json`

## 10. DINO+Depth follow-up (current wave)

주의:
- 여기서 `Depth`는 GT depth 센서값이 아니라 monocular depth prior다.
- 구현은 `front_rgb -> DINO branch + Depth Anything branch`의 concat feature다.
- 관련 backbone 코드:
  - `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/agent.py`

관련 config:
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/sameenv_dino_depth_diffusion_lora4_jvdirect.yaml`
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml`

기준:
- same-env
- `JV direct + diffusion`

### 완료된 결과

| Task | LoRA | Epoch | SR | RMSE |
|---|---:|---:|---:|---:|
| `slide_block_to_target` | `4` | `50` | `0.60` | `0.1112` |
| `slide_block_to_target` | `8` | `50` | `0.75` | `0.1021` |
| `scoop_with_spatula` | `4` | `50` | `0.05` | `0.1282` |

결과 파일:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_dino_depth_diffusion_lora4_jvdirect_v1_e50_s0/env_000/result.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/slide_block_to_target/slide_sameenv_dino_depth_diffusion_lora8_jvdirect_v1_e50_s0/env_000/result.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/scoop_with_spatula/scoop_sameenv_dino_depth_diffusion_lora4_jvdirect_v1_e50_s0/env_000/result.json`

### 진행 중

실행 중 command:

```bash
bash scripts/run_same_env_config_with_fallback.sh \
  close_box 2 experiments/sameenv_dino_depth_diffusion_lora4_jvdirect.yaml \
  close_sameenv_dino_depth_diffusion_lora4_jvdirect_v1 0 20 0 0.5

bash scripts/run_same_env_config_with_fallback.sh \
  scoop_with_spatula 3 experiments/sameenv_dino_depth_diffusion_lora4_jvdirect.yaml \
  scoop_sameenv_dino_depth_diffusion_lora4_jvdirect_v1 0 20 0 0.5
```

현재 상태:
- `close_box`: `50ep` train/eval 진행 중, 최종 result 파일 아직 없음
- `scoop_with_spatula`: `50ep SR 0.05`로 fallback 발동, `100ep` 진행 중

### 2026-04-22 추가 wave

목적:
- `slide` winner였던 `DINO+Depth + diffusion + JV + LoRA8, 50ep`를 나머지 task에도 같은 설정으로 재평가

config:
- `/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml`

tmux 세션:
- `same_env_dino_depth_lora8_other_tasks_v1`

실행 command:

```bash
bash scripts/run_same_env_config_one_task.sh \
  close_box 0 experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml \
  50 0 close_sameenv_dino_depth_diffusion_lora8_jvdirect_v1 20 0

bash scripts/run_same_env_config_one_task.sh \
  insert_onto_square_peg 1 experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml \
  50 0 insert_sameenv_dino_depth_diffusion_lora8_jvdirect_v1 20 0

bash scripts/run_same_env_config_one_task.sh \
  scoop_with_spatula 2 experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml \
  50 0 scoop_sameenv_dino_depth_diffusion_lora8_jvdirect_v1 20 0
```

예상 결과 경로:
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/close_box/close_sameenv_dino_depth_diffusion_lora8_jvdirect_v1_e50_s0/env_000/result.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/insert_onto_square_peg/insert_sameenv_dino_depth_diffusion_lora8_jvdirect_v1_e50_s0/env_000/result.json`
- `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite/scoop_with_spatula/scoop_sameenv_dino_depth_diffusion_lora8_jvdirect_v1_e50_s0/env_000/result.json`

## 11. 현재까지의 결론

### task별 현재 best

| Task | 현재 best | 근거 |
|---|---|---|
| `slide_block_to_target` | `DINO LoRA8 + DINO+Depth + diffusion + JV`, `50ep`, `SR 0.75` | DINO-only JV winner `0.70`보다 높음 |
| `close_box` | `DINO LoRA4 + diffusion + JV`, `50ep`, `SR 0.20` 또는 `JP keyframe4 + diffusion`, `SR 0.20` | DINO+Depth close는 아직 진행 중 |
| `scoop_with_spatula` | `JP keyframe4 + diffusion`, `50ep`, `SR 0.10` 또는 `JV + diffusion`, `SR 0.05` | DINO+Depth 100ep fallback 진행 중 |
| `insert_onto_square_peg` | 아직 유효 winner 없음 | JV/JP 모두 `0.0` 수준 |

### 구조적 결론
- same-env 기준 주력 라인은 `DINO LoRA + diffusion + JV direct`다.
- `JP`는 일부 task에서 신호가 있으나 범용 winner가 아니다.
- FL에선 diffusion head가 동작했지만, 현재 설정에선 `FedAvg > FedProx`다.
- official `CNN + JV` baseline은 재측정해도 강하지 않았다.
- 현재 단계에서 `slide`는 DINO+Depth가 유효한 개선으로 보인다.

## 12. 다음 업데이트 규칙

이 문서는 다음 조건에서 갱신한다.
- running wave result 파일이 새로 생김
- 4-task full eval이 끝남
- FL pilot / official rerun / same-env ablation이 추가로 완료됨

특히 다음 결과가 생기면 바로 이 문서에 추가한다.
- `close_sameenv_dino_depth_diffusion_lora4_jvdirect_v1_e50_s0`
- `scoop_sameenv_dino_depth_diffusion_lora4_jvdirect_v1_e100_s0`
