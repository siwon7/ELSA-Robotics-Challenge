# Validation Agent Report

이 문서는 현재 repo에 대해 코드/설정 검증 경로를 정리한 결과다.

관련 스크립트:
- [validate_experiment_config.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/validate_experiment_config.py)
- [audit_experiment_suite.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/audit_experiment_suite.py)

관련 공통 모듈:
- [config_validation.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/config_validation.py)
- [config_utils.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/config_utils.py)
- [path_utils.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/dataset/path_utils.py)
- [model_registry.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/model_registry.py)
- [fl_method_registry.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/fl_method_registry.py)

## 1. 무엇을 검증하나

현재 검증 경로는 아래를 본다.
- dataset root / env shard가 실제로 존재하는가
- action pipeline preset / action representation / execution adapter가 서로 일치하는가
- vision backbone 이름과 의존성이 현재 env에서 유효한가
- dataloader에서 실제 샘플을 읽을 수 있는가
- 모델 forward 시 output shape가 target shape와 일치하는가

즉 단순 YAML 문법이 아니라:
- dataset load
- model instantiate
- one-batch forward
까지 확인한다.

## 2. 이번에 고친 구조 문제

### A. server-side `prox_mu` NameError

문제:
- [server_app.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/server_app.py) 안 `create_config()`가 정의되지 않은 `prox_mu`를 참조하고 있었다.

조치:
- `conf.model.prox_mu`를 명시적으로 넘기도록 수정

### B. eval script config drift

문제:
- eval script가 기본적으로 repo 기본 [dataset_config.yaml](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/dataset_config.yaml)만 읽어서,
- 학습 당시 backbone / action pipeline / projector 설정과 평가 설정이 어긋날 수 있었다.

조치:
- checkpoint 옆에 `*.config.yaml` snapshot을 저장하도록 수정
  - [strategies.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/strategies.py)
- eval script가 explicit config가 없으면 checkpoint sidecar config를 우선 사용
  - [eval_flower_checkpoint_live.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/eval_flower_checkpoint_live.py)
  - [eval_flower_checkpoint_live_videos.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/eval_flower_checkpoint_live_videos.py)

### C. train/val split leakage

문제:
- loader가 `train_split`과 `test_split`를 서로 다른 의미로 써서 train/val overlap이 생길 수 있었다.

조치:
- train은 앞 `train_split`
- test는 나머지 뒤 구간
으로 통일
  - [dataset_loader.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/dataset/dataset_loader.py)

### D. dataloader worker dead field

문제:
- `dataset.num_workers`가 실제 Flower main path에 반영되지 않았다.

조치:
- [task.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/task.py)에서 train/val DataLoader가 `num_workers`를 실제 사용하도록 수정

### E. FL config precedence drift

문제:
- `federated.prox_mu`와 `model.prox_mu`가 중복되고, 우선순위가 legacy 쪽에 있었다.

조치:
- precedence를 아래로 변경:
  - explicit override
  - `federated.prox_mu`
  - `model.prox_mu`
  - preset default
  - [fl_method_registry.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/federated_elsa_robotics/fl_method_registry.py)

## 3. audit 결과

audit 리포트:
- [latest_report.json](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/results/experiment_audit/latest_report.json)

요약:
- 전체 YAML: `21`
- 성공: `18`
- 실패: `3`

실패한 3개는 전부 template 파일이다.
- [action_pipeline_joint_position_direct_template.yaml](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/action_pipeline_joint_position_direct_template.yaml)
- [action_pipeline_joint_position_to_benchmark_joint_velocity_servo_template.yaml](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/action_pipeline_joint_position_to_benchmark_joint_velocity_servo_template.yaml)
- [action_pipeline_joint_velocity_direct_template.yaml](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/action_pipeline_joint_velocity_direct_template.yaml)

이유:
- template이라 `dataset.task`가 비어 있고, 실제 dataset shard를 특정할 수 없기 때문이다.

즉 현재 실험용 baseline YAML들은 구조적으로는 통과하고, template만 의도적으로 미완성 상태다.

## 4. 대표 baseline 검증 결과

대표 추천 baseline:
- [slide_block_to_target_chunk3_dinov3_fedprox_main.yaml](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/experiments/slide_block_to_target_chunk3_dinov3_fedprox_main.yaml)

CPU validator 결과:
- dataset size: `8428`
- image shape: `(1, 3, 128, 128)`
- low-dim shape: `(1, 8)`
- action shape: `(1, 24)`
- predicted shape: `(1, 24)`
- warnings: `없음`

의미:
- dataloader
- model backbone
- action chunk dim
- FL preset
이 현재 코드와 맞는다.

## 5. 지금 남아 있는 정리 포인트

아직 남은 건 크게 두 개다.

1. 여러 YAML에 `dataset.task`가 명시돼 있지 않다
- 현재 runner는 task를 주입하므로 실행은 가능하다
- 하지만 self-contained config로 만들려면 YAML에 직접 넣는 게 낫다

2. 여러 legacy YAML이 `legacy_auto` action preset과 구 필드를 아직 같이 쓴다
- 실행은 가능하다
- 다만 장기적으로는 `action_pipeline_preset`을 명시하는 쪽이 더 낫다

## 6. 앞으로의 사용법

실험 전:

```bash
source scripts/common_env.sh
conda activate "$ELSA_ENV_NAME"
python scripts/validate_experiment_config.py \
  --config experiments/slide_block_to_target_chunk3_dinov3_fedprox_main.yaml \
  --task slide_block_to_target \
  --env-id 0 \
  --split train \
  --normalize
```

suite 전체 확인:

```bash
source scripts/common_env.sh
conda activate "$ELSA_ENV_NAME"
python scripts/audit_experiment_suite.py \
  --experiments-dir experiments \
  --output results/experiment_audit/latest_report.json \
  --normalize
```

즉 앞으로는:
- 실험 YAML 작성
- validator 통과
- GPU/CPU run
순서로 가면 된다.
