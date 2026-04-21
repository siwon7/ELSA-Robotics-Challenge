# VolumeDP To Federated Design

## 목표

`VolumeDP`의 핵심 아이디어를 현재 ELSA/FLAME 파이프라인에 맞게 번역해서,

- `same-env`에서 먼저 구조 타당성을 확인하고
- 그 다음 `federated`로 올릴 수 있는 실험 순서를 정리한다.

논문:
- `VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning`
- arXiv: `2603.17720`
- URL: `https://arxiv.org/pdf/2603.17720`

## 논문 핵심

논문은 세 모듈로 구성된다.

1. `Volumetric Representation`
   - RGB 이미지 feature를 3D voxel volume으로 lift한다.
   - camera intrinsic/extrinsic으로 voxel을 image plane에 투영하고,
   - `Volume-Image Cross-Attention`으로 3D feature를 만든다.

2. `Spatial Token Generation`
   - volumetric feature 중 task-relevant voxel만 compact token으로 압축한다.

3. `Multi-Token Decoder`
   - spatial token 전체를 condition으로 diffusion decoder가 action sequence를 예측한다.

논문 기준으로 이 방식은 `keyframe` 예측이 아니라, `action sequence`를 생성하는 diffusion policy에 가깝다.

## 현재 repo 제약

현재 repo의 기본 관측은 다음뿐이다.

- `front_rgb`
- `low_dim_state = 7 joint positions + 1 gripper`

현재 데이터 로더는 camera intrinsics/extrinsics를 읽지 않는다.

관련 파일:
- [dataset_loader.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/dataset/dataset_loader.py)
- [utils.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/utils.py)

즉 논문의 `volumetric lifting`을 그대로 구현하려면 먼저 dataset path에서 camera metadata를 꺼내도록 바꿔야 한다.

## 왜 FL에 맞는가

VolumeDP 계열은 `camera viewpoint shift`와 `background shift`에 대응하려는 목적에 잘 맞는다.

현재 파이프라인은 `2D image -> action` 직결이라,
- 시점이 바뀌면 feature-action mapping이 흔들리고
- client별 visual distribution shift가 큰 FL에서 drift가 커진다.

반면 VolumeDP 계열은
- image feature를 world-aligned voxel space로 lift하고
- spatial token으로 다시 압축하기 때문에
- 서로 다른 camera client 간 정렬 가능성이 더 높다.

## FL로 옮길 때 그대로 가져오면 안 되는 것

논문 구조를 그대로 전부 옮기는 건 현재 repo 기준으로 과하다.

이유:
- dataset loader가 camera metadata를 아직 직접 쓰지 않음
- multi-view 입력이 없음
- current runtime은 same-env 확인이 먼저 필요함
- FL에서 full backbone finetuning은 drift가 커짐

따라서 바로 가야 할 구조는 `VolumeDP-lite`다.

## 권장 단계

### Stage 0: same-env 구조 검증

논문 full 구현 전에, action family가 맞는지 확인한다.

우선순위:
- `JV direct + diffusion`
- `JP direct one-step + diffusion`
- `JP direct chunk4 + exec2 + diffusion`
- `JP direct keyframe4 + diffusion`

이 단계에서는
- `DINO LoRA`
- `diffusion head`
- `same-env`
만 사용한다.

### Stage 1: VolumeDP-lite

최소 구현:

1. `DINO` image feature 추출
2. predefined volume grid 생성
3. camera intrinsics/extrinsics로 voxel -> image projection
4. lightweight deformable / projected cross-attention으로 voxel feature 생성
5. voxel token top-k pooling
6. diffusion decoder로 action sequence 생성

여기서 action은 `JV direct`를 우선 추천한다.

이유:
- 논문도 sequence decoder 구조다
- 현재 same-env 실험에서도 diffusion은 `JV` 쪽이 가장 강했다
- `JP`는 execution semantics 영향이 커서 논문 구조의 장점이 덜 직접적으로 드러난다

### Stage 2: Federated

첫 FL 버전은 아래처럼 보수적으로 간다.

global trainable:
- `DINO LoRA`
- `VolumeDP-lite volumetric module`
- `spatial token module`
- `diffusion decoder`

global frozen:
- base DINO weights

server method:
- `FedAvg` 먼저
- 그 다음 비교군으로 `FedProx`

권장 스케줄:
- `local_epochs = 1~5`
- `rounds = 20+`

이유:
- diffusion + LoRA + volumetric module은 parameter drift가 크다
- local epoch가 길면 client-specific camera bias가 global model을 망치기 쉽다

## 현재 기준 추천 action

현재 same-env 결과를 기준으로 하면:

- `slide`: `JV direct + diffusion`이 가장 유망
- `JP` 계열은 diffusion을 넣어도 `slide`에서 `JV`보다 약했음
- 다른 task는 아직 범용 winner가 명확하지 않아서 same-env 4-task 추가 sweep이 필요함

따라서 VolumeDP-lite의 첫 action head는

- `primary`: `JV direct`
- `ablation`: `JP one-step`, `JP chunk4`, `JP keyframe4`

로 두는 것이 맞다.

## 추천 실험 순서

1. `same-env 4-task`
   - `DINO LoRA + diffusion + JV`
   - `DINO LoRA + diffusion + JP one-step`
   - `DINO LoRA + diffusion + JP chunk4 + exec2`
   - `DINO LoRA + diffusion + JP keyframe4`

2. task별 winner 확인

3. winner action family를 기준으로 `VolumeDP-lite same-env`

4. 마지막으로 `federated`

## 실무적 판단

지금 당장 `DINO + fede`를 밀고 싶다면,

- `VolumeDP full`을 바로 넣는 것보다
- `DINO LoRA + diffusion + same-env 4-task action sweep`
- 그 다음 `VolumeDP-lite`
- 그 다음 `FedAvg/FedProx`

순서가 맞다.

즉 논문을 FL로 옮기는 건 가능하지만,
현재 repo 상태에선 `same-env action family 검증 -> volumetric module 추가 -> federated` 순서로 가야 한다.
