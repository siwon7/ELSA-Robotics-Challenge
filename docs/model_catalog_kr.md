# Model Catalog

이 문서는 현재 repo에서 지원하는 visual encoder와 upstream source를 정리한 파일이다.

핵심 목적:
- 모델을 `로컬 임시 코드`로 추가하지 말고
- 어떤 upstream source를 쓰는지 기록하고
- action space / FL 설정과 함께 재현 가능하게 관리하는 것

관련 코드:
- [model_registry.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/model_registry.py)
- [agent.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/agent.py)
- [config_validation.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/elsa_learning_agent/config_validation.py)

## 1. `cnn`

- 표시 이름: `Local CNN baseline`
- upstream source: local implementation
- dependency: `torch`
- trainable parts: 전체 encoder + policy head

언제 쓰나:
- 가장 빠른 smoke
- visual backbone이 문제인지 아닌지 아주 빠르게 분리할 때

단점:
- 색상 변화, 배경 변화, 카메라 시점 변화가 큰 FL 세팅에서는 제일 약하다

## 2. `dinov3_vits16_frozen`

- 표시 이름: `Frozen DINOv3 ViT-S/16`
- upstream source: `timm pretrained vit_small_patch16_dinov3`
- dependency: `timm`
- trainable parts: projector + policy head

언제 쓰나:
- color / background / camera shift가 큰 global FL baseline
- 지금 repo 기준 첫 번째 추천 visual encoder

장점:
- CNN보다 시각 domain shift에 더 강함
- backbone을 frozen으로 두면 FL 통신량과 drift가 줄어듦

## 3. `depth_anything_small_frozen`

- 표시 이름: `Frozen Depth Anything Small`
- upstream source: `LiheYoung/depth-anything-small-hf`
- dependency: `transformers`
- trainable parts: depth projector + policy head

언제 쓰나:
- 대상 접근은 되는데 마지막 object localization / contact initiation이 약할 때
- DINO-only baseline 다음 단계

주의:
- depth branch만 바로 메인 baseline으로 쓰기보다
- 먼저 `DINO` baseline을 안정화하는 게 낫다

## 4. `dinov3_depth_anything_small_frozen`

- 표시 이름: `Frozen DINOv3 + Depth Anything concat`
- upstream source:
  - `timm pretrained vit_small_patch16_dinov3`
  - `LiheYoung/depth-anything-small-hf`
- dependency: `timm + transformers`
- trainable parts: 두 projector + policy head

언제 쓰나:
- DINO-only baseline이 이미 정리된 뒤
- object-centric visual signal을 더 강화하고 싶을 때

주의:
- 원인 분리가 어려워지므로 첫 baseline으로는 비추천

## 5. upstream repo를 붙일 때 원칙

새 모델을 가져올 때는:
1. upstream source를 문서에 기록
2. dependency를 명시
3. `model_registry.py`에 등록
4. `config_validation.py`에서 의존성 검사
5. action space / FL method와 함께 실험 YAML을 남김

즉 “일단 코드만 복사”가 아니라:
- source
- dependency
- trainable parts
- intended FL usage
를 같이 적어야 한다

## 6. 현재 추천

현재 global FL baseline에서 첫 번째 추천은:
- `dinov3_vits16_frozen`

그 다음 확장:
- `dinov3_depth_anything_small_frozen`

즉 순서는:
1. `DINO`
2. `DINO + Depth`
3. camera adaptor / LoRA / FK-conditioned 모델
