# ELSA FL Strategy Blueprints

이 폴더는 IC613 강의에서 나온 FL 아이디어를 ELSA/FLAME 코드에 옮길 때 쓸
작은 hook 모음이다. 아직 production strategy가 아니라, `federated_elsa_robotics`
안으로 옮겨 구현하기 전의 설계 조각이다.

- `fedprox_mu_grid`: 현재 코드의 FedProx sweep 기본 grid.
- `power_of_choice_candidates`: high-loss client/env를 의도적으로 뽑는 diagnostic selection.
- `qffl_weight`, `qffl_h`, `qffl_server_scale`: q-FedAvg식 high-loss
  delta scaling과 denominator. 단순 loss 가중과 논문식 dynamic step-size를 구분한다.
- `afl_update_lambdas`, `simplex_project`: AFL의 worst-mixture client/domain 가중.
- `maxfl_appeal_weight`: client별 요구 threshold 근처를 강조하는 MaxFL식 hook.
- `maxfl_server_lr`: MaxFL식 appeal weight 합 기반 server LR.
- `fednova_normalizer`, `fednova_effective_tau`: local step 수가 달라질 때 delta 정규화 방향.
- `fedexp_server_lr`: FedExP 논문식 norm ratio에 bounded clamp를 얹은 server LR hook.
- `squared_l2_norm`: q-FedAvg/FedExp용 delta norm 계산 smoke helper.

ELSA에 바로 넣을 때의 우선순위는 `FedExP-style server LR` -> `q-FedAvg/MaxFL
delta weighting` -> `FedNova local-step normalization` -> `SCAFFOLD control variates`
순서다. SCAFFOLD는 client별 control variate를 유지해야 해서 가장 늦게 넣는다.
