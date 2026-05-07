# Federated Learning Repo Manifest

작성일: 2026-05-07 KST

모든 clone은 `--depth 1 --filter=blob:none`로 받았다. 큰 framework repo는 참고용이며,
ELSA/FLAME에 바로 vendoring하지 않는다.

| Local path | Source | Commit | 역할 | ELSA 적용 우선순위 |
|---|---|---:|---|---|
| `repos/flower` | https://github.com/adap/flower | `9ff756caa9` | 현재 ELSA가 이미 쓰는 Flower framework, custom Strategy/ClientApp 참고 | 최상 |
| `repos/FedProx` | https://github.com/litian96/FedProx | `d2a4501f31` | FedProx 공식 구현, `mu` grid와 dissimilarity 분석 참고 | 높음 |
| `repos/FedNova` | https://github.com/JYWa/FedNova | `47b4e096df` | FedNova 공식 PyTorch 구현, local step 불균형 정규화 참고 | 중간 |
| `repos/Scaffold-Federated-Learning` | https://github.com/ki-ljl/Scaffold-Federated-Learning | `6114b73509` | SCAFFOLD PyTorch 참고 구현, control variate 구조 파악 | 중간 |
| `repos/fair_flearn` | https://github.com/litian96/fair_flearn | `f097971131` | q-FFL/q-FedAvg/AFL 구현, fairness weighting 참고 | 중간 |
| `repos/ditto` | https://github.com/litian96/ditto | `26d9b29a62` | Ditto/PFL 구현, local head/personalization ablation 참고 | 중간 |
| `repos/FedML` | https://github.com/FedML-AI/FedML | `03e11dfee6` | 대형 FL framework, benchmark/algorithm catalog 참고 | 낮음 |
| `repos/FATE` | https://github.com/FederatedAI/FATE | `5a06d9e4c4` | 산업용 secure/vertical FL framework, privacy/security 참고 | 낮음 |

## Clone하지 않은 항목

| 항목 | 이유 | 대체 |
|---|---|---|
| MaxFL | OpenReview supplementary zip은 있으나 git repo는 확인하지 못함 | 논문 objective를 `fede/modules/strategy_blueprints.py`에 hook으로 요약 |
| FedExP | 공식 git repo를 확인하지 못함 | 논문과 Flower Strategy subclass로 server LR ablation 구현 가능 |
| FedVARP | 강의 reference는 arXiv 중심이고 바로 적용 가능한 공식 repo를 확인하지 못함 | partial participation variance 분석 아이디어만 문서화 |

## 현재 레포에 바로 붙일 수 있는 부분

1. Flower custom strategy 확장: `federated_elsa_robotics/server_app.py`의
   `TrainableOnlySaveModelStrategy`를 기준으로 server LR, metric weighting,
   manifest 비교를 추가한다.
2. FedProx: 이미 local objective에 들어가 있으므로 `mu` grid만 실험화하면 된다.
3. FedNova: client가 `local_steps`와 normalized delta scale을 metric으로 반환하고,
   server가 delta averaging을 바꿔야 하므로 중간 난이도다.
4. SCAFFOLD: server/client control variate가 필요해 stateful client 관리가 들어간다.
   ELSA Flower simulation 구조에서 가장 구현비가 크다.
5. q-FFL/MaxFL: online success나 train loss 기반 client weighting으로 시작할 수 있다.
   다만 biased objective가 mean success를 망칠 수 있어 worst-env metric과 같이 봐야 한다.
