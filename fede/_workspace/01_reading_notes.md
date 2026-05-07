# IC613 Reading Notes

## 강의가 중점적으로 푸는 문제

1. SGD/local update가 왜 수렴하고 왜 variance가 생기는지.
2. FL에서 FedAvg가 non-IID client에서 왜 client drift를 만드는지.
3. statistical heterogeneity, computational heterogeneity, partial participation을 어떻게 나눠 보는지.
4. FedProx/SCAFFOLD/FedNova/FedExp가 각각 어떤 drift 또는 inconsistency를 줄이는지.
5. fairness, personalization, privacy는 단순 optimizer 문제가 아니라 objective/evaluation contract 문제라는 점.

## ELSA 적용 요약

- 지금은 SCAFFOLD부터 구현할 때가 아니라, TGAC로 action/controller heterogeneity를 줄이고
  trainable-only aggregation surface를 명확히 하는 단계다.
- FedAvg는 반드시 남긴다. 기존 pilot에서 FedAvg가 FedProx보다 좋았기 때문이다.
- FedProx는 `mu=1e-3` 하나가 아니라 작은 grid로 봐야 한다.
- VolumeDP는 camera geometry heterogeneity를 줄일 수 있지만, coordinate contract가
  깨지면 FedAvg drift를 더 키운다.
- q-FFL/AFL/MaxFL은 mean SR보다 worst-env/task SR을 논문 metric으로 끌어올릴 때 유용하다.
