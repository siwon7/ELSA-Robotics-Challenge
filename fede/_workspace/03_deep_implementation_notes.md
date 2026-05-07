# Deep Implementation Notes

Date: 2026-05-07

## Decision

Do not treat the IC613 material as a menu of strategy names. Each FL method has
a different client/server contract:

- FedProx: client-local proximal objective only.
- FedExp/FedAvgM: server-side pseudo-gradient step size or momentum.
- q-FedAvg: client start loss + pseudo-gradient norm + server denominator.
- AFL: server-side simplex weights over clients/domains.
- MaxFL: client requirement threshold + appeal weight + adaptive server LR.
- FedNova: local step/normalizer-aware delta aggregation.
- SCAFFOLD: stateful server/client control variates.
- Ditto: global model plus persistent personalized local models.

## Current ELSA Gap

Current client metrics only return `train_loss`. This is enough for vanilla
FedAvg and basic FedProx, but not enough for exact q-FedAvg, FedNova, or
SCAFFOLD.

Minimum telemetry before new strategies:

- `client_id`, `partition_id`, `env_id`, `task`
- `num_batches`, `local_epochs`, `local_steps`
- `pre_train_loss`, `post_train_loss`
- `delta_norm_sq`
- `trainable_manifest_hash`
- event-aware metrics once TGAC loss is finalized

## Safe Implementation Order

1. Add telemetry without changing aggregation.
2. Add FedExp-style bounded server LR.
3. Add q-FedAvg/MaxFL weighting as explicit strategy presets.
4. Add FedNova only if local step heterogeneity is observed.
5. Add SCAFFOLD only after stable client identity/state checkpointing exists.
6. Keep Ditto/FedPer separate from global-only benchmark runs.

## Files Added In This Pass

- `fede/docs/ic613_deep_reading_and_elsa_implementation_plan_20260507_kr.md`
- `fede/modules/strategy_blueprints.py`
- `fede/modules/smoke_strategy_blueprints.py`
- `fede/modules/README.md`

## Main Code Preparation

The active aggregation rule remains FedAvg. The main FL code now returns extra
fit metrics needed by later strategies:

- `federated_elsa_robotics/task.py`: optional pre/post loss probe and local step count.
- `federated_elsa_robotics/client_app.py`: client/env/task telemetry and delta norm.
- `federated_elsa_robotics/server_app.py`: scalar aggregation for telemetry.
- `pyproject.toml`: default `metrics-probe-batches = 0`.
- `federated_elsa_robotics/parameter_surfaces.py`: separates server-aggregated
  parameters from client-local parameters for FedPer/FedRep-style methods.
- `federated_elsa_robotics/fl_method_registry.py`: adds `fedper_head` and
  `fedprox_fedper_head` presets.

Because the default probe count is zero and the server still calls FedAvg
aggregation, this should not change the semantics of future FedAvg runs except
for additional metrics being returned. It does not affect the already running
Ralph queue process.

Non-server-only support now exists for persistent local heads. Use
`experiments/fl_dinov3_diffusion_lora4_jvdirect_fedprox_fedper_head.yaml` for a
small smoke run before trying VolumeDP full configs.
