# CoRL Experiment Layout

This repository now keeps the research workflow separate from the default Flower entrypoint.

## Agent split

- `research_director`: owns claims, priorities, and synthesis
- `repro_guardian`: owns dataset integrity, rerun hygiene, and clean-run certification
- `implementer`: the only write agent
- `result_analyst`: owns metric summaries and checkpoint comparisons
- `counterexample_scout`: owns reviewer-style criticism

## Task slots

The codebase exposes four tasks, so the current plan treats them as four slots:

- slot `0`: `close_box` (`protected`)
- slot `1`: `insert_onto_square_peg` (`protected`)
- slot `2`: `scoop_with_spatula` (`queued`)
- slot `3`: `slide_block_to_target` (`active`)

Slots `0` and `1` are intentionally left untouched.

## Current execution order

1. `slot3_slide_baseline_repro`
2. `slot3_slide_structured_retry`
3. `slot2_scoop_baseline_pilot`

## Useful commands

List the current plan:

```bash
python -m federated_elsa_robotics.experiment_plan --list
```

Show one experiment:

```bash
python -m federated_elsa_robotics.experiment_plan --experiment slot3_slide_baseline_repro
```

Run the dataset preflight for the active slide task:

```bash
python -m federated_elsa_robotics.repro_guard --slot 3 --split training --retries 3
```

Validate a config/model/dataloader/action combination on CPU before a real run:

```bash
python scripts/validate_experiment_config.py \
  --config experiments/slide_block_to_target_chunk3_dinov3_fedprox_main.yaml \
  --task slide_block_to_target \
  --env-id 0 \
  --split train \
  --normalize
```

Reference docs:
- [FL roadmap](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/fl_experiment_roadmap_kr.md)
- [FL method plan](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/federated_method_plan_kr.md)
- [Action presets](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/action_pipeline_presets_kr.md)
- [Model catalog](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/model_catalog_kr.md)

## IC613 FL Smoke Configs

Runnable FL presets now cover both server-only and non-server-only axes:

| Config | Method | Notes |
| --- | --- | --- |
| `fl_dinov3_diffusion_lora4_jvdirect_fedavg.yaml` | `fedavg` | global-only baseline |
| `fl_dinov3_diffusion_lora4_jvdirect_fedprox.yaml` | `fedprox_visual_shift` | FedProx local objective |
| `fl_dinov3_diffusion_lora4_jvdirect_fedper_head.yaml` | `fedper_head` | shared body + client-local diffusion/gripper heads |
| `fl_dinov3_diffusion_lora4_jvdirect_fedprox_fedper_head.yaml` | `fedprox_fedper_head` | FedProx shared body + local heads |
| `fl_dinov3_diffusion_lora4_jvdirect_fedexp.yaml` | `fedexp` | paper-form adaptive server LR, bounded to 3.0 by default for smoke stability |
| `fl_dinov3_diffusion_lora4_jvdirect_fednova.yaml` | `fednova` | local-step-normalized aggregation |
| `fl_dinov3_diffusion_lora4_jvdirect_qfedavg.yaml` | `qfedavg` | q-FFL/q-FedAvg dynamic-step update; `--no-qffl-dynamic-step` switches to a safer loss-weighted ablation |
| `fl_dinov3_diffusion_lora4_jvdirect_afl.yaml` | `afl` | agnostic FL-style client/domain weights |
| `fl_dinov3_diffusion_lora4_jvdirect_maxfl.yaml` | `maxfl` | MaxFL-inspired threshold weighting |

CPU smoke only, with unique artifact roots:

```bash
ELSA_DATASET_CONFIG_PATH_OVERRIDE=experiments/fl_dinov3_diffusion_lora4_jvdirect_qfedavg.yaml \
METRICS_PROBE_BATCHES=2 \
SUMMARY_ROOT=/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/ic613_fl_smoke_cpu \
CHECKPOINT_ROOT=/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/model_checkpoints_ic613_fl_smoke_cpu \
NUM_CLIENTS=40 \
RAY_NUM_CPUS=8 \
scripts/run_flower_programmatic_one_task_cpu.sh \
  slide_block_to_target 2 1 ic613-smoke-qfedavg-r2e1-20260507-v1 0.05 0.9 0.0
```

For `fedper_head` and `fedprox_fedper_head`, personalized eval needs the server checkpoint plus:

```text
<checkpoint-root>/<task>/client_local_state/<run-tag>/partition_<id>_env_<env>.pt
```
