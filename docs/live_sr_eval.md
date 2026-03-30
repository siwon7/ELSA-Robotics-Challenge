# Live SR Evaluation

This repo includes reusable scripts for measuring online success rate (SR) from saved federated checkpoints.

## What is included

- `scripts/run_eval_checkpoint_online.sh`
  Runs simulator-based SR evaluation for a single checkpoint.
- `scripts/run_eval_checkpoint_offline.sh`
  Runs offline RMSE evaluation for a single checkpoint.
- `scripts/run_fk_eval_one.sh`
  Convenience wrapper for FKCameraObjectPolicy checkpoints using the default naming convention.
- `scripts/run_eval_task_sr_gpu.sh`
  Thin wrapper around `run_fk_eval_one.sh`.
- `scripts/watchdog_fk_eval.py`
  Watches for saved checkpoints and launches SR evaluation automatically.
- `scripts/start_watchdog_live_eval_tmux.sh`
  Starts the SR watchdog in a tmux session.

## Required environment variables

Most paths are configured through `scripts/common_env.sh`.

- `ELSA_ROOT`
- `ELSA_ENV_NAME`
- `CONDA_BASE`
- `COPPELIASIM_ROOT`
- `ELSA_RLBENCH_ROOT`

Optional runtime overrides:

- `ELSA_SIM_DEVICE`
  `cpu` or `cuda:0` for online SR inference.
- `ELSA_SIM_HEADLESS`
  `1` for headless mode, `0` to use Xvfb-backed rendering.
- `ELSA_SIM_RENDERER`
  Defaults to `opengl`.
- `ELSA_SIM_NUM_EPISODES`
- `ELSA_SIM_MAX_ENVS`
- `ELSA_SIM_MAX_STEPS`

## Single checkpoint usage

Online SR:

```bash
scripts/run_eval_checkpoint_online.sh \
  model_checkpoints/close_box/fedavg_FKCameraObjectPolicy_l-ep_25_ts_0.9_fclients_0.05_round_10.pth \
  close_box \
  results/live_eval/close_box_round_10.json \
  eval
```

Offline RMSE:

```bash
scripts/run_eval_checkpoint_offline.sh \
  model_checkpoints/close_box/fedavg_FKCameraObjectPolicy_l-ep_25_ts_0.9_fclients_0.05_round_10.pth \
  close_box \
  1 \
  results/live_eval/close_box_round_10.offline.json \
  eval
```

## Watchdog usage

The watchdog polls for checkpoints and evaluates them as soon as they appear.

```bash
ELSA_LIVE_EVAL_ROUNDS=5,10,15 \
ELSA_LIVE_EVAL_LOCAL_EPOCHS=25 \
ELSA_LIVE_EVAL_FRACTION_FIT=0.05 \
python scripts/watchdog_fk_eval.py
```

Or in tmux:

```bash
scripts/start_watchdog_live_eval_tmux.sh
```

Outputs are written to:

- `results/live_eval/*.json`
- `results/live_eval/watchdog_status.json`
- `logs/live_eval/`

## Notes

- The online evaluator prepares a local Xvfb display automatically when needed.
- Policy type is inferred from checkpoint file names, so FKCameraObjectPolicy checkpoints can be evaluated without manually editing `dataset_config.yaml`.
- Cached-feature policies are also supported through the same `scripts/eval_checkpoint.py` entrypoint.
