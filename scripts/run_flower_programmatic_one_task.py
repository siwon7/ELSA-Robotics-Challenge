#!/usr/bin/env python3
"""Run one Flower simulation task in an isolated process/GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import ray
from flwr.common import Context
from flwr.common.record.recorddict import RecordDict
from flwr.simulation import start_simulation

from federated_elsa_robotics.client_app import client_fn
from federated_elsa_robotics.server_app import server_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--fraction-fit", type=float, default=0.05)
    parser.add_argument("--fraction-eval", type=float, default=0.0)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--prox-mu", type=float, default=0.0)
    parser.add_argument("--dataset-config-path", default="dataset_config.yaml")
    parser.add_argument("--checkpoint-root", default="model_checkpoints")
    parser.add_argument("--run-tag", default="programmatic")
    parser.add_argument("--wandb-project", default="BCPolicy-Training")
    parser.add_argument("--server-device", default="cuda:0")
    parser.add_argument("--client-device", default="cuda:0")
    parser.add_argument("--num-clients", type=int, default=400)
    parser.add_argument("--client-num-cpus", type=float, default=2.0)
    parser.add_argument("--client-num-gpus", type=float, default=0.1)
    parser.add_argument("--ray-num-cpus", type=int, default=24)
    parser.add_argument("--ray-num-gpus", type=int, default=1)
    parser.add_argument("--ray-temp-dir", default="")
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def make_run_config(args: argparse.Namespace) -> dict[str, bool | float | int | str]:
    return {
        "num-server-rounds": args.rounds,
        "local-epochs": args.local_epochs,
        "fraction-fit": args.fraction_fit,
        "fraction-eval": args.fraction_eval,
        "server-device": args.server_device,
        "client-device": args.client_device,
        "use-wandb": False,
        "wandb-project": args.wandb_project,
        "checkpoint-root": args.checkpoint_root,
        "run-tag": args.run_tag,
        "dataset-config-path": args.dataset_config_path,
        "dataset-task": args.task,
        "train-split": args.train_split,
        "prox-mu": args.prox_mu,
    }


def main() -> int:
    args = parse_args()
    run_config = make_run_config(args)
    os.environ["ELSA_DATASET_CONFIG_PATH"] = args.dataset_config_path
    os.environ["ELSA_DATASET_TASK"] = args.task
    os.environ["ELSA_TRAIN_SPLIT"] = str(args.train_split)
    os.environ["ELSA_LOCAL_EPOCHS"] = str(args.local_epochs)
    os.environ["ELSA_CLIENT_DEVICE"] = args.client_device
    os.environ["ELSA_PROX_MU"] = str(args.prox_mu)

    ctx = Context(
        run_id=0,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config=run_config,
    )

    components = server_fn(ctx)

    history = None
    try:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "num_cpus": args.ray_num_cpus,
            "num_gpus": args.ray_num_gpus,
        }
        if args.ray_temp_dir:
            ray_temp_dir = Path(args.ray_temp_dir)
            ray_temp_dir.mkdir(parents=True, exist_ok=True)
            ray_init_args["_temp_dir"] = str(ray_temp_dir)
        print(
            "Programmatic Flower task="
            f"{args.task} visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')} "
            f"client_device={args.client_device} prox_mu={args.prox_mu}"
        )
        history = start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=components.config,
            strategy=components.strategy,
            client_resources={
                "num_cpus": args.client_num_cpus,
                "num_gpus": args.client_num_gpus,
            },
            ray_init_args=ray_init_args,
        )
        print(f"Simulation finished for task={args.task}")
        print(history)
    finally:
        if ray.is_initialized():
            ray.shutdown()

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": args.task,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "fraction_fit": args.fraction_fit,
            "train_split": args.train_split,
            "prox_mu": args.prox_mu,
            "checkpoint_root": args.checkpoint_root,
            "run_tag": args.run_tag,
            "history_repr": repr(history),
        }
        summary_path.write_text(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
