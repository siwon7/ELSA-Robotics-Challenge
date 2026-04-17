#!/usr/bin/env python3
"""Run the Flower app directly through the in-process simulation API."""

from __future__ import annotations

import argparse
from pathlib import Path

from flwr.common import Context, EventType, RecordDict
from flwr.common.config import get_fused_config_from_dir
from flwr.common.constant import RUN_ID_NUM_BYTES
from flwr.common.typing import Run
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes
from flwr.simulation.run_simulation import _run_simulation
from flwr.supercore.constant import NOOP_FEDERATION

from federated_elsa_robotics.client_app import app as client_app
from federated_elsa_robotics.server_app import app as server_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--dataset-config-path", type=str, default="dataset_config.yaml")
    parser.add_argument("--dataset-task", type=str, required=True)
    parser.add_argument("--policy-name", type=str, required=True)
    parser.add_argument("--server-device", type=str, default="cuda:0")
    parser.add_argument("--strategy-name", type=str, default="fedavg")
    parser.add_argument("--num-supernodes", type=int, default=400)
    parser.add_argument("--num-server-rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=50)
    parser.add_argument("--fraction-fit", type=float, default=0.05)
    parser.add_argument("--fraction-eval", type=float, default=0.0025)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--save-rounds", type=str, default="5,25,50,100")
    parser.add_argument("--client-num-cpus", type=float, default=2.0)
    parser.add_argument("--client-num-gpus", type=float, default=0.188)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ray-temp-dir", type=str, default="")
    return parser.parse_args()


def build_override_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "num-server-rounds": args.num_server_rounds,
        "local-epochs": args.local_epochs,
        "fraction-fit": args.fraction_fit,
        "fraction-eval": args.fraction_eval,
        "use-wandb": args.use_wandb,
        "strategy-name": args.strategy_name,
        "dataset-config-path": args.dataset_config_path,
        "dataset-task": args.dataset_task,
        "policy-name": args.policy_name,
        "server-device": args.server_device,
        "seed": args.seed,
        "deterministic-training": args.deterministic_training,
        "train-split": args.train_split,
        "save-rounds": args.save_rounds,
    }


def build_backend_config(args: argparse.Namespace) -> dict[str, object]:
    init_args: dict[str, object] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": True,
    }
    if args.ray_temp_dir:
        init_args["_temp_dir"] = args.ray_temp_dir
    return {
        "init_args": init_args,
        "client_resources": {
            "num_cpus": args.client_num_cpus,
            "num_gpus": args.client_num_gpus,
        },
        "actor": {
            "tensorflow": 0,
        },
    }


def main() -> None:
    args = parse_args()
    app_dir = args.app_dir.resolve()
    override_config = build_override_config(args)
    fused_config = get_fused_config_from_dir(app_dir, override_config)

    run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
    run = Run.create_empty(run_id)
    run.override_config = override_config
    run.federation = NOOP_FEDERATION

    context = Context(
        run_id=run_id,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config=fused_config.copy(),
    )

    print(
        "Starting direct Flower simulation: "
        f"task={args.dataset_task} policy={args.policy_name} "
        f"rounds={args.num_server_rounds} local_epochs={args.local_epochs} "
        f"supernodes={args.num_supernodes}"
    )

    _run_simulation(
        num_supernodes=args.num_supernodes,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
        client_app=client_app,
        server_app=server_app,
        backend_name="ray",
        backend_config=build_backend_config(args),
        server_app_context=context,
        app_dir=str(app_dir),
        run=run,
        verbose_logging=args.verbose,
    )


if __name__ == "__main__":
    main()
