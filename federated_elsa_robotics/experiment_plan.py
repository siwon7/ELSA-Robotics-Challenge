from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_plan() -> tuple[Path, dict]:
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = repo_root / "experiments" / "corl_plan.json"
    with plan_path.open("r", encoding="utf-8") as handle:
        return plan_path, json.load(handle)


def format_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def build_override_string(run_config: dict) -> str:
    return " ".join(f"{key}={format_scalar(value)}" for key, value in run_config.items())


def print_agents(plan: dict) -> None:
    print("Agent Map")
    for entry in plan["agent_map"]:
        print(f"- {entry['role']}: {entry['agent']} :: {entry['responsibility']}")


def print_slots(plan: dict) -> None:
    print("Task Slots")
    for slot in plan["task_slots"]:
        print(
            f"- slot {slot['slot']}: {slot['task']} [{slot['status']}] :: {slot['note']}"
        )


def print_experiment(experiment: dict) -> None:
    print(f"Experiment: {experiment['name']}")
    print(f"- slot: {experiment['slot']}")
    print(f"- variant: {experiment['variant']}")
    print(f"- launcher: {experiment['launcher']}")
    print(f"- priority: {experiment['priority']}")
    print(f"- run_config: {json.dumps(experiment['run_config'], ensure_ascii=True)}")
    print(f"- override_string: {build_override_string(experiment['run_config'])}")
    print("- notes:")
    for note in experiment["notes"]:
        print(f"  - {note}")

    if experiment["launcher"] == "current_repo":
        print('Suggested launch pattern: flwr run . --run-config "<override_string_above>"')
    else:
        print("Suggested launch pattern: use your external structured tree with the same run_config.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the CoRL experiment plan.")
    parser.add_argument("--list", action="store_true", help="List agents, task slots, and experiments.")
    parser.add_argument("--slot", type=int, help="Show experiments for a specific slot.")
    parser.add_argument("--experiment", type=str, help="Show a single experiment by name.")
    args = parser.parse_args()

    _, plan = load_plan()

    if args.experiment:
        for experiment in plan["experiments"]:
            if experiment["name"] == args.experiment:
                print_experiment(experiment)
                return
        raise SystemExit(f"Unknown experiment: {args.experiment}")

    if args.slot is not None:
        matching = [exp for exp in plan["experiments"] if exp["slot"] == args.slot]
        if not matching:
            raise SystemExit(f"No experiments configured for slot {args.slot}")
        for experiment in sorted(matching, key=lambda item: item["priority"]):
            print_experiment(experiment)
            print()
        return

    print_agents(plan)
    print()
    print_slots(plan)
    print()
    print("Experiments")
    for experiment in sorted(plan["experiments"], key=lambda item: item["priority"]):
        print(
            f"- {experiment['name']} :: slot {experiment['slot']} :: {experiment['variant']} :: {experiment['launcher']}"
        )


if __name__ == "__main__":
    main()
