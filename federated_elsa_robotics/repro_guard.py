from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_plan() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = repo_root / "experiments" / "corl_plan.json"
    with plan_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_task(plan: dict, slot: int | None, task: str | None, allow_protected: bool) -> tuple[int | None, str]:
    if task:
        return None, task

    if slot is None:
        raise SystemExit("Provide either --slot or --task.")

    for task_slot in plan["task_slots"]:
        if task_slot["slot"] == slot:
            if task_slot["status"] == "protected" and not allow_protected:
                raise SystemExit(
                    f"Slot {slot} is protected ({task_slot['task']}). Use --allow-protected to override."
                )
            return slot, task_slot["task"]

    raise SystemExit(f"Unknown slot: {slot}")


def parse_env_ids(task_root: Path) -> list[int]:
    env_ids = []
    for env_dir in sorted(task_root.glob("env_*")):
        try:
            env_ids.append(int(env_dir.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return env_ids


def try_load_data(data_path: Path, retries: int) -> dict:
    last_error = None
    start_time = time.time()
    for attempt in range(1, retries + 1):
        try:
            from colosseum.rlbench.datacontainer import DataContainer

            demos = DataContainer().load(str(data_path)).data
            return {
                "ok": True,
                "attempts": attempt,
                "num_demos": len(demos),
                "seconds": round(time.time() - start_time, 4),
            }
        except Exception as exc:  # pragma: no cover - mirrors runtime loader failures
            last_error = f"{type(exc).__name__}: {exc}"

    return {
        "ok": False,
        "attempts": retries,
        "error": last_error,
        "seconds": round(time.time() - start_time, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight dataset integrity before launching a training run.")
    parser.add_argument("--slot", type=int, help="Task slot index from experiments/corl_plan.json.")
    parser.add_argument("--task", type=str, help="Task name override.")
    parser.add_argument("--split", type=str, default="training", choices=["training", "eval", "test"])
    parser.add_argument("--env-start", type=int, help="First env id to inspect.")
    parser.add_argument("--env-stop", type=int, help="Stop env id (exclusive).")
    parser.add_argument("--limit", type=int, help="Limit the number of envs after filtering.")
    parser.add_argument("--retries", type=int, default=3, help="Number of load retries per env.")
    parser.add_argument("--allow-protected", action="store_true", help="Allow scanning protected slots.")
    parser.add_argument("--output", type=str, help="Optional JSON output path.")
    args = parser.parse_args()

    try:
        from colosseum.rlbench.datacontainer import DataContainer  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency `colosseum`. Run this script inside the ELSA environment that can load RLBench data."
        ) from exc

    plan = load_plan()
    slot, task = resolve_task(plan, args.slot, args.task, args.allow_protected)

    repo_root = Path(__file__).resolve().parents[1]
    task_root = repo_root / "datasets" / args.split / task
    if not task_root.exists():
        raise SystemExit(f"Task path does not exist: {task_root}")

    env_ids = parse_env_ids(task_root)
    if args.env_start is not None:
        env_ids = [env_id for env_id in env_ids if env_id >= args.env_start]
    if args.env_stop is not None:
        env_ids = [env_id for env_id in env_ids if env_id < args.env_stop]
    if args.limit is not None:
        env_ids = env_ids[: args.limit]

    if not env_ids:
        raise SystemExit("No environments matched the requested filters.")

    print(f"Repro guard for task={task}, split={args.split}, slot={slot}, env_count={len(env_ids)}")
    results = []
    failures = []

    for env_id in env_ids:
        data_path = task_root / f"env_{env_id}" / "episodes_observations.pkl.gz"
        entry = {"env_id": env_id, "path": str(data_path)}
        if not data_path.exists():
            entry["ok"] = False
            entry["error"] = "missing_file"
            failures.append(entry)
            results.append(entry)
            print(f"- env_{env_id}: missing_file")
            continue

        load_result = try_load_data(data_path, retries=args.retries)
        entry.update(load_result)
        results.append(entry)

        if load_result["ok"]:
            print(
                f"- env_{env_id}: ok in {load_result['attempts']} attempt(s), demos={load_result['num_demos']}, seconds={load_result['seconds']}"
            )
        else:
            failures.append(entry)
            print(f"- env_{env_id}: FAILED after {load_result['attempts']} attempt(s) :: {load_result['error']}")

    summary = {
        "task": task,
        "slot": slot,
        "split": args.split,
        "env_count": len(results),
        "failure_count": len(failures),
        "failed_env_ids": [entry["env_id"] for entry in failures],
        "results": results,
    }

    print()
    print("Summary")
    print(f"- failures: {summary['failure_count']} / {summary['env_count']}")
    print(f"- failed_env_ids: {summary['failed_env_ids']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"- wrote_report: {output_path}")


if __name__ == "__main__":
    main()
