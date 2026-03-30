import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(os.environ.get("ELSA_ROOT", Path(__file__).resolve().parents[1]))
TASKS = [
    "slide_block_to_target",
    "scoop_with_spatula",
    "insert_onto_square_peg",
    "close_box",
]
ROUNDS = [
    int(item)
    for item in os.environ.get("ELSA_LIVE_EVAL_ROUNDS", "5,25,50,100").split(",")
    if item
]
MAX_WORKERS = int(os.environ.get("ELSA_LIVE_EVAL_WORKERS", "2"))
LOCAL_EPOCHS = int(os.environ.get("ELSA_LIVE_EVAL_LOCAL_EPOCHS", "25"))
TRAIN_SPLIT = os.environ.get("ELSA_LIVE_EVAL_TRAIN_SPLIT", "0.9")
FRACTION_FIT = os.environ.get("ELSA_LIVE_EVAL_FRACTION_FIT", "0.05")
STRATEGY = os.environ.get("ELSA_LIVE_EVAL_STRATEGY", "fedavg")
MAX_RETRIES = int(os.environ.get("ELSA_LIVE_EVAL_MAX_RETRIES", "3"))
POLL_SECONDS = int(os.environ.get("ELSA_LIVE_EVAL_POLL_SECONDS", "60"))


def ckpt_path(task: str, round_num: int) -> Path:
    stem = (
        f"{STRATEGY}_FKCameraObjectPolicy_l-ep_{LOCAL_EPOCHS}"
        f"_ts_{TRAIN_SPLIT}_fclients_{FRACTION_FIT}_round_{round_num}.pth"
    )
    return ROOT / "model_checkpoints" / task / stem


def result_path(task: str, round_num: int) -> Path:
    return ROOT / "results" / "live_eval" / f"{task}_round_{round_num}.json"


def worker_log_path(worker_idx: int) -> Path:
    return ROOT / "logs" / "live_eval" / f"watchdog_worker{worker_idx}.log"


def status_path() -> Path:
    return ROOT / "results" / "live_eval" / "watchdog_status.json"


def build_jobs():
    return [{"task": task, "round": round_num, "retries": 0} for task in TASKS for round_num in ROUNDS]


def write_status(pending, running, completed):
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "root": str(ROOT),
        "strategy": STRATEGY,
        "local_epochs": LOCAL_EPOCHS,
        "train_split": TRAIN_SPLIT,
        "fraction_fit": FRACTION_FIT,
        "rounds": ROUNDS,
        "pending": pending,
        "running": running,
        "completed": completed,
    }
    out = status_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def launch_job(job, worker_idx):
    log_file = worker_log_path(worker_idx)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_path(job["task"], job["round"])
    out = result_path(job["task"], job["round"])
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"launch task={job['task']} round={job['round']} worker={worker_idx} retry={job['retries']}\n"
        )
        fh.flush()
        return subprocess.Popen(
            [
                str(ROOT / "scripts" / "run_eval_checkpoint_online.sh"),
                str(ckpt),
                job["task"],
                str(out),
                "eval",
            ],
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )


def main():
    pending = []
    running = {}
    completed = []

    for job in build_jobs():
        if result_path(job["task"], job["round"]).exists():
            completed.append(job)
        else:
            pending.append(job)

    while pending or running:
        for worker_idx, item in list(running.items()):
            proc = item["proc"]
            if proc.poll() is None:
                continue
            job = item["job"]
            out = result_path(job["task"], job["round"])
            if proc.returncode == 0 and out.exists():
                completed.append(job)
            else:
                job["retries"] += 1
                if job["retries"] <= MAX_RETRIES:
                    pending.insert(0, job)
            del running[worker_idx]

        for worker_idx in range(MAX_WORKERS):
            if worker_idx in running:
                continue
            selected_idx = None
            for idx, job in enumerate(pending):
                if ckpt_path(job["task"], job["round"]).exists():
                    selected_idx = idx
                    break
            if selected_idx is None:
                continue
            job = pending.pop(selected_idx)
            if result_path(job["task"], job["round"]).exists():
                completed.append(job)
                continue
            running[worker_idx] = {"job": job, "proc": launch_job(job, worker_idx)}

        pending_view = [{k: job[k] for k in ("task", "round", "retries")} for job in pending]
        running_view = {
            str(worker_idx): {
                "task": item["job"]["task"],
                "round": item["job"]["round"],
                "retries": item["job"]["retries"],
                "pid": item["proc"].pid,
            }
            for worker_idx, item in running.items()
        }
        completed_view = [{k: job[k] for k in ("task", "round", "retries")} for job in completed]
        write_status(pending_view, running_view, completed_view)
        if pending or running:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
