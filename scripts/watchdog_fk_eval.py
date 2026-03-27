import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/cv25/siwon/ELSA-Robotics-Challenge")
TASKS = [
    "slide_block_to_target",
    "scoop_with_spatula",
    "insert_onto_square_peg",
    "close_box",
]
ROUNDS = [5, 25, 50, 100]
GPUS = [1, 3]
MAX_RETRIES = 5
POLL_SECONDS = 60


def ckpt_path(task: str, round_num: int) -> Path:
    return ROOT / "model_checkpoints" / task / (
        f"fedavg_FKBCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_{round_num}.pth"
    )


def result_path(task: str, round_num: int) -> Path:
    return ROOT / "results" / "fk_eval" / f"{task}_round_{round_num}.json"


def worker_log_path(gpu: int) -> Path:
    return ROOT / "logs" / "fk_eval" / f"watchdog_gpu{gpu}.log"


def status_path() -> Path:
    return ROOT / "results" / "fk_eval" / "watchdog_status.json"


def build_jobs():
    jobs = []
    for task in TASKS:
        for round_num in ROUNDS:
            jobs.append(
                {
                    "task": task,
                    "round": round_num,
                    "retries": 0,
                }
            )
    return jobs


def write_status(pending, running, completed):
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "pending": pending,
        "running": running,
        "completed": completed,
    }
    out = status_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def launch_job(job, gpu):
    log_file = worker_log_path(gpu)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"launch task={job['task']} round={job['round']} gpu={gpu} retry={job['retries']}\n"
        )
        fh.flush()
        proc = subprocess.Popen(
            [
                str(ROOT / "scripts" / "run_fk_eval_one.sh"),
                job["task"],
                str(job["round"]),
                str(gpu),
            ],
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    return proc


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
        for gpu, item in list(running.items()):
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
            del running[gpu]

        for gpu in GPUS:
            if gpu in running:
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
            if not ckpt_path(job["task"], job["round"]).exists():
                pending.append(job)
                continue

            running[gpu] = {"job": job, "proc": launch_job(job, gpu)}

        pending_view = [
            {k: job[k] for k in ("task", "round", "retries")} for job in pending
        ]
        running_view = {
            str(gpu): {
                "task": item["job"]["task"],
                "round": item["job"]["round"],
                "retries": item["job"]["retries"],
                "pid": item["proc"].pid,
            }
            for gpu, item in running.items()
        }
        completed_view = [
            {k: job[k] for k in ("task", "round", "retries")} for job in completed
        ]
        write_status(pending_view, running_view, completed_view)

        if pending or running:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
