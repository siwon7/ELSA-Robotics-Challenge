"""Aggregate ceiling sweep results into a markdown table.

Reads results/ceiling_sweep_2026_04_30/<task>/<mode>.json files and prints a
task x mode SR table, plus a per-mode all-task summary.
"""
import json
from pathlib import Path

ROOT = Path("results/ceiling_sweep_2026_04_30")
TASKS = ["slide_block_to_target", "close_box", "insert_onto_square_peg", "scoop_with_spatula"]
TASK_LABEL = {
    "slide_block_to_target": "slide",
    "close_box": "close",
    "insert_onto_square_peg": "insert",
    "scoop_with_spatula": "scoop",
}

# Phase A modes (existing scripts) followed by Phase B modes (unified script)
MODE_ORDER = [
    "jv_stored",
    "jv_finite_diff",
    "jp_absolute",
    "jp_delta_naive",
    "jp_interp2",
    "jp_interp3",
    "jp_servo_g20",
    "jp_servo_g40",
    "jp_rel",
    "ee_ik_abs_world",
    "ee_ik_rel_world",
    "ee_ik_abs_ee",
    "ee_ik_rel_ee",
    "ee_ik_abs_world_coll",
    "ee_plan_abs_world",
    "ee_plan_rel_world",
]


def load_sr(json_path: Path) -> tuple[float | None, int, int, float | None]:
    if not json_path.exists():
        return None, 0, 0, None
    try:
        d = json.load(json_path.open())
    except Exception:
        return None, 0, 0, None
    sr = d.get("sr") or d.get("success_rate") or d.get("mean_success_rate")
    n = d.get("num_packs") or len(d.get("results") or [])
    k = d.get("num_success")
    if k is None and isinstance(d.get("results"), list):
        k = sum(1 for r in d["results"] if r.get("success"))
    if sr is None and n:
        sr = (k or 0) / n
    elapsed = d.get("elapsed_sec")
    return sr, int(k or 0), int(n or 0), elapsed


def main():
    rows = []
    for mode in MODE_ORDER:
        row = {"mode": mode}
        for task in TASKS:
            sr, k, n, _ = load_sr(ROOT / task / f"{mode}.json")
            row[task] = (sr, k, n)
        rows.append(row)

    print("# Replay-ceiling sweep — 2026-04-30")
    print()
    header = "| mode | " + " | ".join(TASK_LABEL[t] for t in TASKS) + " |"
    sep = "|" + "|".join(["---"] * (len(TASKS) + 1)) + "|"
    print(header)
    print(sep)
    for row in rows:
        cells = []
        for task in TASKS:
            sr, k, n = row[task]
            if sr is None:
                cells.append("—")
            else:
                cells.append(f"**{sr:.2f}** ({k}/{n})" if sr >= 0.5 else f"{sr:.2f} ({k}/{n})")
        print(f"| `{row['mode']}` | " + " | ".join(cells) + " |")
    print()

    print("## Best mode per task")
    print()
    print("| task | best mode | SR |")
    print("|---|---|---|")
    for task in TASKS:
        best = (None, -1.0)
        for row in rows:
            sr, _, _ = row[task]
            if sr is not None and sr > best[1]:
                best = (row["mode"], sr)
        if best[0]:
            print(f"| {TASK_LABEL[task]} | `{best[0]}` | {best[1]:.2f} |")
        else:
            print(f"| {TASK_LABEL[task]} | — | — |")


if __name__ == "__main__":
    main()
