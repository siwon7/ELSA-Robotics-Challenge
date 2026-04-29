#!/usr/bin/env bash
set -euo pipefail
cd /home/cvlab-dgx/siwon/ELSA-Robotics-Challenge

CHECK_INTERVAL=3540
RESULT_ROOT="/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results"
CODEX_BIN="/home/cvlab-dgx/.local/bin/codex"
LOG="wave_loop_log.txt"

log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

collect_results() {
    echo "=== Results $(date '+%Y-%m-%d %H:%M') ===" > /tmp/wave_results.txt
    find "$RESULT_ROOT" -name "result.json" 2>/dev/null | sort | while read -r f; do
        sr=$(python3 -c "
import json
d=json.load(open('$f'))
print(d.get('success_rate',d.get('sr',d.get('mean_success_rate','?'))))" 2>/dev/null || echo "err")
        echo "SR=$sr  ${f#$RESULT_ROOT/}" >> /tmp/wave_results.txt
    done
}

design_next() {
    local results learnings
    results=$(cat /tmp/wave_results.txt)
    learnings=$(cat learnings.md 2>/dev/null || echo "")

    log "Calling codex for next experiment design..."
    "$CODEX_BIN" exec <<PROMPT
You are an experiment designer for ELSA/FLAME robotic manipulation.

## Results so far
$results

## Existing learnings
$learnings

## Strategy (follow strictly)
Phase 1 - Action sweep (same-env, env0, 50ep):
  For each task, try: JV direct, JP direct, JP servo, with one-step and chunk4exec2.
  Skip combinations already in results above.

Phase 2 - Vision encoder sweep (same-env):
  Once best action per task is known, try: CNN, DINO frozen, DINO+LoRA4, DINO+LoRA8, DINO+Depth+LoRA8.
  Skip if Phase 1 is not yet complete for all tasks.

## Instructions
1. Read results. Identify what has NOT been tried yet.
2. Write a NEW prd.json with 2-4 stories, each creating ONE new YAML config.
3. branchName: ralph/auto-wave
4. Use experiments/sameenv_dino_depth_diffusion_lora8_jvdirect.yaml as template.
5. JP bounds: min [-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973,0.0] max [2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973,1.0]
6. Tasks: slide_block_to_target, close_box, insert_onto_square_peg, scoop_with_spatula
7. Dataset: /mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets/training
8. Do NOT modify existing code, only create new YAML configs.
9. Append observations to learnings.md (append-only).
10. Output DONE when finished.
PROMPT
}

launch_training() {
    local wave_name="$1"
    local configs
    configs=$(find experiments/ -name "*.yaml" -newer prd.json 2>/dev/null | sort)
    [ -z "$configs" ] && { log "No new configs."; return 1; }

    tmux kill-session -t "$wave_name" 2>/dev/null || true
    tmux new-session -d -s "$wave_name" -n "gpu0"
    local gpu=0 first=1

    while IFS= read -r cfg; do
        [ -z "$cfg" ] && continue
        local task
        task=$(python3 -c "
import yaml,os
c=yaml.safe_load(open('$cfg'))
t=c.get('dataset',{}).get('task','')
if not t:
    n=os.path.basename('$cfg')
    for x in ['slide_block_to_target','close_box','insert_onto_square_peg','scoop_with_spatula']:
        if x.split('_')[0] in n: t=x; break
print(t or 'slide_block_to_target')" 2>/dev/null)

        [ $first -eq 0 ] && tmux new-window -t "$wave_name" -n "gpu${gpu}"
        first=0
        local cmd="cd '$PWD' && source scripts/prepare_live_eval_env.sh && conda activate elsa_challenge && CUDA_VISIBLE_DEVICES='$gpu' python scripts/train_same_env_bcpolicy_probe.py --task '$task' --dataset-config-path '$cfg' --epochs 50 --eval-episodes 20 --device cuda:0 --seed 0 --run-name '${task}_${wave_name}'"
        tmux send-keys -t "$wave_name:gpu${gpu}" "$cmd" C-m
        log "GPU$gpu: $task <- $cfg"
        gpu=$((gpu+1))
        [ $gpu -ge 4 ] && break
    done <<< "$configs"
    return 0
}

# ===== Main =====
WAVE=2
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -gt 0 ] && log "Wave 1 running. Monitoring..."

while true; do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then
        log "Training running ($gpu_procs GPUs). Sleep 59m."
        sleep $CHECK_INTERVAL
        continue
    fi

    log "=== GPU idle. Wave $WAVE cycle start ==="
    collect_results
    log "Results:"; cat /tmp/wave_results.txt >> "$LOG"

    # Codex 다음 실험 설계
    design_next

    # Ralph 실행
    git stash 2>/dev/null || true
    git checkout main 2>/dev/null || true
    git branch -D ralph/auto-wave 2>/dev/null || true
    if python3 ralph.py 2>&1 | tee -a "$LOG"; then
        log "Ralph OK."
        git checkout main 2>/dev/null || true
        git merge ralph/auto-wave --no-edit 2>/dev/null || true
    else
        log "Ralph failed. Retry next cycle."
        git checkout main 2>/dev/null || true
        sleep $CHECK_INTERVAL; continue
    fi

    # 학습 시작
    if launch_training "auto_wave${WAVE}"; then
        log "Wave $WAVE launched."
        WAVE=$((WAVE+1))
    fi
    sleep $CHECK_INTERVAL
done
