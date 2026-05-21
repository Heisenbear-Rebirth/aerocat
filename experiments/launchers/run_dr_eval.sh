#!/usr/bin/env bash
# Standalone DR/OOD eval launcher (re-runnable independent of post-T3 chain).
# Used to recover from the 2026-05-09 chain failure where `-m aerocat.scripts...`
# raised ModuleNotFoundError. We invoke the script directly instead.
set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$(pwd)/../.jax_cache}"

PYTHON="${AEROCAT_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "python" ]; then
        PYTHON=python
    elif [ -x "/root/jax_env/bin/python" ]; then
        PYTHON=/root/jax_env/bin/python
    else
        PYTHON=python3
    fi
fi

LOGDIR=experiments/_logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/_post_T3_dr_eval.log"
DR_OUT="experiments/_dr_eval_results.json"
mkdir -p experiments/_dr_eval

echo "===============================================" | tee "$LOG"
echo "[DR eval] starting at $(date) PID=$$"           | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"

"$PYTHON" src/aerocat/scripts/run_dr_eval.py \
    --groups A B C D E F \
    --seeds 42 123 456 789 1024 \
    --lambdas 0.0 0.3 0.5 0.7 1.0 \
    --num-episodes 100 \
    --num-envs 256 \
    --output "$DR_OUT" \
    >> "$LOG" 2>&1

RC=$?
echo "" | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"
echo "[DR eval] finished at $(date) exit=$RC"          | tee -a "$LOG"
echo "[DR eval] results: $DR_OUT"                      | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"
