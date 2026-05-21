#!/usr/bin/env bash
# E2 Per-basis leave-one-out launcher
# 5 basis × 5 seeds × 1B steps each, sequential
# Estimated wall time: ~84 min/run × 25 = ~35 hours
set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$(pwd)/../.jax_cache}"

PYTHON="${AEROCAT_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "python" ]; then
        PYTHON=python
    else
        PYTHON=python3
    fi
fi

LOGDIR=experiments/_logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/_e2_per_basis.log"

echo "===============================================" | tee "$LOG"
echo "[E2] starting at $(date) PID=$$"                | tee -a "$LOG"
echo "[E2] 5 bases × 5 seeds × 1B steps each (25 runs total)" | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"

SEEDS="42 123 456 789 1024"
TOTAL_TS=1000000000

# Loop over basis index (0=vel_err, 1=omega, 2=tilt, 3=PID_integral, 4=saturation)
for IDX in 0 1 2 3 4; do
    for S in $SEEDS; do
        echo "" | tee -a "$LOG"
        echo "===== [E2] basis=$IDX seed=$S start at $(date) =====" | tee -a "$LOG"
        "$PYTHON" src/aerocat/scripts/run_ablation.py \
            --group D \
            --seeds $S \
            --disable-basis-idx $IDX \
            --total-timesteps $TOTAL_TS \
            --num-envs 4096 \
            --num-steps 128 \
            --save-interval 10 \
            --log-interval 5 \
            >> "$LOG" 2>&1
        RC=$?
        echo "===== [E2] basis=$IDX seed=$S done at $(date) exit=$RC =====" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"
echo "[E2] all runs finished at $(date)"              | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"
