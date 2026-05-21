#!/usr/bin/env bash
# E1 Cross-task PSC weight transfer launcher
# Two configs on T3 (disturbance), group D (PSC + sparse):
#   (1) baseline   : default PSC init  -> also fills the long-open P1 "T3 sparse" gap
#   (2) transfer   : D-T1 converged PSC weights as init (cross-task transfer test)
# 5 seeds each x 1B steps = 10 runs, sequential. ~1.18h/run -> ~12h total.
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
LOG="$LOGDIR/_e1_transfer.log"

SEEDS="42 123 456 789 1024"
TOTAL_TS=1000000000

# D-T1 converged PSC weights (5-seed mean), from _extract_d_t1_weights.py
T1_W="44.0986 2.3207 -0.0232 -0.5416 0.0608"
T1_B="22.6794"

echo "===============================================" | tee "$LOG"
echo "[E1] starting at $(date) PID=$$"                | tee -a "$LOG"
echo "[E1] T3-sparse: baseline (default init) + transfer (T1 winit)" | tee -a "$LOG"
echo "[E1] 2 configs x 5 seeds x 1B = 10 runs"        | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"

# --- Config 1: baseline T3 sparse, default PSC init (also = P1 gap fill) ---
for S in $SEEDS; do
    echo "" | tee -a "$LOG"
    echo "===== [E1-baseline] D T3-sparse default-init seed=$S start $(date) =====" | tee -a "$LOG"
    "$PYTHON" src/aerocat/scripts/run_ablation.py \
        --group D \
        --seeds $S \
        --task disturbance \
        --total-timesteps $TOTAL_TS \
        --num-envs 4096 \
        --num-steps 128 \
        --save-interval 10 \
        --log-interval 5 \
        >> "$LOG" 2>&1
    echo "===== [E1-baseline] seed=$S done $(date) exit=$? =====" | tee -a "$LOG"
done

# --- Config 2: transfer T3 sparse, D-T1 converged PSC init ---
for S in $SEEDS; do
    echo "" | tee -a "$LOG"
    echo "===== [E1-transfer] D T3-sparse T1-winit seed=$S start $(date) =====" | tee -a "$LOG"
    "$PYTHON" src/aerocat/scripts/run_ablation.py \
        --group D \
        --seeds $S \
        --task disturbance \
        --psc-init-w $T1_W \
        --psc-init-b $T1_B \
        --init-tag T1transfer \
        --total-timesteps $TOTAL_TS \
        --num-envs 4096 \
        --num-steps 128 \
        --save-interval 10 \
        --log-interval 5 \
        >> "$LOG" 2>&1
    echo "===== [E1-transfer] seed=$S done $(date) exit=$? =====" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"
echo "[E1] all runs finished at $(date)"              | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"
