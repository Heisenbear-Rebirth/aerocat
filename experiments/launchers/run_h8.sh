#!/bin/bash
set -e
cd ""/bin/../.."
exec > experiments/_logs/_h8_perdim_t3.log 2>&1
echo "===== H8 starting at $(date) PID=$$ ====="
echo "===== H8 = H5 per-dim decomposition on T3 (disturbance task) ====="
echo "===== Groups: A C F (dense T3 full) + D (E1 T3-sparse baseline) ====="
python -u src/aerocat/scripts/run_dr_eval_perdim.py \
  --task disturbance \
  --groups A C F D \
  --seeds 42 123 456 789 1024 \
  --dims nominal mass wind turb sensor actuator init_state collision all_l1 \
  --num-episodes 100 --num-envs 256 \
  --output experiments/_h8_perdim_t3/results.json
echo "===== H8 finished at $(date) ====="
