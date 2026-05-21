#!/bin/bash
set -e
cd ""/bin/../.."
exec > experiments/_logs/_h5_perdim.log 2>&1
echo "===== H5 starting at $(date) PID=$$ ====="
python -u src/aerocat/scripts/run_dr_eval_perdim.py \
  --groups A B C D E F \
  --seeds 42 123 456 789 1024 \
  --dims nominal mass wind turb sensor actuator init_state collision all_l1 \
  --num-episodes 100 --num-envs 256 \
  --output experiments/_h5_perdim/results.json
echo "===== H5 finished at $(date) ====="
