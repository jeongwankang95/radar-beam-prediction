#!/bin/bash
# Scenarios 32 and 33 (supplementary S7): RA+RD and vision-only students trained on Scenario 32, seeds 42, 1 and 2
# (_s32rd_s*, _s32vis_s*); src/eval_daynight.py then scores them by day (Scenario 32) and by night (Scenario 33).
# Prepare both scenarios with src/s3x_prep.py first. Usage: bash scripts/run_g3.sh [workers, default 12]. Resumable.
set -u
cd "$(dirname "$0")/.."
# Run inside the Python environment of requirements.txt.
NW=${1:-12}
LOG=cmkd_results/_g3.log
say() { echo -e "\n######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "G3 START"
for S in 42 1 2; do
  T=_s32rd_s$S
  if ls cmkd_results/$T/phase0_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $T" | tee -a $LOG
  else
    rm -rf cmkd_results/$T
    say "$T (radar RA+RD, scenario32)"
    ${PYTHON:-python} -u src/train_cmkd.py --phase 0 --second_map RD --radar_channels 2 --seed $S \
        --data_root dataset/scenario32 --train_csv train_seqs_RA.csv \
        --num_workers $NW --epochs 70 --save_dir cmkd_results/$T 2>&1 \
      | grep -E "params|Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
  fi
done

for S in 42 1 2; do
  T=_s32vis_s$S
  if ls cmkd_results/$T/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $T" | tee -a $LOG
  else
    say "$T (vision-only, scenario32)"
    ${PYTHON:-python} -u src/train_vision_only.py $S dataset/scenario32 _s32vis train_seqs_RA.csv $NW 2>&1 \
      | grep -E "seed=|val ADBA|ATop-|ADBA:|early stop|Error|Traceback" | tee -a $LOG
  fi
done

say "G3 EVAL (day vs night)"
${PYTHON:-python} -u src/eval_daynight.py 2>&1 | tee -a $LOG
say "G3 ALL DONE"
