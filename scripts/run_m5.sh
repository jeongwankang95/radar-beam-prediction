#!/bin/bash
# Range-Doppler map with its rows permuted (Table III(a)): the reference student trained and tested with
# SCRAMBLE_DOPPLER=1, which permutes the rows of the second map (for the range-Doppler map, its range axis) per frame,
# SCRAMBLE_SEED=1000+seed; seeds 42, 1, 2, 3 and 4 (_m5scr_s*). Usage: bash scripts/run_m5.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python} -u src/train_cmkd.py --phase 0 --num_workers 16 --epochs 70 --second_map RD --radar_channels 2"
LOG=cmkd_results/_m5scramble.log
echo "######## M5 start $(date '+%m-%d %H:%M') ########" | tee -a $LOG
for S in 42 1 2 3 4; do
  TAG=_m5scr_s${S}
  if ls cmkd_results/${TAG}/phase0_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG (done)" | tee -a $LOG; continue
  fi
  rm -rf cmkd_results/${TAG}
  echo "" | tee -a $LOG
  echo "-------- $TAG  $(date '+%m-%d %H:%M') --------" | tee -a $LOG
  SCRAMBLE_DOPPLER=1 SCRAMBLE_SEED=$((1000+S)) $PY --seed $S --save_dir cmkd_results/${TAG} 2>&1 \
    | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
done
echo "" | tee -a $LOG
echo "######## M5 ALL DONE $(date '+%m-%d %H:%M') ########" | tee -a $LOG
