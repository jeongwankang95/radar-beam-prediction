#!/bin/bash
# Scenario 31 (the second base station): window preparation, then the three range-Doppler screening seeds of
# supplementary Sec. S6. Resumable. Extract the DeepSense 6G Scenario 31 archive into dataset/scenario31 first.
# The ten-seed replication arms (_s31ra_s*, _s31rd_s* and, after the label correction, _s31fix_ra_s*, _s31fix_rd_s*) have
# no launcher; scripts/train_command.py prints their training commands, and src/s31_test.py scores them.
set -u
cd "$(dirname "$0")/.." || exit 1
NW=${1:-12}
L=cmkd_results/_s31.log
mkdir -p cmkd_results
say() { echo -e "\n######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $L; }

say "S31 START"
[ -d dataset/scenario31/unit1 ] || { echo "dataset/scenario31/unit1 not found: extract the DeepSense 6G Scenario 31 archive there"; exit 1; }
if [ ! -f dataset/scenario31/test_seqs_RA.csv ]; then
  say "prep"
  ${PYTHON:-python} -u src/s31_prep.py 2>&1 | tee -a $L
  grep -q S31_PREP_DONE $L || { say "PREP FAILED"; exit 1; }
fi

for S in 42 1 2; do
  T=_s31rd_s$S
  if ls cmkd_results/$T/phase0_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $T" | tee -a $L
  else
    rm -rf cmkd_results/$T
    say "$T (radar RA+RD, scenario31)"
    ${PYTHON:-python} -u src/train_cmkd.py --phase 0 --second_map RD --radar_channels 2 --seed $S \
        --data_root dataset/scenario31 --train_csv train_seqs_RA.csv \
        --num_workers $NW --epochs 70 --save_dir cmkd_results/$T 2>&1 \
      | grep -E "params|Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $L
  fi
done
say "S31 ALL DONE"
