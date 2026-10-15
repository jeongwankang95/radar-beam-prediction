#!/bin/bash
# Single-map students that replace the range-angle map: Doppler-angle only (_m6DAonly_s*) and range-Doppler only
# (_m6RDonly_s*), ten seeds each (phase 0; Tables I and II). Usage: bash scripts/run_m6.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
export SCRAMBLE_DOPPLER=0
PY="${PYTHON:-python} -u src/train_cmkd.py --phase 0 --num_workers 16 --epochs 70 \
    --radar_channels 1 --single_channel second"
LOG=cmkd_results/_m6.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "M6 replacement-arm queue start"
for MAP in DA RD; do
  for S in 42 1 2 3 4 5 6 7 8 9; do
    TAG=_m6${MAP}only_s${S}
    if ls cmkd_results/${TAG}/phase0_*/eval_results.npz >/dev/null 2>&1; then
      echo "SKIP $TAG" | tee -a $LOG; continue
    fi
    rm -rf cmkd_results/${TAG}
    say "$TAG  (${MAP} alone, replacing range--angle)"
    $PY --second_map $MAP --seed $S --save_dir cmkd_results/${TAG} \
        > cmkd_results/${TAG}.log 2>&1
    grep -aoE "ATop-1: [0-9.]+" cmkd_results/${TAG}.log | tail -1 | tee -a $LOG
  done
done
say "M6 ALL DONE"
