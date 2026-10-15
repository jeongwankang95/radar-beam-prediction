#!/bin/bash
# Three-map student, [RA, DA, RD] as three channels (_m4_3ch_s<seed>; phase 0; Tables I and II).
# Usage: bash scripts/run_m4_3ch.sh "42 1 2 3 4 5 6 7 8 9". Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
SEEDS=${1:?usage: run_m4_3ch.sh "42 1 2"}
LOG=cmkd_results/_m4_3ch.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }
say "starting 3-channel [RA,DA,RD] seeds: $SEEDS"

PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 16 --epochs 70 \
    --phase 0 --radar_channels 3 --second_map DA --extra_map RD"

for S in $SEEDS; do
  TAG=_m4_3ch_s$S
  if ls cmkd_results/${TAG}/phase*_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG" | tee -a $LOG; continue
  fi
  rm -rf cmkd_results/${TAG}
  say "$TAG"
  $PY --seed $S --save_dir cmkd_results/${TAG} > cmkd_results/${TAG}.log 2>&1
  grep -aoE "ATop-1: [0-9.]+" cmkd_results/${TAG}.log | head -1 | tee -a $LOG
done
say "M4 3-CHANNEL QUEUE DONE"
