#!/bin/bash
# Axis transformations of Table III(a), ten seeds each, on the single-map student: the Doppler-angle map collapsed to its
# per-angle mean (_gridDAmarg_s*, MAP_TRANSFORM=marginal) or binarized (_gridDAbin_s*, MAP_TRANSFORM=binary), and the
# range-Doppler map collapsed to its per-velocity mean (_gridRDmarg_s*, MAP_TRANSFORM=marginal).
# Usage: bash scripts/run_grid.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python} -u src/train_cmkd.py --phase 0 --num_workers 16 --epochs 70 \
    --radar_channels 1 --single_channel second"
LOG=cmkd_results/_grid.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "representation grid start"
run_arm () {                       # $1 tag  $2 map  $3 transform
  for S in 42 1 2 3 4 5 6 7 8 9; do
    TAG=_grid$1_s${S}
    if ls cmkd_results/${TAG}/phase0_*/eval_results.npz >/dev/null 2>&1; then
      echo "SKIP $TAG" | tee -a $LOG; continue
    fi
    rm -rf cmkd_results/${TAG}
    say "$TAG  ($2, $3)"
    MAP_TRANSFORM=$3 $PY --second_map $2 --seed $S --save_dir cmkd_results/${TAG} \
        > cmkd_results/${TAG}.log 2>&1
    grep -aoE "ATop-1: [0-9.]+" cmkd_results/${TAG}.log | tail -1 | tee -a $LOG
  done
}
run_arm DAmarg DA marginal
run_arm DAbin  DA binary
run_arm RDmarg RD marginal
say "GRID ALL DONE"
