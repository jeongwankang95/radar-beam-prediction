#!/bin/bash
# Seeds 3 to 9 of the range-angle-only and RA+DA students (_c1ra_s*, _c1da_s*; phase 0; Tables I and II), then phase 2
# of the Doppler-angle teachers _v3ca_s1 and _v3ca_s2, whose phase 1 comes from scripts/run_fusion_ablation.sh.
# Usage: bash scripts/run_c1.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
export SCRAMBLE_DOPPLER=0
PY="${PYTHON:-python} -u src/train_cmkd.py --phase 0 --num_workers 16 --epochs 70"
LOG=cmkd_results/_c1.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "C1 symmetry queue start"

# ---- range-angle only, seeds 3 to 9
for S in 3 4 5 6 7 8 9; do
  TAG=_c1ra_s${S}
  if ls cmkd_results/${TAG}/phase0_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG" | tee -a $LOG; continue
  fi
  rm -rf cmkd_results/${TAG}
  say "$TAG  (RA only)"
  $PY --radar_channels 1 --seed $S --save_dir cmkd_results/${TAG} \
      > cmkd_results/${TAG}.log 2>&1
  grep -aoE "Test ATop-1[^|]*|ATop-1: [0-9.]+" cmkd_results/${TAG}.log | tail -1 | tee -a $LOG
done

# ---- RA+DA, seeds 3 to 9
for S in 3 4 5 6 7 8 9; do
  TAG=_c1da_s${S}
  if ls cmkd_results/${TAG}/phase0_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG" | tee -a $LOG; continue
  fi
  rm -rf cmkd_results/${TAG}
  say "$TAG  (RA+DA)"
  $PY --radar_channels 2 --second_map DA --seed $S --save_dir cmkd_results/${TAG} \
      > cmkd_results/${TAG}.log 2>&1
  grep -aoE "Test ATop-1[^|]*|ATop-1: [0-9.]+" cmkd_results/${TAG}.log | tail -1 | tee -a $LOG
done

# ---- phase 2 of the Doppler-angle teachers of seeds 1 and 2
for D in _v3ca_s1 _v3ca_s2; do
  if ls cmkd_results/$D/phase2_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $D phase2" | tee -a $LOG; continue
  fi
  P1=$(ls -t cmkd_results/$D/phase1_*/teacher_phase1.pth 2>/dev/null | head -1)
  S=${D##*_s}
  [ -z "$P1" ] && { echo "NO phase1 ckpt for $D" | tee -a $LOG; continue; }
  say "$D phase2 (self-KD, seed $S)"
  $PY --phase 2 --teacher_ckpt "$P1" --second_map DA --seed $S \
      --save_dir cmkd_results/$D > cmkd_results/${D}_ph2.log 2>&1
  grep -aoE "Test ATop-1[^|]*|ATop-1: [0-9.]+" cmkd_results/${D}_ph2.log | tail -1 | tee -a $LOG
done

say "C1 queue ALL DONE"
