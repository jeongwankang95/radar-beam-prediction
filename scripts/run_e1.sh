#!/bin/bash
# Range-angle-only student distilled from the seed-42 Doppler-angle teacher _v3_teacher (enlarged-gap test, Sec. VII-B):
# _v3rakd_s<seed>, phase 3, logit distillation, the student reading the range-angle channel only. Seeds 42, 1 and 2 come
# from scripts/run_rakd_control.sh. Usage: bash scripts/run_e1.sh "3 4 5 6 7 8 9". Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
SEEDS=${1:?usage: run_e1.sh "3 4 5 6"}
LOG=cmkd_results/_e1.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "starting E1 seeds: $SEEDS"

TCK=$(ls cmkd_results/_v3_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)
[ -f "$TCK" ] || { echo "FATAL: missing teacher checkpoint $TCK" | tee -a $LOG; exit 1; }

PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 16 --epochs 70 \
    --phase 3 --radar_channels 2 --student_radar_channels 1 \
    --teacher_ckpt $TCK --teacher_type cross_attn \
    --lambda1 1 --lambda2 0 --lambda3 0"

for S in $SEEDS; do
  TAG=_v3rakd_s$S
  if ls cmkd_results/${TAG}/phase*_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG" | tee -a $LOG; continue
  fi
  rm -rf cmkd_results/${TAG}
  say "$TAG"
  $PY --seed $S --save_dir cmkd_results/${TAG} > cmkd_results/${TAG}.log 2>&1
  # head, not tail: a phase-3 log prints the student first and a Teacher Reference
  # block after it, and tail would report the teacher's accuracy as the student's
  grep -aoE "ATop-1: [0-9.]+" cmkd_results/${TAG}.log | head -1 | tee -a $LOG
done
say "E1 QUEUE ALL DONE"
