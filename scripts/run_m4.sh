#!/bin/bash
# Logit + feature (_v3_feat_s*) and logit + feature + temporal (_v3_full_s*) distillation into the RA+DA student from the
# seed-42 Doppler-angle teacher _v3_teacher, seeds 1 to 9 (Table I; seed 42 from scripts/run_valsplit_retrain.sh), then
# scripts/run_kdvar_rd.sh. Usage: bash scripts/run_m4.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
TCK=$(ls cmkd_results/_v3_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)
[ -f "$TCK" ] || { echo "FATAL: teacher checkpoint missing: $TCK"; exit 1; }

LOG=cmkd_results/_kdvar_da.log
PY="${PYTHON:-python} -u src/train_cmkd.py --phase 3 --radar_channels 2 \
    --teacher_ckpt $TCK --teacher_type cross_attn --num_workers 16 --epochs 70"
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "DA-student KD-variant queue start (seeds 1-9; s42 already exists)"

run () {                     # $1 stem  $2 lambda2  $3 lambda3  $4 seed
  TAG=_v3_$1_s$4
  if ls cmkd_results/${TAG}/phase3_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG" | tee -a $LOG; return
  fi
  rm -rf cmkd_results/${TAG}
  say "$TAG  (lambda1=1 lambda2=$2 lambda3=$3)"
  $PY --lambda1 1 --lambda2 $2 --lambda3 $3 --seed $4 \
      --save_dir cmkd_results/${TAG} > cmkd_results/${TAG}.log 2>&1
  # phase-3 logs print the student first, then a "Teacher Reference" block; tail -1
  # would report the teacher instead of the student.
  awk "/Student Evaluation/{f=1} f&&/ATop-1:/{print;exit}" cmkd_results/${TAG}.log | tee -a $LOG
}

# alternate the two variants per seed, so that an interrupted queue leaves both arms on the same seeds
for S in 1 2 3 4 5 6 7 8 9; do
  run feat 1 0 $S
  run full 1 1 $S
done
say "DA KD-VARIANT QUEUE ALL DONE"

# then the reference-student variants, seeds 8 and 9
exec bash scripts/run_kdvar_rd.sh
