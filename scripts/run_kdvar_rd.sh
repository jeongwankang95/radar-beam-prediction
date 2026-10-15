#!/bin/bash
# Logit + feature (_v3rd_feat_s*) and logit + feature + temporal (_v3rd_full_s*) distillation into the reference student
# from the range-Doppler teacher _v3rd_teacher, seeds 8 and 9 (Table I; Fig. 3(b)). Seeds 1 and 2 come from
# scripts/run_all.sh; seeds 42 and 3 to 7 were trained with the same arguments without a launcher
# (scripts/train_command.py prints them). Usage: bash scripts/run_kdvar_rd.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
TCK=$(ls cmkd_results/_v3rd_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)
[ -f "$TCK" ] || { echo "FATAL: teacher checkpoint missing: $TCK"; exit 1; }

LOG=cmkd_results/_kdvar.log
PY="${PYTHON:-python} -u src/train_cmkd.py --phase 3 --radar_channels 2 --second_map RD \
    --teacher_ckpt $TCK --teacher_type cross_attn --num_workers 16 --epochs 70"
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "KD-variant queue start (deployed RA+RD student)"

run () {                     # $1 tag-stem  $2 lambda2  $3 lambda3  $4 seed
  TAG=_v3rd$1_s$4
  if ls cmkd_results/${TAG}/phase3_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG" | tee -a $LOG; return
  fi
  rm -rf cmkd_results/${TAG}
  say "$TAG  (lambda1=1 lambda2=$2 lambda3=$3)"
  $PY --lambda1 1 --lambda2 $2 --lambda3 $3 --seed $4 \
      --save_dir cmkd_results/${TAG} > cmkd_results/${TAG}.log 2>&1
  # phase-3 logs print the student first, then a "Teacher Reference" block; tail -1
  # would report the teacher (45.5) instead of the student (~41).
  awk "/Student Evaluation/{f=1} f&&/ATop-1:/{print;exit}" cmkd_results/${TAG}.log | tee -a $LOG
}

# seeds 8 and 9 (the other seeds: see the header)
for S in 8 9; do
  run _feat 1 0 $S
  run _full 1 1 $S
done
say "KD-VARIANT QUEUE ALL DONE"
