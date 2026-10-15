#!/bin/zsh
# Seed-42 runs on the pass-level validation split: range-angle-only (_v3_noKD_RA) and RA+DA (_v3_noKD_RADA) students, the
# Doppler-angle vision-radar teacher _v3_teacher (phase 1, then phase 2), and the RA+DA student distilled from it with
# logit (_v3_logit), logit + feature (_v3_feat) and logit + feature + temporal (_v3_full) terms.
# Usage: zsh scripts/run_valsplit_retrain.sh. Completed stages are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
EP=70
PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 0 --epochs $EP"
TDIR=cmkd_results/_v3_teacher
LOG=cmkd_results/_v3_retrain.log
mkdir -p cmkd_results
say() { echo "\n########## $1  $(date '+%m-%d %H:%M') ##########" | tee -a $LOG; }

# ---- no-KD baselines (phase 0: task loss only, no teacher needed) ----
run_p0 () {  # $1=name  $2=radar_channels
  if ls cmkd_results/_v3_$1/phase0_*/eval_results.npz >/dev/null 2>&1; then echo "SKIP $1" | tee -a $LOG; return; fi
  rm -rf cmkd_results/_v3_$1
  say "$1 (phase0, radar_channels=$2)"
  eval $PY --phase 0 --radar_channels $2 --save_dir cmkd_results/_v3_$1 2>&1 \
    | grep -E "Train:|Val loss|ATop-|ADBA|Early stopping|Error|Traceback" | tee -a $LOG
}
run_p0 noKD_RA   1
run_p0 noKD_RADA 2

# ---- teacher: phase 1 (task) -> phase 2 (self-distillation) ----
if ! ls $TDIR/phase1_*/teacher_phase1.pth >/dev/null 2>&1; then
  say "TEACHER phase1"
  eval $PY --phase 1 --save_dir $TDIR 2>&1 \
    | grep -E "Train:|Val loss|ATop-|ADBA|Early stopping|Error|Traceback" | tee -a $LOG
else echo "SKIP teacher phase1" | tee -a $LOG; fi
P1=$(ls -t $TDIR/phase1_*/teacher_phase1.pth 2>/dev/null | head -1)
echo "P1=$P1" | tee -a $LOG

if ! ls $TDIR/phase2_*/teacher_selfkd.pth >/dev/null 2>&1; then
  say "TEACHER phase2 (self-KD)"
  eval $PY --phase 2 --teacher_ckpt "$P1" --save_dir $TDIR 2>&1 \
    | grep -E "Train:|Val loss|ATop-|ADBA|Early stopping|Error|Traceback" | tee -a $LOG
else echo "SKIP teacher phase2" | tee -a $LOG; fi
P2=$(ls -t $TDIR/phase2_*/teacher_selfkd.pth 2>/dev/null | head -1)
echo "P2=$P2" | tee -a $LOG

# ---- KD students (phase 3) ----
run_p3 () {  # $1=name  $2=l1 $3=l2 $4=l3
  if ls cmkd_results/_v3_$1/phase3_*/eval_results.npz >/dev/null 2>&1; then echo "SKIP $1" | tee -a $LOG; return; fi
  rm -rf cmkd_results/_v3_$1
  say "$1 (phase3, lambda=$2,$3,$4)"
  eval $PY --phase 3 --teacher_ckpt "$P2" --lambda1 $2 --lambda2 $3 --lambda3 $4 \
    --save_dir cmkd_results/_v3_$1 2>&1 \
    | grep -E "Train:|Val loss|ATop-|ADBA|Early stopping|Error|Traceback" | tee -a $LOG
}
run_p3 logit 1 0 0   # logit distillation
run_p3 feat  1 1 0
run_p3 full  1 1 1

say "ALL DONE"
