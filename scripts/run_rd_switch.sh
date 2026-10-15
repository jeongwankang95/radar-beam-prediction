#!/bin/zsh
# The seed-42 range-Doppler vision-radar teacher _v3rd_teacher (phase 1, then phase 2), the distillation source of every
# RA+RD distillation arm, and logit distillation from it into the reference student, seeds 42, 1 and 2 (_v3rdkd_s*).
# Usage: zsh scripts/run_rd_switch.sh. Completed stages are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 0 --epochs 70 --second_map RD"
TDIR=cmkd_results/_v3rd_teacher
LOG=cmkd_results/_rdswitch.log
say() { echo "\n######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "RD switch start"

# ---- teacher phase 1 (task loss) ----
if ! ls $TDIR/phase1_*/teacher_phase1.pth >/dev/null 2>&1; then
  say "TEACHER phase1 (RA+RD)"
  eval $PY --phase 1 --save_dir $TDIR 2>&1 \
    | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
else echo "SKIP teacher phase1" | tee -a $LOG; fi
P1=$(ls -t $TDIR/phase1_*/teacher_phase1.pth 2>/dev/null | head -1)
echo "P1=$P1" | tee -a $LOG

# ---- teacher phase 2 (self-distillation) ----
if ! ls $TDIR/phase2_*/teacher_selfkd.pth >/dev/null 2>&1; then
  say "TEACHER phase2 (self-KD)"
  eval $PY --phase 2 --teacher_ckpt "$P1" --save_dir $TDIR 2>&1 \
    | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
else echo "SKIP teacher phase2" | tee -a $LOG; fi
P2=$(ls -t $TDIR/phase2_*/teacher_selfkd.pth 2>/dev/null | head -1)
echo "P2=$P2" | tee -a $LOG

# ---- distilled students, 3 seeds (lambda1=1 only, the deployed objective) ----
for S in 42 1 2; do
  TAG=_v3rdkd_s${S}
  if ls cmkd_results/${TAG}/phase3_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG (done)" | tee -a $LOG; continue
  fi
  rm -rf cmkd_results/${TAG}
  say "$TAG (RA+RD student, logit KD)"
  eval $PY --phase 3 --teacher_ckpt "$P2" --lambda1 1 --lambda2 0 --lambda3 0 \
       --seed $S --save_dir cmkd_results/${TAG} 2>&1 \
    | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
done

say "RD switch ALL DONE"
