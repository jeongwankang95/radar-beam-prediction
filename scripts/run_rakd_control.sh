#!/bin/zsh
# Range-angle-only student distilled from the seed-42 Doppler-angle teacher (enlarged-gap test, Sec. VII-B), seeds 42, 1
# and 2 (_v3rakd_s*); seeds 3 to 9 come from scripts/run_e1.sh. Usage: zsh scripts/run_rakd_control.sh. Completed runs
# are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
EP=70
PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 0 --epochs $EP"
P2=$(ls -t cmkd_results/_v3_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | head -1)
LOG=cmkd_results/_rakd.log
echo "\n######## RA-only+KD control start $(date '+%m-%d %H:%M')  teacher=$P2 ########" | tee -a $LOG
for S in 42 1 2; do
  TAG=_v3rakd_s${S}
  if ls cmkd_results/${TAG}/phase*_*/eval_results.npz >/dev/null 2>&1; then echo "SKIP $TAG" | tee -a $LOG; continue; fi
  rm -rf cmkd_results/${TAG}
  echo "==== $TAG (RA-only, 1-ch, +logit KD) $(date '+%m-%d %H:%M') ====" | tee -a $LOG
  eval $PY --phase 3 --radar_channels 2 --student_radar_channels 1 --teacher_ckpt "$P2" \
       --lambda1 1 --lambda2 0 --lambda3 0 --seed $S --save_dir cmkd_results/${TAG} 2>&1 \
    | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
done
echo "######## RA-only+KD control ALL DONE $(date '+%m-%d %H:%M') ########" | tee -a $LOG
