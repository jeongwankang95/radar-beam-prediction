#!/bin/zsh
# Seeds 1 and 2 of the range-angle-only (_v3s<seed>_noKD_RA) and RA+DA (_v3s<seed>_noKD_RADA) students and of logit
# distillation into the RA+DA student from the seed-42 Doppler-angle teacher (_v3s<seed>_logit); seed 42 comes from
# scripts/run_valsplit_retrain.sh. Usage: zsh scripts/run_multiseed.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
EP=70
PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 0 --epochs $EP"
P2=$(ls -t cmkd_results/_v3_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | head -1)
LOG=cmkd_results/_ms_retrain.log
echo "\n######## multi-seed start $(date '+%m-%d %H:%M')  teacher=$P2 ########" | tee -a $LOG

run () {  # $1=tag $2=phase-args...
  local tag=$1; shift
  # inline glob (not a variable), so that zsh expands it and completed runs are skipped
  if ls cmkd_results/${tag}/phase*_*/eval_results.npz >/dev/null 2>&1; then echo "SKIP $tag" | tee -a $LOG; return; fi
  rm -rf cmkd_results/${tag}
  echo "==== $tag  $(date '+%m-%d %H:%M') ====" | tee -a $LOG
  eval $PY "$@" --save_dir cmkd_results/${tag} 2>&1 \
    | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
}

for S in 1 2; do
  run _v3s${S}_noKD_RA    --phase 0 --radar_channels 1 --seed $S
  run _v3s${S}_noKD_RADA  --phase 0 --radar_channels 2 --seed $S
  run _v3s${S}_logit      --phase 3 --radar_channels 2 --teacher_ckpt "$P2" --lambda1 1 --lambda2 0 --lambda3 0 --seed $S
done
echo "######## multi-seed ALL DONE $(date '+%m-%d %H:%M') ########" | tee -a $LOG
