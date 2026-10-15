#!/bin/zsh
# Vision-only student, seeds 42, 1 and 2 (_v3_visiononly_s*; seeds 3 to 9 come from scripts/run_m4_original.sh). Starts
# once scripts/run_multiseed.sh has finished (it waits for that script's log). Usage: zsh scripts/run_visiononly_chain.sh
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
LOG=cmkd_results/_visiononly.log
echo "\n######## vision-only chain armed $(date '+%m-%d %H:%M'), waiting for multi-seed ########" | tee -a $LOG
until grep -q "multi-seed ALL DONE" cmkd_results/_ms_retrain.log 2>/dev/null; do sleep 300; done
echo "==== multi-seed done; starting vision-only $(date '+%m-%d %H:%M') ====" | tee -a $LOG
for S in 42 1 2; do
  if ls cmkd_results/_v3_visiononly_s${S}/eval_results.npz >/dev/null 2>&1; then echo "SKIP vision s$S" | tee -a $LOG; continue; fi
  echo "==== vision-only seed $S  $(date '+%m-%d %H:%M') ====" | tee -a $LOG
  ${PYTHON:-python} -u src/train_vision_only.py $S 2>&1 \
    | grep -E "Train |val ADBA|ATop-|ADBA:|early stop|Error|Traceback" | tee -a $LOG
done
echo "######## vision-only ALL DONE $(date '+%m-%d %H:%M') ########" | tee -a $LOG
