#!/bin/zsh
# The reference RA+RD student without distillation, seeds 42, 1 and 2 (_v3rd_s*; phase 0); seeds 3 to 9 come from
# scripts/run_all.sh. Usage: zsh scripts/run_rd_ablation.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 0 --epochs 70"
LOG=cmkd_results/_rd.log

echo "\n######## RA+RD vs RA+DA ablation start $(date '+%m-%d %H:%M') ########" | tee -a $LOG
for S in 42 1 2; do
  TAG=_v3rd_s${S}
  if ls cmkd_results/${TAG}/phase0_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG (done)" | tee -a $LOG; continue
  fi
  rm -rf cmkd_results/${TAG}
  echo "==== $TAG (RA+RD, 2-ch, no KD) $(date '+%m-%d %H:%M') ====" | tee -a $LOG
  eval $PY --phase 0 --radar_channels 2 --second_map RD --seed $S \
       --save_dir cmkd_results/${TAG} 2>&1 \
    | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
done
echo "######## RA+RD ablation ALL DONE $(date '+%m-%d %H:%M') ########" | tee -a $LOG
