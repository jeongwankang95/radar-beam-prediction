#!/bin/zsh
# Teacher fusion at the task-loss stage (phase 1, Doppler-angle second map; supplementary S11): concatenation teachers
# _v3cc_s{42,1,2} and cross-attention teachers _v3ca_s{1,2} (the seed-42 cross-attention teacher is _v3_teacher). Seeds 3
# to 9 come from scripts/run_tail.sh (concatenation) and scripts/run_m4_original.sh (cross-attention).
# Usage: zsh scripts/run_fusion_ablation.sh [pid of a job to wait for]. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
EP=70
PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 0 --epochs $EP"
LOG=cmkd_results/_fusion.log

# ---- optionally wait for another job (its pid as the first argument), so that the two never share the GPU
WAIT_PID=${1:-0}
if [ "$WAIT_PID" != "0" ] && kill -0 $WAIT_PID 2>/dev/null; then
  echo "\n######## waiting for pid $WAIT_PID (RA-only control) $(date '+%m-%d %H:%M') ########" | tee -a $LOG
  while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
  echo "######## pid $WAIT_PID finished $(date '+%m-%d %H:%M') ########" | tee -a $LOG
fi

echo "\n######## fusion ablation start $(date '+%m-%d %H:%M') ########" | tee -a $LOG

run () {  # $1=tag  $2=teacher_type  $3=seed
  TAG=$1
  if ls cmkd_results/${TAG}/phase1_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG (done)" | tee -a $LOG; return
  fi
  rm -rf cmkd_results/${TAG}
  echo "==== $TAG  teacher_type=$2  seed=$3  $(date '+%m-%d %H:%M') ====" | tee -a $LOG
  eval $PY --phase 1 --teacher_type $2 --seed $3 --save_dir cmkd_results/${TAG} 2>&1 \
    | grep -E "Teacher params|Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
}

# informative order: gives a paired n=2 after the third run
run _v3cc_s42 concat     42   # pairs with existing cross_attn seed 42
run _v3cc_s1  concat      1
run _v3ca_s1  cross_attn   1   # completes the paired seed-1 comparison
run _v3cc_s2  concat       2
run _v3ca_s2  cross_attn   2

echo "\n######## fusion ablation ALL DONE $(date '+%m-%d %H:%M') ########" | tee -a $LOG
