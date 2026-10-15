#!/bin/bash
# Seeds 3 to 9 of one of two arms, chosen by the argument: logit distillation into the RA+DA student from the seed-42
# Doppler-angle teacher (logit: _v3_logit_s*, Table I), or the concatenation Doppler-angle teacher at phase 1 (concat:
# _v3cc_s*, supplementary S11). Waits until no other training runs. Usage: bash scripts/run_tail.sh logit|concat
cd "$(dirname "$0")/.." || exit 1
ARM=${1:?usage: run_tail.sh [logit|concat]}
LOG=cmkd_results/_tail_${ARM}.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }

say "waiting for the KD-variant queue to drain"
while pgrep -f "src/train_cmkd.py" >/dev/null; do sleep 60; done
say "queue drained, starting $ARM"

PY="${PYTHON:-python} -u src/train_cmkd.py --num_workers 16 --epochs 70 --radar_channels 2"
TCK=$(ls cmkd_results/_v3_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)

run () {                       # $1 tag  $2... args
  TAG=$1; shift
  if ls cmkd_results/${TAG}/phase*_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP $TAG" | tee -a $LOG; return
  fi
  rm -rf cmkd_results/${TAG}
  say "$TAG"
  $PY "$@" --save_dir cmkd_results/${TAG} > cmkd_results/${TAG}.log 2>&1
  # head, not tail: a phase-3 log prints the student first and a Teacher Reference
  # block after it, and tail would report the teacher's accuracy as the student's
  grep -aoE "ATop-1: [0-9.]+" cmkd_results/${TAG}.log | head -1 | tee -a $LOG
}

if [ "$ARM" = logit ]; then
  [ -f "$TCK" ] || { echo "FATAL: missing $TCK"; exit 1; }
  for S in 3 4 5 6 7 8 9; do
    run _v3_logit_s$S --phase 3 --teacher_ckpt $TCK --teacher_type cross_attn \
        --lambda1 1 --lambda2 0 --lambda3 0 --seed $S
  done
else
  for S in 3 4 5 6 7 8 9; do
    run _v3cc_s$S --phase 1 --teacher_type concat --seed $S
  done
fi
say "TAIL QUEUE ALL DONE ($ARM)"
