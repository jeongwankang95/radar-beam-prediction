#!/bin/bash
# Range-Doppler runs around the reference student. Needs the range-Doppler teacher _v3rd_teacher (scripts/run_rd_switch.sh).
# Usage: bash scripts/run_all.sh [workers, default 16]. Trains, in order:
#   _v3rdkd_s{42,1,2}                     logit distillation into the reference student (skipped if already trained)
#   _teacher_rd_s{1,2}                    two further range-Doppler teachers, phase 1 then phase 2 (teacher-seed replication)
#   _v3rd_s{3..9}, _v3rdkd_s{3..9}        the reference student without and with logit distillation, seeds 3 to 9
#   _cap128_s42, _cap256_s42              wider students, seed 42
#   _kd_from_phase1_s42                   distillation from the phase-1 teacher (not used in the paper)
#   _cc_teacher_rd, _cckd_s{42,1,2}       concatenation teacher (phase 1 then phase 2) and the students distilled from it
#   _v3rd_feat_s{1,2}, _v3rd_full_s{1,2}  logit + feature and logit + feature + temporal distillation
#   _kd_T{1,2,5}_s42                      logit distillation at temperatures 1, 2 and 5
# Restartable: a run whose eval_results.npz exists is skipped.
set -u
cd "$(dirname "$0")/.."
# Run inside the Python environment of requirements.txt.
NW=${1:-16}
PY="python -u src/train_cmkd.py --num_workers $NW --epochs 70"
LOG=cmkd_results/_run_all.log
mkdir -p cmkd_results
say() { echo -e "\n######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }
done_p() { ls cmkd_results/$1/phase*_*/eval_results.npz >/dev/null 2>&1; }
run() {   # run <tag> <args...>
  local tag=$1; shift
  if done_p "$tag"; then echo "SKIP $tag" | tee -a $LOG; return; fi
  rm -rf cmkd_results/$tag
  say "$tag"
  eval $PY "$@" --save_dir cmkd_results/$tag 2>&1 \
    | grep -E "params|Val ADBA|ATop-|ADBA:|Early stopping|Error|Traceback" | tee -a $LOG
}

say "QUEUE START  workers=$NW  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ---- logit distillation into the reference student, seeds 42, 1 and 2
TD=cmkd_results/_v3rd_teacher
P2=$(ls -t $TD/phase2_*/teacher_selfkd.pth 2>/dev/null | head -1)
P1=$(ls -t $TD/phase1_*/teacher_phase1.pth 2>/dev/null | head -1)
echo "RD teacher: ${P2:-MISSING}" | tee -a $LOG
for S in 42 1 2; do
  [ -n "$P2" ] && run _v3rdkd_s$S --phase 3 --second_map RD --teacher_ckpt "$P2" \
                       --lambda1 1 --lambda2 0 --lambda3 0 --seed $S
done

# ---- teachers 1 and 2 of the teacher-seed replication
for S in 1 2; do
  T=cmkd_results/_teacher_rd_s$S
  if ! ls $T/phase2_*/teacher_selfkd.pth >/dev/null 2>&1; then
    say "teacher seed $S (RA+RD): phase1 -> phase2"
    eval $PY --phase 1 --second_map RD --seed $S --save_dir $T 2>&1 \
      | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error" | tee -a $LOG
    Q1=$(ls -t $T/phase1_*/teacher_phase1.pth 2>/dev/null | head -1)
    [ -n "$Q1" ] && eval $PY --phase 2 --second_map RD --seed $S --teacher_ckpt "$Q1" \
        --save_dir $T 2>&1 | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error" | tee -a $LOG
  else echo "SKIP teacher seed $S" | tee -a $LOG; fi
done

# ---- the reference student without and with logit distillation, seeds 3 to 9
for S in 3 4 5 6 7 8 9; do
  run _v3rd_s$S --phase 0 --second_map RD --radar_channels 2 --seed $S
  [ -n "$P2" ] && run _v3rdkd_s$S --phase 3 --second_map RD --teacher_ckpt "$P2" \
                       --lambda1 1 --lambda2 0 --lambda3 0 --seed $S
done

# ---- wider students (feature and GRU width 128 and 256), seed 42
for W in 128 256; do
  run _cap${W}_s42 --phase 0 --second_map RD --radar_channels 2 --seed 42 \
                   --feature_size $W --gru_hidden $W
done

# ---- distillation from the phase-1 teacher (not used in the paper)
[ -n "$P1" ] && run _kd_from_phase1_s42 --phase 3 --second_map RD --teacher_ckpt "$P1" \
                     --lambda1 1 --lambda2 0 --lambda3 0 --seed 42

# ---- concatenation teacher (phase 1, then phase 2) and the students distilled from it
CC=cmkd_results/_cc_teacher_rd
if ! ls $CC/phase2_*/teacher_selfkd.pth >/dev/null 2>&1; then
  say "concat teacher (RA+RD): phase1 -> phase2"
  eval $PY --phase 1 --teacher_type concat --second_map RD --seed 42 --save_dir $CC 2>&1 \
    | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error" | tee -a $LOG
  C1=$(ls -t $CC/phase1_*/teacher_phase1.pth 2>/dev/null | head -1)
  [ -n "$C1" ] && eval $PY --phase 2 --teacher_type concat --second_map RD --seed 42 \
      --teacher_ckpt "$C1" --save_dir $CC 2>&1 \
      | grep -E "Val ADBA|ATop-|ADBA:|Early stopping|Error" | tee -a $LOG
fi
C2=$(ls -t $CC/phase2_*/teacher_selfkd.pth 2>/dev/null | head -1)
for S in 42 1 2; do
  [ -n "$C2" ] && run _cckd_s$S --phase 3 --teacher_type concat --second_map RD \
                       --teacher_ckpt "$C2" --lambda1 1 --lambda2 0 --lambda3 0 --seed $S
done

# ---- logit + feature and logit + feature + temporal distillation, seeds 1 and 2; temperatures 1, 2 and 5 at seed 42
for S in 1 2; do
  [ -n "$P2" ] && run _v3rd_feat_s$S --phase 3 --second_map RD --teacher_ckpt "$P2" \
                       --lambda1 1 --lambda2 1 --lambda3 0 --seed $S
  [ -n "$P2" ] && run _v3rd_full_s$S --phase 3 --second_map RD --teacher_ckpt "$P2" \
                       --lambda1 1 --lambda2 1 --lambda3 1 --seed $S
done
for G in 1 2 5; do
  [ -n "$P2" ] && run _kd_T${G}_s42 --phase 3 --second_map RD --teacher_ckpt "$P2" \
                       --lambda1 1 --lambda2 0 --lambda3 0 --temperature $G --seed 42
done

say "QUEUE DONE"
