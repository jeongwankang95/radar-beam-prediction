#!/bin/bash
# Seeds 3 to 9 of the vision-only student (_v3_visiononly_s*) and of the Doppler-angle vision-radar teacher, phase 1 then
# phase 2 (_v3ca_s*), for Tables I and II. Usage: bash scripts/run_m4_original.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
export SCRAMBLE_DOPPLER=0 MAP_TRANSFORM=none
LOG=cmkd_results/_m4.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $LOG; }
PY=${PYTHON:-python}

say "M4 power queue start"

# ---- vision-only, the missing seeds ----
for S in 3 4 5 6 7 8 9; do
  D=cmkd_results/_v3_visiononly_s${S}
  if [ -f $D/eval_results.npz ]; then echo "SKIP vision s$S" | tee -a $LOG; continue; fi
  rm -rf $D
  say "vision-only s$S"
  $PY -u src/train_vision_only.py $S dataset/scenario9 _v3_visiononly train_seqs_RA_tr.csv 16 \
      > cmkd_results/_vis_s${S}.log 2>&1
  grep -aoE "ATop-1: [0-9.]+" cmkd_results/_vis_s${S}.log | tail -1 | tee -a $LOG
done

# ---- cross-attention teacher, both phases, the missing seeds ----
for S in 3 4 5 6 7 8 9; do
  D=cmkd_results/_v3ca_s${S}
  if ls $D/phase2_*/eval_results.npz >/dev/null 2>&1; then
    echo "SKIP teacher s$S" | tee -a $LOG; continue
  fi
  if ! ls $D/phase1_*/teacher_phase1.pth >/dev/null 2>&1; then
    say "teacher s$S phase 1"
    $PY -u src/train_cmkd.py --phase 1 --teacher_type cross_attn --num_workers 16 \
        --epochs 70 --seed $S --save_dir $D > cmkd_results/_te1_s${S}.log 2>&1
  fi
  P1=$(ls -t $D/phase1_*/teacher_phase1.pth 2>/dev/null | head -1)
  [ -z "$P1" ] && { echo "NO phase1 ckpt for s$S" | tee -a $LOG; continue; }
  say "teacher s$S phase 2 (self-KD)"
  $PY -u src/train_cmkd.py --phase 2 --teacher_ckpt "$P1" --teacher_type cross_attn \
      --num_workers 16 --epochs 70 --seed $S --save_dir $D \
      > cmkd_results/_te2_s${S}.log 2>&1
  grep -aoE "ATop-1: [0-9.]+" cmkd_results/_te2_s${S}.log | tail -1 | tee -a $LOG
done
say "M4 ALL DONE"
