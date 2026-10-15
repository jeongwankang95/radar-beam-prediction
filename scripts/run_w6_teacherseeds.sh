#!/bin/bash
# Teacher-seed replication (prespecification/TEACHER_SEED_REPLICATION_PRESPEC_2026-09-30.md; Sec. VII-A, supplementary
# S10): logit distillation into the reference student from range-Doppler teacher 1 or 2 (_teacher_rd_s1, _teacher_rd_s2),
# ten seeds (_v3rdkdT<TT>_s*). Parallel copies claim runs through cmkd_results/.claims.
# Usage: TT=1 LANE=0 bash scripts/run_w6_teacherseeds.sh
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TT=${TT:-1}; LANE=${LANE:-0}; PY=${PYTHON:-python}
case $TT in 1) TCK=$(ls cmkd_results/_teacher_rd_s1/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1);; 2) TCK=$(ls cmkd_results/_teacher_rd_s2/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1);; esac
L=cmkd_results/_w6_T${TT}_lane$LANE.log; mkdir -p cmkd_results/.claims
COMMON="--epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 --eta_min 1e-06 --grad_clip 10.0 --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 --radar_h 128 --radar_w 64 --num_classes 64 --num_workers 16"
S9="--data_root dataset/scenario9 --train_csv train_seqs_RA_tr.csv --val_csv val_seqs_RA.csv --test_csv test_seqs_RA.csv"
KD="--phase 3 --radar_channels 2 --second_map RD --lambda1 1.0 --lambda2 0.0 --lambda3 0.0 --temperature 3.0 --teacher_type cross_attn --teacher_ckpt $TCK"
echo "######## W6 T$TT lane $LANE start $(date -u '+%m-%d %H:%M') ########" >> $L
for S in 42 1 2 3 4 5 6 7 8 9; do
  R=_v3rdkdT${TT}_s$S
  ls cmkd_results/$R/*/eval_results.npz >/dev/null 2>&1 && continue
  mkdir cmkd_results/.claims/$R 2>/dev/null || continue
  mkdir -p cmkd_results/$R
  echo "W6 teacher-seed replication: logit KD from $TCK; seed $S; $(date -u)" > cmkd_results/$R/ENV.txt
  echo "######## $R $(date -u '+%m-%d %H:%M') ########" >> $L
  $PY -u src/train_cmkd.py $KD $COMMON $S9 --seed $S --save_dir cmkd_results/$R 2>&1 | grep --line-buffered -E "Loaded teacher|Val ADBA|ATop-1|Early stopping|Error|Traceback" >> $L
done
echo "######## W6 T$TT lane $LANE DONE $(date -u '+%m-%d %H:%M') ########" >> $L
