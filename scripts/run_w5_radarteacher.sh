#!/bin/bash
# Learning-curve students distilled from a radar-only teacher (prespecification/RADAR_TEACHER_CONTROL_PRESPEC_2026-09-29.md;
# Sec. VII-B, supplementary S11): the reference student trained on P passes (P = 10 or 20) with logit distillation from
# the seed-42 RA+RD student _v3rd_s42 (--teacher_type radar), seeds 42, 1, 2, 3 and 4 (_lc<P>_rdkdR_s*). Starts the five
# runs in the background. Usage: P=10 bash scripts/run_w5_radarteacher.sh
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=${P:-10}; PY=${PYTHON:-python}
COMMON="--epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 --eta_min 1e-06 --grad_clip 10.0 --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 --radar_h 128 --radar_w 64 --num_classes 64 --num_workers 16"
S9="--data_root dataset/scenario9 --val_csv val_seqs_RA.csv --test_csv test_seqs_RA.csv"
KDR="--phase 3 --radar_channels 2 --second_map RD --lambda1 1.0 --lambda2 0.0 --lambda3 0.0 --temperature 3.0 --teacher_type radar --teacher_ckpt $(ls cmkd_results/_v3rd_s42/phase0_*/radar_baseline.pth 2>/dev/null | tail -1)"
for S in 42 1 2 3 4; do
  R=_lc${P}_rdkdR_s$S
  if ls cmkd_results/$R/*/eval_results.npz >/dev/null 2>&1; then echo "skip $R (done)"; continue; fi
  mkdir -p cmkd_results/$R
  echo "W5 radar-only teacher control: 91-pass RA+RD student s42 as logit-KD source; train passes p$P; seed $S; $(date -u)" > cmkd_results/$R/ENV.txt
  setsid nohup $PY -u src/train_cmkd.py $KDR $COMMON $S9 --train_csv train_seqs_RA_tr_p$P.csv --seed $S --save_dir cmkd_results/$R > cmkd_results/$R.log 2>&1 < /dev/null &
  sleep 5
done
echo "launched p$P"
