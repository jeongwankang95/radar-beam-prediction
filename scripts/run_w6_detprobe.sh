#!/bin/bash
# Determinism check (supplementary S13): two identical two-epoch runs of the Doppler-angle-only student (A, B) and one
# through the identity pooling transform MAP_TRANSFORM=pool128 (C), seed 42, in parallel.
# Usage: bash scripts/run_w6_detprobe.sh
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PYTHON:-python}
COMMON="--epochs 2 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 --eta_min 1e-06 --grad_clip 10.0 --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 --radar_h 128 --radar_w 64 --num_classes 64 --num_workers 16"
S9="--data_root dataset/scenario9 --train_csv train_seqs_RA_tr.csv --val_csv val_seqs_RA.csv --test_csv test_seqs_RA.csv"
ARGS="--phase 0 --radar_channels 1 --single_channel second --second_map DA --seed 42"
for R in A B; do $PY -u src/train_cmkd.py $ARGS $COMMON $S9 --save_dir cmkd_results/_det_probe_$R > cmkd_results/_det_probe_$R.log 2>&1 & done
MAP_TRANSFORM=pool128 $PY -u src/train_cmkd.py $ARGS $COMMON $S9 --save_dir cmkd_results/_det_probe_C > cmkd_results/_det_probe_C.log 2>&1 &
wait; echo "DET PROBE DONE $(date -u)" >> cmkd_results/_det_probe_A.log
