#!/bin/bash
# Doppler-resolution sweep (supplementary S13; a separate five-seed campaign): the Doppler-angle-only student with the
# map's 128 velocity rows mean-pooled into B groups and repeated back over them (MAP_TRANSFORM=pool<B>), seeds 42, 1, 2,
# 3 and 4 (_gridDApool<B>_s*); B = 128 is the identity. Usage: bash scripts/run_pool.sh 1 2 4 8 16 32 64 128
cd "$(dirname "$0")/.." || exit 1
for B in "$@"; do
  for S in 42 1 2 3 4; do
    [ -d cmkd_results/_gridDApool${B}_s$S ] && continue
    echo "######## _gridDApool${B}_s$S  $(date +%m-%d\ %H:%M) ########"
    MAP_TRANSFORM=pool${B} ${PYTHON:-python} -u src/train_cmkd.py \
      --phase 0 --data_root dataset/scenario9 \
      --radar_channels 1 --second_map DA --single_channel second \
      --epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 \
      --eta_min 1e-06 --grad_clip 10.0 \
      --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 \
      --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 \
      --radar_h 128 --radar_w 64 --num_classes 64 --num_workers 16 \
      --train_csv train_seqs_RA_tr.csv --val_csv val_seqs_RA.csv --test_csv test_seqs_RA.csv \
      --seed $S --save_dir cmkd_results/_gridDApool${B}_s$S
  done
done
echo POOL_DONE
