#!/bin/bash
# All-zero second map (Sec. V-B; supplementary S14): per seed, the range-angle-only student (_mcap_ra_s<seed>) and the
# two-channel student whose second channel is set to zero (_mcap_ra0_s<seed>, --zero_second), which shares the graph and
# all 96,624 parameters of the RA+RD student. Usage: bash scripts/run_mcap.sh 42 1 2 3 4 5 6 7 8 9
cd "$(dirname "$0")/.." || exit 1
common="--phase 0 --data_root dataset/scenario9 \
  --epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 \
  --eta_min 1e-06 --grad_clip 10.0 \
  --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 \
  --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 \
  --radar_h 128 --radar_w 64 --num_classes 64 --num_workers 16 \
  --train_csv train_seqs_RA_tr.csv --val_csv val_seqs_RA.csv --test_csv test_seqs_RA.csv"

for S in "$@"; do
  # one channel, the published range-angle configuration
  if [ ! -d cmkd_results/_mcap_ra_s$S ]; then
    echo "######## _mcap_ra_s$S  $(date +%m-%d\ %H:%M) ########"
    ${PYTHON:-python} -u src/train_cmkd.py $common \
      --radar_channels 1 --second_map DA \
      --seed $S --save_dir cmkd_results/_mcap_ra_s$S
  fi
  # two channels, second one zeroed: same graph and parameter count as RA+RD
  if [ ! -d cmkd_results/_mcap_ra0_s$S ]; then
    echo "######## _mcap_ra0_s$S  $(date +%m-%d\ %H:%M) ########"
    ${PYTHON:-python} -u src/train_cmkd.py $common \
      --radar_channels 2 --second_map RD --zero_second \
      --seed $S --save_dir cmkd_results/_mcap_ra0_s$S
  fi
done
echo MCAP_DONE
