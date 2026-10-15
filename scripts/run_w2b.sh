#!/bin/bash
# Moment map of Table III(b): the Doppler-angle map's Gaussian moments plus each column's floor, MAP_TRANSFORM=dagaussfl,
# ten seeds (_dagaussfl_s*). Parallel copies split the runs with LANE=k NLANES=n. Usage: bash scripts/run_w2b.sh
cd "$(dirname "$0")/.." || exit 1
L=cmkd_results/_w2b_lane${LANE:-0}.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $L; }
done_run() { ls cmkd_results/$1/*/eval_results.npz >/dev/null 2>&1; }
LANE=${LANE:-0}; NLANES=${NLANES:-1}; k=0
lane_ok() { local r=$((k % NLANES)); k=$((k+1)); [ "$r" = "$LANE" ]; }
prov() { mkdir -p cmkd_results/$1; echo "$2" > cmkd_results/$1/ENV.txt; }
FILT="Epoch|Val ADBA|ATop|ADBA|params|Error|error|Traceback"
PY=${PYTHON:-python}
COMMON="--epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 --eta_min 1e-06 --grad_clip 10.0 --seq_len 8 --num_pred 6 --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 --radar_h 128 --radar_w 64 --num_classes 64 --num_workers 16 --feature_size 64 --gru_hidden 64"
S9="--data_root dataset/scenario9 --train_csv train_seqs_RA_tr.csv --val_csv val_seqs_RA.csv --test_csv test_seqs_RA.csv"
say "W2b start"
for S in 42 1 2 3 4 5 6 7 8 9; do
  R=_dagaussfl_s$S
  if lane_ok && ! done_run $R; then say "$R"; prov $R "T8b MAP_TRANSFORM=dagaussfl (Gaussian moments + column floor); DA only";
    MAP_TRANSFORM=dagaussfl $PY -u src/train_cmkd.py --phase 0 --radar_channels 1 --single_channel second --second_map DA $COMMON $S9 --seed $S --save_dir cmkd_results/$R 2>&1 | grep -E "$FILT" | tee -a $L; fi
done
say "W2b DONE"
