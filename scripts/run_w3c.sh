#!/bin/bash
# Continuation of scripts/run_w3.sh with the same lanes and claims: two-stem encoder, matched-size vision-only student,
# wider students and the DA-spread moment map. Usage: LANE=0 bash scripts/run_w3c.sh
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
LANE=${LANE:-0}; L=cmkd_results/_w3c_lane$LANE.log; mkdir -p cmkd_results/.claims
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $L; }
done_run() { ls cmkd_results/$1/*/eval_results.npz >/dev/null 2>&1 || ls cmkd_results/$1/eval_results.npz >/dev/null 2>&1; }
running() { pgrep -f -- "save_dir cmkd_results/$1\$" >/dev/null 2>&1 || pgrep -f -- "src/train_vision_only.py ${1##*_s} dataset/scenario9 ${1%_s*} " >/dev/null 2>&1; }
MAXGPU=${MAXGPU:-5}
gpu_procs() { local a b; a=$(pgrep -af "src/train_cmkd.py" | grep -o "save_dir cmkd_results/_[a-z][^ ]*" | sort -u | wc -l); b=$(pgrep -af "src/train_vision_only.py" | grep -o "src/train_vision_only.py [0-9][0-9]*" | sort -u | wc -l); echo $((a+b)); }
wait_slot() { while [ "$(gpu_procs)" -ge "$MAXGPU" ]; do sleep $((60 + LANE * 7)); done; }
claim() { mkdir cmkd_results/.claims/$1 2>/dev/null; }
take() { ! done_run $1 && ! running $1 && { wait_slot; ! done_run $1 && ! running $1 && claim $1; }; }
prov() { echo "$2" > cmkd_results/$1/ENV.txt 2>/dev/null || { mkdir -p cmkd_results/$1; echo "$2" > cmkd_results/$1/ENV.txt; }; }
FILT="Val ADBA|ATop|ADBA|params|Error|error|Traceback"
PY=${PYTHON:-python}
COMMON="--epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 --eta_min 1e-06 --grad_clip 10.0 --seq_len 8 --num_pred 6 --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 --radar_h 128 --radar_w 64 --num_classes 64 --num_workers 16"
S9="--data_root dataset/scenario9 --train_csv train_seqs_RA_tr.csv --val_csv val_seqs_RA.csv --test_csv test_seqs_RA.csv"
W64="--feature_size 64 --gru_hidden 64"
TCK=$(ls cmkd_results/_v3rd_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)
KD="--phase 3 --radar_channels 2 --second_map RD --lambda2 0.0 --lambda3 0.0 --temperature 3.0 --teacher_type cross_attn --teacher_ckpt $TCK"
run() { local R=$1 P=$2; shift 2
  if take $R; then say "$R"; prov $R "$P"; $PY -u src/train_cmkd.py "$@" --save_dir cmkd_results/$R 2>&1 | grep --line-buffered -E "$FILT" | tee -a $L; fi; }
say "W3c lane $LANE start (T6 -> T4 -> T3-128 -> optional)"
# T6
for S in 42 1 2 3 4 5 6 7 8 9; do run _twostem_rd_s$S "T6 --two_stem RA+RD (moved from the second machine)" --phase 0 --radar_channels 2 --second_map RD --two_stem $W64 $COMMON $S9 --seed $S; done
# T4
for S in 42 1 2 3 4 5 6 7 8 9; do R=_v3_visgap_s$S
  if take $R; then say "$R"; prov $R "T4 vision-only GAP encoder (moved from the second machine)"; $PY -u src/train_vision_only.py $S dataset/scenario9 _v3_visgap train_seqs_RA_tr.csv 16 gap 2>&1 | grep --line-buffered -E "ep..|ATop|ADBA|params|Error|error|Traceback" | tee -a $L; fi; done
# T3 width 128
for S in 1 2 3 4 5 6 7 8 9; do run _cap128_s$S "T3 feature_size=gru_hidden=128; RA+RD (moved from the second machine)" --phase 0 --radar_channels 2 --second_map RD --feature_size 128 --gru_hidden 128 $COMMON $S9 --seed $S; done
# optional
for S in 1 2 3 4 5 6 7 8 9; do run _cap256_s$S "T3 (optional) feature_size=gru_hidden=256; RA+RD" --phase 0 --radar_channels 2 --second_map RD --feature_size 256 --gru_hidden 256 $COMMON $S9 --seed $S; done
for S in 42 1 2 3 4 5 6 7 8 9; do MAP_TRANSFORM=daspread run _daspread_s$S "T8 (optional) MAP_TRANSFORM=daspread; DA only" --phase 0 --radar_channels 1 --single_channel second --second_map DA $W64 $COMMON $S9 --seed $S; done
say "W3c lane $LANE DONE"
