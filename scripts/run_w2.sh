#!/bin/bash
# Exploratory controls and moment maps, ten seeds each (supplementary S5 and S13; Table III(b)):
#   T8  moment maps of the Doppler-angle map, MAP_TRANSFORM=dadelta | dagauss | daspread (_dadelta_s*, _dagauss_s*, _daspread_s*)
#   T7  single frame, LAST_FRAME_ONLY=1 (_lastframe_rd_s*)
#   T3  wider students, seeds 1 to 9 (_cap128_s*, _cap256_s*)
#   T4  vision-only student with a pooled encoder of matched size (_v3_visgap_s*)
#   T5  range-angle map without mean subtraction, RA_VARIANT=RAnosub (_ranosub_s*; maps from src/preprocess_ra_nosub.py)
# Parallel copies split the runs with LANE=k NLANES=n. Usage: bash scripts/run_w2.sh. Completed runs are skipped.
cd "$(dirname "$0")/.." || exit 1
L=cmkd_results/_w2_lane${LANE:-0}.log
say() { echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $L; }
done_run() { ls cmkd_results/$1/*/eval_results.npz >/dev/null 2>&1 || ls cmkd_results/$1/eval_results.npz >/dev/null 2>&1; }
LANE=${LANE:-0}; NLANES=${NLANES:-1}; k=0
lane_ok() { local r=$((k % NLANES)); k=$((k+1)); [ "$r" = "$LANE" ]; }   # deterministic round-robin over candidate runs
prov() { mkdir -p cmkd_results/$1; echo "$2" > cmkd_results/$1/ENV.txt; }
FILT="Epoch|Val ADBA|ATop|ADBA|params|Error|error|Traceback"
PY=${PYTHON:-python}
COMMON="--epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 \
  --eta_min 1e-06 --grad_clip 10.0 --seq_len 8 --num_pred 6 \
  --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 \
  --radar_h 128 --radar_w 64 --num_classes 64 --num_workers 16"
S9="--data_root dataset/scenario9 --train_csv train_seqs_RA_tr.csv --val_csv val_seqs_RA.csv --test_csv test_seqs_RA.csv"
W64="--feature_size 64 --gru_hidden 64"
say "W2 queue start"

# ---- T8: Doppler-angle moment maps (pre-specified encoder-level test) ----
for M in dadelta dagauss daspread; do
  for S in 42 1 2 3 4 5 6 7 8 9; do
    R=_${M}_s$S
    if lane_ok && ! done_run $R; then say "$R"; prov $R "T8 MAP_TRANSFORM=$M; DA only";
      MAP_TRANSFORM=$M $PY -u src/train_cmkd.py --phase 0 --radar_channels 1 --single_channel second --second_map DA $W64 $COMMON $S9 --seed $S --save_dir cmkd_results/$R 2>&1 | grep -E "$FILT" | tee -a $L; fi
  done
done

# ---- T7: single-frame control (last frame replicated), RA+RD ----
for S in 42 1 2 3 4 5 6 7 8 9; do
  R=_lastframe_rd_s$S
  if lane_ok && ! done_run $R; then say "$R"; prov $R "T7 LAST_FRAME_ONLY=1; RA+RD";
    LAST_FRAME_ONLY=1 $PY -u src/train_cmkd.py --phase 0 --radar_channels 2 --second_map RD $W64 $COMMON $S9 --seed $S --save_dir cmkd_results/$R 2>&1 | grep -E "$FILT" | tee -a $L; fi
done

# ---- T3: student width, seeds 1-9 (seed 42 exists as _cap128_s42/_cap256_s42) ----
for WD in 128 256; do
  for S in 1 2 3 4 5 6 7 8 9; do
    R=_cap${WD}_s$S
    if lane_ok && ! done_run $R; then say "$R"; prov $R "T3 feature_size=gru_hidden=$WD; RA+RD";
      $PY -u src/train_cmkd.py --phase 0 --radar_channels 2 --second_map RD --feature_size $WD --gru_hidden $WD $COMMON $S9 --seed $S --save_dir cmkd_results/$R 2>&1 | grep -E "$FILT" | tee -a $L; fi
  done
done

# ---- T4: parameter-matched vision-only student (GAP encoder) ----
for S in 42 1 2 3 4 5 6 7 8 9; do
  R=_v3_visgap_s$S
  if lane_ok && ! done_run $R; then say "$R"; prov $R "T4 vision-only GAP encoder";
    $PY -u src/train_vision_only.py $S dataset/scenario9 _v3_visgap train_seqs_RA_tr.csv 16 gap 2>&1 | grep -E "ep..|ATop|ADBA|params|Error|error|Traceback" | tee -a $L; fi
done

# ---- T5: range-angle only without slow-time mean subtraction (maps from src/preprocess_ra_nosub.py) ----
if [ -d dataset/scenario9/unit1/radar_RAnosub ] && [ "$(ls dataset/scenario9/unit1/radar_RAnosub | wc -l)" -ge 5964 ]; then
  for S in 42 1 2 3 4 5 6 7 8 9; do
    R=_ranosub_s$S
    if lane_ok && ! done_run $R; then say "$R"; prov $R "T5 RA_VARIANT=RAnosub; RA only";
      RA_VARIANT=RAnosub $PY -u src/train_cmkd.py --phase 0 --radar_channels 1 --single_channel RA --second_map DA $W64 $COMMON $S9 --seed $S --save_dir cmkd_results/$R 2>&1 | grep -E "$FILT" | tee -a $L; fi
  done
else
  say "T5 skipped: radar_RAnosub maps not present"
fi
say "W2 QUEUE DONE"
