#!/bin/bash
# Three blocks. (A) Logit distillation at temperature 1 into the reference student, seeds 1 to 9 (_kd_T1_s*; seed 42
# comes from scripts/run_all.sh; supplementary S10), after a one-epoch probe that checks the arguments against
# _kd_T1_s42. (B) The all-zero second map, seeds 42, 1, 2, 3, 4 and 9 (scripts/run_mcap.sh). (C) MLP-projector seed 9,
# which did not finish. Usage: bash scripts/run_rev0922.sh. Runs with an eval_results.npz are skipped.
set -u
cd "$(dirname "$0")/.." || exit 1
L=cmkd_results/_rev0922.log
TCK=$(ls cmkd_results/_v3rd_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)
[ -f "$TCK" ] || { echo "FATAL: teacher checkpoint missing: $TCK" | tee -a $L; exit 1; }
say(){ echo "######## $1  $(date '+%m-%d %H:%M') ########" | tee -a $L; }
done_run(){ ls cmkd_results/$1/*/eval_results.npz >/dev/null 2>&1; }
FILT="########|params|best|ATop-|ADBA:|Early stopping|Error|Traceback"
COMMON="--data_root dataset/scenario9 --epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 --eta_min 1e-06 --grad_clip 10.0 --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 --gru_layers_student 1 --gru_layers_teacher 2 --num_heads_mha 8 --num_heads_ca 4 --num_classes 64 --num_workers 16"
T1ARGS="--phase 3 --radar_channels 2 --second_map RD --lambda1 1.0 --lambda2 0.0 --lambda3 0.0 --temperature 1.0 --teacher_type cross_attn --teacher_ckpt $TCK"

say "REV0922 QUEUE START  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ---- probe: 1 epoch, then diff args.json against _kd_T1_s42 (keys present in both; seed/save_dir/epochs ignored)
if ! done_run _kd_T1_s1; then
  rm -rf cmkd_results/_probe_T1
  say "probe _probe_T1 (1 epoch, args check)"
  ${PYTHON:-python} -u src/train_cmkd.py $T1ARGS $COMMON --epochs 1 --seed 1 --save_dir cmkd_results/_probe_T1 2>&1 | tail -2 | tee -a $L
  ${PYTHON:-python} - <<'PY' 2>&1 | tee -a $L
import json, glob, sys
a = json.load(open(glob.glob("cmkd_results/_kd_T1_s42/*/args.json")[0]))
b = json.load(open(glob.glob("cmkd_results/_probe_T1/*/args.json")[0]))
ign = {"seed", "save_dir", "epochs"}
both = [k for k in a if k in b and k not in ign and a[k] != b[k]]
only = sorted((set(a) ^ set(b)) - ign)
print("PROBE diff on shared keys:", {k: (a[k], b[k]) for k in both} or "NONE")
print("PROBE keys present on one side only (schema drift, informational):", {k: (a.get(k, "<absent>"), b.get(k, "<absent>")) for k in only} or "NONE")
sys.exit(1 if both else 0)
PY
  if [ "${PIPESTATUS[0]}" != "0" ]; then say "PROBE ARGS MISMATCH - ABORT"; exit 1; fi
  rm -rf cmkd_results/_probe_T1
fi

# ---- Block A
for S in 1 2 3 4 5 6 7 8 9; do
  R=_kd_T1_s$S
  if done_run $R; then echo "SKIP $R" | tee -a $L; continue; fi
  rm -rf cmkd_results/$R
  say "$R"
  ${PYTHON:-python} -u src/train_cmkd.py $T1ARGS $COMMON --seed $S --save_dir cmkd_results/$R 2>&1 | grep -E "$FILT" | tee -a $L
done

# ---- Block B
if [ -d cmkd_results/_mcap_ra_s9 ] && ! done_run _mcap_ra_s9; then
  mv cmkd_results/_mcap_ra_s9 cmkd_results/_mcap_ra_s9.incomplete_0831 && echo "moved incomplete _mcap_ra_s9 aside" | tee -a $L
fi
say "matched-capacity: run_mcap.sh 42 1 2 3 4 9"
bash scripts/run_mcap.sh 42 1 2 3 4 9 2>&1 | grep -E "$FILT" | tee -a $L

# ---- Block C
if [ -d cmkd_results/_v3rd_featmlp_s9 ] && ! done_run _v3rd_featmlp_s9; then
  mv cmkd_results/_v3rd_featmlp_s9 cmkd_results/_v3rd_featmlp_s9.incomplete_0901 && echo "moved incomplete _v3rd_featmlp_s9 aside" | tee -a $L
fi
if ! done_run _v3rd_featmlp_s9; then
  say "_v3rd_featmlp_s9"
  ${PYTHON:-python} -u src/train_cmkd.py --phase 3 --radar_channels 2 --second_map RD --feat_projector mlp \
    --lambda1 1.0 --lambda2 1.0 --lambda3 0.0 --temperature 3.0 --teacher_type cross_attn --teacher_ckpt $TCK \
    $COMMON --seed 9 --save_dir cmkd_results/_v3rd_featmlp_s9 2>&1 | grep -E "$FILT" | tee -a $L
fi

say "REV0922 QUEUE DONE"
