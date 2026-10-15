#!/bin/bash
# Test-set prediction dumps from trained checkpoints (cmkd_results/<run>/...), in the format of the dumps shipped in
# results/. Run from the repository root after training; each dump keeps runs it already holds, so it can be rerun.
# Output: cmkd_results/predictions/. The analysis scripts read results/; copy a regenerated dump there to use it.
# The order of seeds inside a dump fixes the bootstrap draw, so intervals match the paper exactly only with the shipped
# dumps (the nine arms of Table II are listed below in the shipped order).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}
O=cmkd_results/predictions; mkdir -p $O
S9="--data_root dataset/scenario9 --test_csv test_seqs_RA.csv"
P3=(--leaf "phase3_*/student_cmkd_best.pth")                        # distilled students
seeds() { local p=$1; shift; local s=""; for x in "$@"; do s+="${p}${x},"; done; echo "${s%,}"; }
TEN="42 1 2 3 4 5 6 7 8 9"

# ---- Table II: the nine predictors (results/_pred_dump.npz)
$PY src/w2_pred.py --name "RA only" --tags "_v3_noKD_RA,_v3s1_noKD_RA,_v3s2_noKD_RA,$(seeds _c1ra_s 3 4 5 6 7 8 9)" --chan first --second_map DA $S9 --out $O/_pd_1.npz
$PY src/w2_pred.py --name "DA only" --tags "$(seeds _m6DAonly_s $TEN)" --chan second --second_map DA $S9 --out $O/_pd_2.npz
$PY src/w2_pred.py --name "RD only" --tags "$(seeds _m6RDonly_s $TEN)" --chan second --second_map RD $S9 --out $O/_pd_3.npz
$PY src/w2_pred.py --name "RA+DA" --tags "_v3_noKD_RADA,_v3s1_noKD_RADA,_v3s2_noKD_RADA,$(seeds _c1da_s 3 4 5 6 7 8 9)" --chan both --second_map DA $S9 --out $O/_pd_4.npz
$PY src/w2_pred.py --name "RA+RD" --tags "$(seeds _v3rd_s $TEN)" --chan both --second_map RD $S9 --out $O/_pd_5.npz
$PY src/w2_pred.py --name "RA+DA+RD" --tags "$(seeds _m4_3ch_s $TEN)" --chan three --second_map DA --extra_map RD $S9 --out $O/_pd_6.npz
$PY src/w2_pred.py --name "RA+RD+KD" --tags "$(seeds _v3rdkd_s $TEN)" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_pd_7.npz
$PY src/w2_pred.py --name "Vision-only" --tags "$(seeds _v3_visiononly_s $TEN)" --model vision --leaf "vision_best.pth" $S9 --out $O/_pd_8.npz
$PY src/w2_pred.py --name "Teacher" --tags "_v3_teacher,$(seeds _v3ca_s 1 2 3 4 5 6 7 8 9)" --model teacher --second_map DA --leaf "phase2_*/teacher_selfkd.pth" $S9 --out $O/_pd_9.npz
$PY src/merge_dumps.py $O/_pred_dump.npz $O/_pd_1.npz $O/_pd_2.npz $O/_pd_3.npz $O/_pd_4.npz $O/_pd_5.npz $O/_pd_6.npz $O/_pd_7.npz $O/_pd_8.npz $O/_pd_9.npz

# ---- distillation variants and teachers
$PY src/w2_pred.py --name rdteacher --tags "_v3rd_teacher,_teacher_rd_s1,_teacher_rd_s2" --model teacher --second_map RD --leaf "phase2_*/teacher_selfkd.pth" $S9 --out $O/_w2pred_rdteacher.npz
$PY src/w2_pred.py --name rdfeat --tags "_v3rd_feat_s*" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_w2pred_rdfeat.npz
$PY src/w2_pred.py --name rdfull --tags "_v3rd_full_s*" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_w2pred_rdfull.npz
$PY src/w2_pred.py --name cckd --tags "_cckd_s*" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_w2pred_cckd.npz
$PY src/w2_pred.py --name rakd --tags "_v3rakd_s*" --chan first --second_map DA "${P3[@]}" $S9 --out $O/_w2pred_rakd.npz
$PY src/w2_pred.py --name rdkdT1 --tags "_v3rdkdT1_s*" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_w2pred_rdkdT1.npz
$PY src/w2_pred.py --name rdkdT2 --tags "_v3rdkdT2_s*" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_w2pred_rdkdT2.npz
for LAM in 0p1 0p3 3p0 10p0; do
  $PY src/w2_pred.py --name lam$LAM --tags "_lam${LAM}_rdkd_s*" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_w2pred_lam$LAM.npz
done

# ---- learning curve and the radar-only teacher
for P in 10 20 45; do
  $PY src/w2_pred.py --name lc$P --tags "_lc${P}_rd_s*" --chan both --second_map RD $S9 --out $O/_w2pred_lc$P.npz
  $PY src/w2_pred.py --name lc${P}kd --tags "_lc${P}_rdkd_s*" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_w2pred_lc${P}kd.npz
done
for P in 10 20; do
  $PY src/w2_pred.py --name lc${P}kdR --tags "_lc${P}_rdkdR_s*" --chan both --second_map RD "${P3[@]}" $S9 --out $O/_w2pred_lc${P}kdR.npz
done

# ---- encoder and input controls (the map transforms are read by src/DataFeed.py from the environment)
$PY src/w2_pred.py --name twostem --tags "_twostem_rd_s*" --chan both --second_map RD --kwargs '{"two_stem": true}' $S9 --out $O/_w2pred_twostem.npz
for WD in 128 256; do
  $PY src/w2_pred.py --name cap$WD --tags "_cap${WD}_s*" --chan both --second_map RD --kwargs "{\"feature_size\": $WD, \"gru_hidden\": $WD}" $S9 --out $O/_w2pred_cap$WD.npz
done
$PY src/w2_pred.py --name visgap --tags "_v3_visgap_s*" --model vision --leaf "vision_best.pth" --kwargs '{"gap": true}' $S9 --out $O/_w2pred_visgap.npz
LAST_FRAME_ONLY=1 $PY src/w2_pred.py --name lastframe --tags "_lastframe_rd_s*" --chan both --second_map RD $S9 --out $O/_w2pred_lastframe.npz
RA_VARIANT=RAnosub $PY src/w2_pred.py --name ranosub --tags "_ranosub_s*" --chan first --second_map DA $S9 --out $O/_w2pred_ranosub.npz
MAP_TRANSFORM=marginal $PY src/w2_pred.py --name damarg --tags "_gridDAmarg_s*" --chan second --second_map DA $S9 --out $O/_w2pred_damarg.npz
for M in dadelta dagauss daspread dagaussfl; do
  MAP_TRANSFORM=$M $PY src/w2_pred.py --name $M --tags "_${M}_s*" --chan second --second_map DA $S9 --out $O/_w2pred_$M.npz
done
# DA rows permuted, re-scored on one draw of permutations shared by the ten seeds (supplementary S13). The draw depends
# on SCRAMBLE_SEED and on the loader's worker count, so a regenerated dump differs slightly from the shipped one.
SCRAMBLE_DOPPLER=1 SCRAMBLE_SEED=2000 $PY src/w2_pred.py --name dascr --tags "_dascr_s*" --chan second --second_map DA $S9 --out $O/_w2pred_dascr.npz
echo "PREDICTIONS DONE -> $O"
