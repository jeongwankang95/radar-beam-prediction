#!/bin/bash
# Regenerates, from the stored predictions, every table, figure and interval of the paper that does not need a trained
# checkpoint, then compares the outputs with results/reference/ and results/. Needs DeepSense 6G Scenario 9 in
# dataset/scenario9 (or CMKD_DATA): the measured beam powers for every loss, the radar maps and GPS for the tracker and
# the Doppler-centroid test. Takes under an hour on a laptop CPU, most of it the 82 bootstrap contrasts and the tracker.
# Run from the repository root; outputs go to results/analysis/ and figures/out/.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
bash analysis/run_bootstraps.sh
for s in distillation_share distillation_share_pooled interval_constructions pred_dump_summary s31_penalty_sensitivity \
         slope_interval doppler_centroid_test flat_peak extract_frames road_ekf_tracker merge_teacher_seeds \
         table1 table2 table3 subgroup training_metrics text_values; do
  echo "== analysis/$s.py"
  $PY analysis/$s.py
done
mkdir -p figures/out
$PY figures/perslot.py
$PY figures/fig_perslot.py figures/out/fig_perslot_tikz.tex
$PY figures/fig_lc.py figures/out/fig_lc_tikz.tex
$PY figures/make_forest.py figures/out/fig_forest_tikz.tex
$PY figures/fig5_calib.py figures/out/fig5_calib_elements.tex
$PY analysis/check_reference.py
