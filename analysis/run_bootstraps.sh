#!/bin/bash
# Every paired contrast behind the paper's intervals, from the prediction dumps in results/ and the measured powers of
# dataset/scenario9 (set CMKD_DATA if the dataset lives elsewhere). Each line writes results/analysis/_boot_<label>.json;
# analysis/check_reference.py compares them with results/reference/. Cluster bootstrap: 28 test passes resampled as
# clusters, seeds resampled within each arm, 20,000 replicates, generator seed 0. Run from the repository root.
set -e
cd "$(dirname "$0")/.."
B() { ${PYTHON:-python} analysis/bootstrap_contrast.py "$@" | tail -4; }

# --- Doppler-resolved input (Sec. V-B, Fig. 3a, supplementary S14)
B --new "_pred_dump.npz:DA only" --ref "_pred_dump.npz:RA only" --label DA_vs_RA
B --new "_pred_dump.npz:RA+RD" --ref "_pred_dump.npz:RA only" --label doppler_check
B --new "_pred_dump.npz:RA+DA" --ref "_pred_dump.npz:RA only" --label nonselected_RADA_vs_RA
B --new "_pred_dump.npz:DA only" --ref "_pred_dump.npz:RD only" --label r5chk_DAonly_vs_RDonly
B --new "_pred_dump.npz:DA only" --ref "_pred_dump.npz:RA+RD" --label r5chk_DAonly_vs_pair
B --new "_pred_dump.npz:RA+RD" --ref "_pred_dump.npz:RA+DA" --label r5chk_RD_vs_DA_pairing
B --new "_pred_dump.npz:RD only" --ref "_pred_dump.npz:RA only" --label r5chk_RDonly_vs_RA
B --new "_pred_dump.npz:RA+RD" --ref "_pred_dump.npz:RD only" --label r5chk_pair_vs_RDonly
B --new "_pred_dump.npz:RA+DA+RD" --ref "_pred_dump.npz:RA only" --label su7_threemaps_vs_RA
B --new "_pred_dump.npz:RA+DA+RD" --ref "_pred_dump.npz:RA+RD" --label threemaps_vs_RD

# --- Distillation (Secs. V-B, VII-A and VII-B, Fig. 3b, supplementary S10 and S11)
B --new "_w2pred_cap128.npz:cap128" --ref "_pred_dump.npz:Teacher" --label cap128_vs_teacher
B --new "_w2pred_cckd.npz:cckd" --ref "_pred_dump.npz:RA+RD" --label cckd_vs_nokd
B --new "_pred_dump.npz:Teacher" --ref "_w2pred_rdteacher.npz:rdteacher" --label dateacher_vs_rdteacher
B --new "_pred_dump.npz:RA+RD+KD" --ref "_pred_dump.npz:RA+RD" --label full_kd_vs_nokd
B --new "_pred_dump.npz:RA+RD+KD" --ref "_pred_dump.npz:RA+RD" --label kd_check
B --new "_w2pred_rdfeat.npz:rdfeat" --ref "_pred_dump.npz:RA+RD" --label kd_feat
B --new "_w2pred_rdfull.npz:rdfull" --ref "_pred_dump.npz:RA+RD" --label kd_full
B --new "_pred_dump.npz:RA+RD+KD" --ref "_pred_dump.npz:RA+RD" --label kd_logit
B --new "_w2pred_lam0p1.npz:lam0p1" --ref "_pred_dump.npz:RA+RD" --label lam0p1_vs_nokd
B --new "_w2pred_lam0p3.npz:lam0p3" --ref "_pred_dump.npz:RA+RD+KD" --label lam0p3_vs_lam1
B --new "_w2pred_lam0p3.npz:lam0p3" --ref "_pred_dump.npz:RA+RD" --label lam0p3_vs_nokd
B --new "_w2pred_lam10p0.npz:lam10p0" --ref "_pred_dump.npz:RA+RD" --label lam10p0_vs_nokd
B --new "_w2pred_lam3p0.npz:lam3p0" --ref "_pred_dump.npz:RA+RD+KD" --label lam3p0_vs_lam1
B --new "_w2pred_lam3p0.npz:lam3p0" --ref "_pred_dump.npz:RA+RD" --label lam3p0_vs_nokd
B --new "_w2pred_rakd.npz:rakd" --ref "_pred_dump.npz:RA only" --label rakd_vs_RA
B --new "_w2pred_rdteacher.npz:rdteacher" --ref "_pred_dump.npz:RA+RD" --label rdteacher_vs_student
B --new "_pred_dump.npz:Vision-only" --ref "_w2pred_rdteacher.npz:rdteacher" --label vision_vs_rdteacher
B --new "_pred_dump.npz:RA+RD+KD" --ref "_pred_dump.npz:RA+RD" --label w6_G0
B --new "_w2pred_rdkdT1.npz:rdkdT1" --ref "_pred_dump.npz:RA+RD" --label w6_G1
B --new "_w2pred_rdkdT2.npz:rdkdT2" --ref "_pred_dump.npz:RA+RD" --label w6_G2
B --new "_w2pred_rdkdAll.npz:rdkdAll" --ref "_pred_dump.npz:RA+RD" --label w6_Gall

# --- Learning curve and radar-only teacher (Sec. VII-B, Fig. 6, Table S13)
B --new "_w2pred_lc10kd.npz:lc10kd" --ref "_w2pred_lc10.npz:lc10" --label lc10_kd_vs_nokd
B --new "_w2pred_lc10.npz:lc10" --ref "_pred_dump.npz:RA+RD" --label lc10_vs_full
B --new "_w2pred_lc20kd.npz:lc20kd" --ref "_w2pred_lc20.npz:lc20" --label lc20_kd_vs_nokd
B --new "_w2pred_lc20.npz:lc20" --ref "_pred_dump.npz:RA+RD" --label lc20_vs_full
B --new "_w2pred_lc45kd.npz:lc45kd" --ref "_w2pred_lc45.npz:lc45" --label lc45_kd_vs_nokd
B --new "_w2pred_lc45.npz:lc45" --ref "_pred_dump.npz:RA+RD" --label lc45_vs_full
B --new "_w2pred_lc10kdR.npz:lc10kdR" --ref "_w2pred_lc10.npz:lc10" --label w5_R10
B --new "_w2pred_lc20kdR.npz:lc20kdR" --ref "_w2pred_lc20.npz:lc20" --label w5_R20
B --new "_w2pred_lc10kd.npz:lc10kd" --ref "_w2pred_lc10.npz:lc10" --label w5_VR10
B --new "_w2pred_lc20kd.npz:lc20kd" --ref "_w2pred_lc20.npz:lc20" --label w5_VR20
B --new "_w2pred_lc10kd.npz:lc10kd" --ref "_w2pred_lc10kdR.npz:lc10kdR" --label w5_X10
B --new "_w2pred_lc20kd.npz:lc20kd" --ref "_w2pred_lc20kdR.npz:lc20kdR" --label w5_X20

# --- Vision and encoder controls (supplementary S5, Table S4)
B --new "_pred_dump.npz:RA+RD" --ref "_w2pred_ranosub.npz:ranosub" --label RD_vs_ranosub
B --new "_w2pred_cap128.npz:cap128" --ref "_pred_dump.npz:RA+RD" --label cap128_vs_64
B --new "_w2pred_cap256.npz:cap256" --ref "_pred_dump.npz:RA+RD" --label cap256_vs_64
B --new "_w2pred_lastframe.npz:lastframe" --ref "_pred_dump.npz:RA only" --label lastframe_vs_RA
B --new "_w2pred_lastframe.npz:lastframe" --ref "_pred_dump.npz:RA+RD" --label lastframe_vs_RD
B --new "_w2pred_lastframe.npz:lastframe" --ref "_pred_dump.npz:RA+RD" --label lastframe_vs_RD_16 --slots 1-6
B --new "_w2pred_ranosub.npz:ranosub" --ref "_pred_dump.npz:DA only" --label ranosub_vs_DA
B --new "_w2pred_ranosub.npz:ranosub" --ref "_pred_dump.npz:RA only" --label ranosub_vs_RA
B --new "_w2pred_ranosub.npz:ranosub" --ref "_pred_dump.npz:RA+DA" --label ranosub_vs_RADA
B --new "_pred_dump.npz:Teacher" --ref "_pred_dump.npz:Vision-only" --label teacher_vs_vision
B --new "_w2pred_twostem.npz:twostem" --ref "_pred_dump.npz:RA only" --label twostem_vs_RA
B --new "_w2pred_twostem.npz:twostem" --ref "_pred_dump.npz:RA+RD" --label twostem_vs_RD
B --new "_w2pred_twostem.npz:twostem" --ref "_pred_dump.npz:RA+RD" --label twostem_vs_RD_16 --slots 1-6
B --new "_w2pred_visgap.npz:visgap" --ref "_pred_dump.npz:RA+RD" --label visgap_vs_RD
B --new "_w2pred_visgap.npz:visgap" --ref "_pred_dump.npz:Vision-only" --label visgap_vs_vision
B --new "_pred_dump.npz:Vision-only" --ref "_pred_dump.npz:RA+RD" --label vision_vs_RD

# --- Doppler-angle map statistics (Sec. VII-C, Table III(b), supplementary S13 and S14)
B --new "_w2pred_dadelta.npz:dadelta" --ref "_w2pred_damarg.npz:damarg" --label P4_delta_vs_collapse_10
B --new "_w2pred_dagauss.npz:dagauss" --ref "_pred_dump.npz:DA only" --label P5_gauss_vs_full_10
B --new "_w2pred_daspread.npz:daspread" --ref "_w2pred_damarg.npz:damarg" --label P6_spread_vs_collapse
B --new "_w2pred_damarg.npz:damarg" --ref "_pred_dump.npz:RA only" --label collapse_vs_RA
B --new "_w2pred_dascr.npz:dascr" --ref "_w2pred_damarg.npz:damarg" --label dascr_vs_collapse
B --new "_w2pred_dascr.npz:dascr" --ref "_w2pred_dadelta.npz:dadelta" --label dascr_vs_delta
B --new "_w2pred_dascr.npz:dascr" --ref "_pred_dump.npz:DA only" --label dascr_vs_full
B --new "_w2pred_dascr.npz:dascr" --ref "_w2pred_dagauss.npz:dagauss" --label dascr_vs_gauss
B --new "_w2pred_dadelta.npz:dadelta" --ref "_pred_dump.npz:RA only" --label delta_vs_RA_10
B --new "_w2pred_dadelta.npz:dadelta" --ref "_pred_dump.npz:DA only" --label delta_vs_full_10
B --new "_w2pred_dagauss.npz:dagauss" --ref "_w2pred_damarg.npz:damarg" --label gauss_vs_collapse_10
B --new "_w2pred_dagauss.npz:dagauss" --ref "_w2pred_dadelta.npz:dadelta" --label gauss_vs_delta_10
B --new "_w2pred_dagaussfl.npz:dagaussfl" --ref "_w2pred_damarg.npz:damarg" --label gaussfl_vs_collapse
B --new "_w2pred_dagaussfl.npz:dagaussfl" --ref "_pred_dump.npz:DA only" --label gaussfl_vs_full
B --new "_w2pred_dagaussfl.npz:dagaussfl" --ref "_w2pred_dagauss.npz:dagauss" --label gaussfl_vs_gauss
B --new "_w2pred_daspread.npz:daspread" --ref "_pred_dump.npz:DA only" --label spread_vs_full
B --new "_w2pred_daspread.npz:daspread" --ref "_w2pred_dagauss.npz:dagauss" --label spread_vs_gauss

# --- Nonlearned trackers (Sec. V-C, Fig. 3a; horizon proper t+1..t+6)
B --new "_trk_road_ekf.npz:full_anchored" --ref "_trk_road_ekf.npz:no_doppler_anchored" --label ekf_full_anchored_vs_no_doppler_anchored --slots 1-6
B --new "_trk_road_ekf.npz:full" --ref "_trk_road_ekf.npz:no_doppler" --label ekf_full_vs_no_doppler --slots 1-6
B --new "_pred_dump.npz:RA+RD" --ref "_trk_road_ekf.npz:full" --label student_vs_ekf_full --slots 1-6
B --new "_pred_dump.npz:RA+RD" --ref "_trk_road_ekf.npz:full_anchored" --label student_vs_ekf_full_anchored --slots 1-6
B --new "_pred_dump.npz:RA+RD" --ref "_trk_road_ekf.npz:no_doppler" --label student_vs_ekf_no_doppler --slots 1-6
B --new "_pred_dump.npz:RA+RD" --ref "_trk_road_ekf.npz:no_doppler_anchored" --label student_vs_ekf_no_doppler_anchored --slots 1-6

echo "BOOTSTRAPS DONE"
