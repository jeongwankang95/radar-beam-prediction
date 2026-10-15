# Radar-only long-term beam prediction: code, predictions and analysis

This repository accompanies

> J. Kang, M. Bordallo López, N. T. Nguyen, J. Kim and S. Kim, "Radar-Only Long-Term Beam Prediction in mmWave
> Vehicular Networks: An Empirical Study of Doppler Input and Cross-Modal Distillation," arXiv:2610.20185, 2026.
> <https://arxiv.org/abs/2610.20185>

It contains the training code, the test-set predictions of every trained arm, a record of every training run, and the
scripts that turn the predictions into the paper's tables, figures and intervals. The DeepSense 6G data are not included
(see `dataset/scenario9/README.md`), and neither are trained checkpoints.

## Layout

| Folder | Contents |
|---|---|
| `src/` | data loader, models, training (`train_cmkd.py`, `train_vision_only.py`), radar preprocessing, prediction dumps (`w2_pred.py`, `predcache_ladder.py`), evaluations that need checkpoints |
| `scripts/` | the launchers the runs were trained with (each header names its runs), `run_predictions.sh`, and `train_command.py`, which prints the command of any run |
| `analysis/` | the bootstrap contrasts and every table and analysis that the stored predictions support; `run_analysis.sh` runs them all |
| `figures/` | Figs. 3 to 6 (TikZ) |
| `results/` | stored test-set predictions, the training record `runs.json`, and in `reference/` the outputs behind the paper |
| `dataset/scenario9/` | the train, validation and test split (the data themselves come from DeepSense 6G) |
| `prespecification/` | the five pre-specification files, unchanged (`SHA256SUMS`) |

## Reproducing the paper's numbers from the stored predictions

This needs Python 3 with numpy, scipy and pandas, and DeepSense 6G Scenario 9 in `dataset/scenario9/`, prepared as in
its README: every loss is read from the measured beam powers, and the tracker and the Doppler-centroid test read the radar
maps. No GPU and no checkpoint are needed.

```bash
bash analysis/run_analysis.sh
```

It runs every script below in order, writes to `results/analysis/` and `figures/out/`, and ends with
`analysis/check_reference.py`, which compares each regenerated output with the copy behind the paper in
`results/reference/` (and the rebuilt prediction files with those in `results/`). It takes under an hour on a laptop
CPU; each script can also be run alone. The scripts were checked with Python 3.11, NumPy 2.4.3, SciPy 1.17.1 and pandas
3.0.1. Every interval is the cluster bootstrap of supplementary Sec. S3: the 28 test passes are resampled as clusters
and the seeds of each arm are resampled separately, with a fixed generator seed. The draw depends on the order of the
seeds inside each prediction file, so the intervals match the paper exactly with the stored files.

### Where each result comes from

| Paper | Script | Input in `results/` |
|---|---|---|
| Table I | `analysis/table1.py` | `eval_table1.json` (test metrics each run wrote at the end of training) |
| Table II; Sec. VI slope, R² and pair ordering; supplementary S9 subset slopes | `analysis/table2.py` | `_pred_dump.npz` |
| Table III | `analysis/table3.py` | `_predcache.npz`; panel (b) from the bootstrap files |
| Fig. 3, every interval in the text | `analysis/run_bootstraps.sh` (82 contrasts, grouped by section), `figures/make_forest.py` | `_pred_dump.npz`, `_w2pred_*.npz`, `_trk_road_ekf.npz` |
| Fig. 4; supplementary Table S14 | `figures/perslot.py`, then `figures/fig_perslot.py`; `analysis/subgroup.py` | `_pred_dump.npz`, `_predcache.npz` |
| Fig. 5 | `analysis/slope_interval.py` (line, band, slope interval), `figures/fig5_calib.py` | `_pred_dump.npz` |
| Fig. 6 | `figures/fig_lc.py` | bootstrap files |
| Sec. V-C trackers; supplementary S8 | `analysis/extract_frames.py`, then `analysis/road_ekf_tracker.py` | radar maps and GPS |
| Sec. VI flat peak | `analysis/flat_peak.py` | measured powers |
| Values computed in the text: Doppler arms within 0.13 dB (V-B), per-slot range and persistence (V-C), the gain at t and t+6 and their difference (VII-C, Table S14 caption), the six-slot reading (S1), leave-one-out slopes (S14) | `analysis/text_values.py` | `_pred_dump.npz`, `_predcache.npz`, `_w2pred_rdfeat/rdfull.npz` |
| Teacher's lead and the share distillation recovers (Secs. V-B and VIII, supplementary S10) | `analysis/distillation_share.py`, `analysis/distillation_share_pooled.py` | `_pred_dump.npz`, `_w2pred_rdteacher.npz`, `_w2pred_rdkdT1/T2.npz` |
| Sec. VII-C angles; supplementary S9 | `analysis/pred_dump_summary.py` | `_pred_dump.npz` |
| Sec. VII-C and supplementary S13 Doppler-centroid test | `analysis/doppler_centroid_test.py` | radar maps and GPS |
| Supplementary S3 alternative intervals and per-beam gain | `analysis/interval_constructions.py` | `_pred_dump.npz` |
| Supplementary S6 Scenario 31 | `analysis/s31_penalty_sensitivity.py` | `_s31fix_test.npz`, `_s31_test.npz` |
| Supplementary S10 temperature sweep and MLP projector; S11 fusion table; S13 Doppler-resolution sweep | `analysis/training_metrics.py` | `runs.json` |
| Supplementary S12 subgroups and horizon | `analysis/subgroup.py` | `_predcache.npz` |
| Sec. V-B MACs; supplementary Table S2 | `src/model_cost.py` (needs PyTorch and thop, no checkpoint) | none |

Figs. 1 and 2 are drawings. In Fig. 5 the paper places the points and bars at the rounded entries of Table II, within
0.1 pp and 0.01 dB of the positions `figures/fig5_calib.py` writes.

Some numbers need the trained checkpoints and are not regenerated here; their outputs are in `results/reference/`:
distillation at temperature 1 over ten seeds (`src/kd_t1_eval.py`), the all-zero second map (`src/mcap_eval.py`), the
Scenario 31 contrast (`src/s31_test.py`, whose per-window losses `results/_s31fix_test.npz` reproduce the reported point
and interval through `analysis/s31_penalty_sensitivity.py`), a per-arm tail summary (`src/exp_power_tail.py`; Table II's
tail columns come from `analysis/table2.py`) and the day/night attempt (`src/eval_daynight.py`). The latencies of Sec. V-B, which the paper cites from its companion paper
(arXiv:2610.20196), are measured by `src/bench_infer.py`, the best
validation scores of supplementary S10 come from the training logs, and the dates of supplementary S4 from the run
directories listed in `results/runs.json`.

### Where the paper's text and these scripts differ

* Table I, row RA+DA + logit + feature: the runs give an ADBA of 92.149 %, which rounds to 92.1; the paper prints 92.2.
* Table III(a), row "RD rows permuted", and supplementary S14: `analysis/table3.py` gives −1.13 pp on [−3.00, +0.66]
  against RA+RD and +3.44 pp on [+0.89, +6.01] against range–angle alone; the paper prints [−2.99, +0.66] and
  [+0.87, +6.02]. These two intervals were first computed in a run whose draw order was not recorded; across draw orders
  their endpoints move by up to about 0.03.
* Sec. VII-C and supplementary S14, the Doppler-resolution sweep: from unrounded means the student trails its
  full-resolution run by 5.5, 3.1 and 1.1 pp at B = 1, 8 and 16 and by 0.6 pp at B = 64, and lies 0.05 pp above it at
  B = 32 (supplementary S13 gives the means); supplementary S14 prints 3.2 and 1.2 (differences of rounded means), and
  both places state a 0.05 pp deficit at B = 32.

## Training

`requirements.txt` pins the environment of the CUDA runs (Python 3.12, PyTorch 2.6.0 with CUDA 12.4). Prepare the data as
in `dataset/scenario9/README.md` and run everything from the repository root. `src/train_cmkd.py` trains in
four phases (`--phase`): 0, a radar-only student without distillation; 1, the vision-radar teacher on the task loss; 2,
the teacher's self-distillation; 3, a student distilled from a phase-2 teacher (`--teacher_ckpt`). The vision-only
student has its own trainer, `src/train_vision_only.py`. A run of `src/train_cmkd.py` writes
`cmkd_results/<tag>/phase<k>_<date>/` with its arguments (`args.json`), checkpoint and test metrics (`eval_results.npz`).

Every run behind a number in the paper is listed in `results/runs.json`: its tag, seed, the environment switches
`src/DataFeed.py` reads (`MAP_TRANSFORM`, `SCRAMBLE_DOPPLER`, `SCRAMBLE_SEED`, `LAST_FRAME_ONLY`, `RA_VARIANT`), the
arguments recorded in `args.json` and the test metrics it reported, grouped into 34 families with the paper item each
serves and its launcher. `python scripts/train_command.py <tag>` prints the exact command of any run, including the
ones whose launcher is not in `scripts/`. A distillation run takes its teacher from the latest run directory of that
teacher's tag, so train a teacher before the students distilled from it; each student's `teacher_ckpt` in
`results/runs.json` names its teacher (mostly `_v3_teacher`, from `scripts/run_valsplit_retrain.sh`, and `_v3rd_teacher`,
from `scripts/run_rd_switch.sh`).

After training, `bash scripts/run_predictions.sh` writes the test-set predictions of every arm to
`cmkd_results/predictions/` in the format of `results/`, and `src/predcache_ladder.py` those of the Table III(a) arms;
copy a regenerated file into `results/` to analyze it. `src/collect_table1.py` rebuilds `results/eval_table1.json`.

Notes:

* Seeds fix initialization and data order and cuDNN runs deterministically, so a run repeats bit for bit on the same
  hardware and software, but not across machines. Predictions regenerated elsewhere differ from the stored ones in a few
  near-tie slots (we see 0 to 8 of 5,418 per seed between our machines), so intervals computed from them can differ
  slightly from the paper's. The runs of July and early August, among them the two seed-42 teachers, were trained on a
  laptop GPU (Apple MPS) with `--num_workers 0`; the later ones on CUDA.
* `src/w2_pred.py` and the other evaluation scripts create their data loaders at module level, which worker processes
  do not support on macOS and Windows; there, run them with `--num_workers 0` where the option exists.
* Scenario 31 has power vectors with missing entries. The paper reports the arms trained before the label correction
  (`_s31ra_*`, `_s31rd_*`), whose labels took the plain arg max, and those retrained after it (`_s31fix_*`), whose labels
  take the arg max over measured entries, as `src/DataFeed.py` computes them now. Scenario 9 has no missing entries, so
  the two agree there.
* The launchers in `scripts/` are the ones the runs were started with, adapted to this repository's layout (paths,
  interpreter, teacher checkpoints found by tag). Some ran several copies in parallel lanes or waited for another queue;
  their headers say so.

## Pre-specification

`prespecification/` holds the five files that fixed a decision rule, and for three of them a prediction, before the
analysis they govern was run. They are reproduced unchanged; `shasum -a 256 -c SHA256SUMS` checks them. Their README
says which analysis each governs.

## License

The code is released under the MIT License (`LICENSE`). Part of `src/` builds on code by Mengyuan Ma, included with his
permission (`NOTICE.md`). The stored predictions and results are derived from DeepSense 6G Scenario 9, whose terms of
use apply to the data.

## Citation

```bibtex
@misc{kang2026radaronly,
  author        = {Kang, Jeongwan and Bordallo L{\'o}pez, Miguel and Nguyen, Nhan Thanh and Kim, Junhyeong and Kim, Sunwoo},
  title         = {Radar-Only Long-Term Beam Prediction in {mmWave} Vehicular Networks: An Empirical Study of {Doppler}
                   Input and Cross-Modal Distillation},
  year          = {2026},
  eprint        = {2610.20185},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SP},
  url           = {https://arxiv.org/abs/2610.20185}
}
```
