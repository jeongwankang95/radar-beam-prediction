# Radar-only teacher control for the learning curve — written 2026-09-29, before any run of this control

## Why
The learning curve (supplement S10, Table t:lc) shows logit distillation from the vision–radar teacher recovering
0.64 dB [+0.46, +0.84] for a student trained on 10 of the 91 training passes and 0.36 dB [+0.25, +0.48] at 20 passes.
The teacher was trained on all 91 passes, so the gain mixes cross-modal transfer with the teacher's larger training
set (reviews 3A M2, 4 #3). This control replaces the vision–radar teacher by a radar-only teacher trained on the same
91 passes.

## Arms (identical to T1 except the distillation source)
- Teacher: the 91-pass reference range–Doppler student, seed 42 (`cmkd_results/_v3rd_s42/phase0_0803_2028/radar_baseline.pth`,
  ATop-1 42.12 % on the test windows), used through a wrapper with the teacher call signature (`--teacher_type radar`).
  For comparison, the vision–radar source of T1 (`_v3rd_teacher/phase2_0804_1456`) reaches 45.53 %.
- Students: range–Doppler reference student on the nested 10- and 20-pass subsets (`train_seqs_RA_tr_p10/p20.csv`),
  logit distillation only (lambda1 = 1, lambda2 = lambda3 = 0, Gamma = 3), seeds 42, 1, 2, 3, 4, all other settings as
  `_lc10_rdkd_s42/args.json`. Run names `_lc10_rdkdR_s*`, `_lc20_rdkdR_s*`.

## Contrasts (seven-slot dB loss, the paper's cluster bootstrap: 28 test passes, seeds unpaired, 20,000 replicates)
- R(P): radar-teacher KD minus no KD at P passes (dB recovered by a radar-only teacher).
- X(P): vision–radar-teacher KD minus radar-teacher KD at P passes (the cross-modal share).

## Decision rule (primary: P = 10)
- If R(10) is at least half the vision–radar recovery (point estimate >= 0.32 dB) and the interval of X(10) includes
  zero: the learning-curve gain is read as mainly data transfer from a better-trained teacher; the paper will say the
  positive control shows the apparatus transfers, not that cross-modal knowledge does.
- If the interval of X(10) excludes zero with the vision–radar teacher better: a cross-modal share exists for a
  data-starved student, and the paper will report its size.
- Otherwise: unresolved, reported as such. P = 20 is read the same way as secondary evidence.

## Prediction written before training
R(10) between 0.4 and 0.6 dB and X(10) with an interval including zero, i.e. mostly data transfer, because a teacher
trained on nine times more passes carries neighbour-beam structure in its soft labels whatever its modality.

## Outcome (2026-09-29, runs 17:19–17:45 KST, read after all ten finished)
| P | no KD (dB) | radar-teacher KD | vision–radar-teacher KD | R(P) | X(P) |
|---|---|---|---|---|---|
| 10 | 2.503 (24.8 %) | 1.950 (25.7 ± 1.0 %) | 1.862 | +0.553 [+0.350, +0.766] | +0.088 [−0.095, +0.270] |
| 20 | 1.417 (32.4 %) | 1.116 (34.5 ± 0.8 %) | 1.055 | +0.301 [+0.170, +0.428] | +0.060 [−0.033, +0.157] |
- R(10) = 0.55 dB ≥ 0.32 and the X(10) interval includes zero → **first branch**: the learning-curve gain is mainly data transfer
  from a better-trained teacher. The radar-only teacher delivers 86 % (10 passes) and 83 % (20 passes) of the vision–radar
  recovery. ATop-1: X(10) +0.10 pp [−1.59, +1.78], X(20) −0.25 pp [−1.79, +1.25].
- The prediction written above (R(10) in 0.4–0.6 dB, X(10) including zero) held.
- Manuscript: abstract, C3, Sec. V-A (Statistics), VII-B, Conclusion, supplement S10.
