# Teacher-seed replication of the distillation null — written 2026-09-30, before any run of this replication

## Why
Review 5 (2.2a): the C3 interval (logit KD minus no KD on the reference range–Doppler student, −0.01 dB [−0.07, +0.06])
is conditional on one frozen teacher checkpoint (`_v3rd_teacher/phase2_0804_1456`, seed 42). Two more range–Doppler
Phase-2 teachers trained the same way exist (`_teacher_rd_s1/phase2_0805_0925`, 45.13 % test ATop-1;
`_teacher_rd_s2/phase2_0805_1025`, 46.38 %; the three together 45.70 % and 0.408 dB).

## Arms
- `_v3rdkdT1_s{42,1,...,9}`: logit KD from teacher s1; `_v3rdkdT2_s*`: from teacher s2. Every other argument identical to
  `_v3rdkd_s42/phase3_0805_0644/args.json` (lambda1 = 1, lambda2 = lambda3 = 0, Gamma = 3, 91 training passes).
- Reference: the ten-seed undistilled student of Table I (`_v3rd_s*`) and the seed-42-teacher arm (`_v3rdkd_s*`).

## Contrasts (seven-slot dB loss, the paper's cluster bootstrap: 28 passes, seeds unpaired, 20,000 replicates)
- G1, G2: dB recovered by distillation from teacher s1 and s2 against no KD. G_all: the thirty distilled runs of all
  three teachers pooled against no KD.

## Decision rule
- If both G1 and G2 have intervals including zero with point estimates within ±0.07 dB: the null is not specific to the
  seed-42 teacher; C3 and Sec. VII-A report the three teachers.
- If either interval excludes zero in favour of distillation: the manuscript reports a teacher-dependent gain and C3 is
  revised accordingly.
- Otherwise: reported as is, with the interval.

## Prediction written before training
G1 and G2 both within ±0.05 dB with intervals including zero.

## Determinism probe (same day, not a hypothesis test)
Three two-epoch runs of the Doppler–angle-only student, seed 42: A and B identical, C with the resolution sweep's identity
transform (MAP_TRANSFORM=pool128). Per-epoch validation loss and ADBA compared to four decimals. Purpose: supplement S7
(the sweep's identity point sits 1.2 pp above Table I's arm on the same seeds).

## Outcome (appended 2026-09-30 21:50, after all twenty runs finished; evaluation `rev_20260927_mech/w6_eval.sh`)
All twenty runs completed without error (T1 on dgx, T2 on dgx2; logs `_w6_T{1,2}_lane*.log`).

| Contrast | ATop-1 (new − ref) | dB recovered (ref − new) |
|---|---|---|
| G1, teacher s1 (10 students, 41.48 ± 0.79 %) | −0.41 pp [−1.51, +0.74] | −0.013 [−0.072, +0.049] |
| G2, teacher s2 (10 students, 41.37 ± 1.02 %) | −0.53 pp [−1.75, +0.74] | −0.033 [−0.097, +0.034] |
| G_all, 30 runs of three teachers | −0.47 pp [−1.49, +0.57] | −0.017 [−0.076, +0.042] |
| (reference) G0, teacher s42 | −0.48 pp [−1.72, +0.78] | −0.006 [−0.068, +0.059] |

Decision: first branch — both intervals include zero and both points lie within ±0.07 dB, so the null is not specific
to the seed-42 teacher; C3 and Sec. VII-A report the three teachers.
Prediction (both within ±0.05 dB, intervals including zero): held (−0.013, −0.033).
Follow-up (not pre-specified): share of each teacher's own lead recovered, joint bootstrap (`r6_joint_R_all.py` →
`_boot_joint_R_all.json`): s1 −0.06 [−0.47, +0.19], s2 −0.14 [−0.53, +0.13], pooled −0.08 [−0.43, +0.15].
Determinism probe (same day): three two-epoch runs bit-identical; a full 70-epoch rerun (`_det_full_D`, dgx) reproduces
the Aug-18 `_m6DAonly_s42` checkpoint bit for bit (45/45 tensors) and all 70 validation epochs, while the Sep-1 sweep
checkpoint `_gridDApool128_s42` differs in 43/45 tensors — the offset belongs to the sweep campaign.
