# Pre-specification: Doppler-as-bearing-proxy tests (written before any result was read)

Date: 2026-09-27. Author: J. Kang (with assistant). Status: **locked before running E4; T8 arms not yet trained.**
Purpose: fix the hypotheses, the statistics and the decision rules for the mechanism tests that revise Sec. VII-C, so that they are read as tests and not as post-hoc narration. This file is to be supplied with the supplementary material.

## Hypothesis

H_proxy. On Scenario 9 the vehicle traverses a straight road in one direction at 3.0–4.7 m/s. The radial velocity seen by the roadside radar is v_r ≈ v·sin θ, where θ is the vehicle bearing from the array broadside. The beam codebook is monotone in sin θ. Therefore the Doppler bin of the moving target is, within a pass, an affine function of the optimal beam index, with slope proportional to the vehicle speed. Because the radar's Doppler resolution is 0.243 m/s per bin (750 MHz, 15 MHz/µs, 128 chirps; maximum velocity 56 km/h, as in [7] for this scenario) and the four-element angle FFT has a beamwidth of the order of 25°, the Doppler axis resolves bearing several times more finely than the range–angle map's angle axis. The Doppler increment of C2 would then be, at this site, chiefly an **angular-resolution** gain rather than an extrapolation (velocity-for-prediction) gain.

Alternative H_spread (current manuscript wording). The encoder uses the distribution of moving energy across Doppler bins without the bin index carrying bearing information; ordering is immaterial.

## E4 — information-level test (no training; local data)

Data: all 5,964 Scenario 9 samples; Doppler–angle map (128 rows = Doppler after FFT shift, 64 columns = angle), recorded 64-beam power vector, GPS of the vehicle, pass id (seq_index). Split as in the paper: passes 1–91 train, 92–108 validation, 109–136 test, in chronological pass order.

Per-frame statistics. Column noise floor removed: w[d,a] = max(map[d,a] − median_d map[·,a], 0). Column energy e[a] = Σ_d w[d,a]; peak angle a* = argmax_a e[a]. Doppler centroid in the target band: c = Σ_{d, |a−a*|≤2} w·d / Σ w. Doppler standard deviation s in the same band. Label b = argmax of the recorded power vector (0-based).

Tests and predictions.
- **P1 (linearity).** Pooled over the training passes, fit b = α + βc. Read the RMS residual on the test passes in beam indices. Prediction: residual **< 3.60**, the residual of the per-angle profile-peak reader (Table S11), and in the range 1.5–2.5 indices; Pearson |r| > 0.978.
- **P2 (speed scaling).** Fit β_p within each pass (c on b). Regress β_p on the pass's median GPS speed v_p through the origin. Prediction: same sign on all 136 passes and R² > 0.5. The implied codebook spacing Δ(sin θ) per beam = β_p·0.243/v_p should agree across passes to within 20%.
- **P3 (joint reader).** A linear reader on (a*, c) has a smaller test residual than either alone.

Decision rule for the manuscript. If P1 and P2 hold, Sec. VII-C states H_proxy as the mechanism supported by the data, C2 is re-read as an angular-resolution gain specific to this geometry, and the generalization limit is stated as "bidirectional or variable-speed traffic weakens the proxy". If P1 fails (residual ≥ 3.60) or P2 fails (R² ≤ 0.5 or sign changes), the current wording is kept and the failed test is reported as evidence against H_proxy. P3 is descriptive.

## T8 — encoder-level test (training; exploratory, 10 seeds per arm)

Arms: Doppler–angle-only student of Table I, input transformed per frame with the architecture, budget and protocol untouched (same framework as Table III):
- **DA-delta**: in every column, the column's total energy e[a] is placed in the single row nearest the column centroid c[a]; all other rows zero. Keeps total and centroid, destroys spread.
- **DA-gauss**: e[a] spread over rows as a Gaussian with mean c[a] and standard deviation s[a]. Keeps total, centroid and spread; destroys finer shape.
- **DA-spread-only**: the same Gaussian with its mean fixed at the map centre (row 64). Keeps total and spread, destroys centroid.
Reference points already measured: DA full 41.8 ± 1.6, DA collapse (total only, Table III) 38.0 ± 1.3, RA only 37.3 ± 1.2 (ten seeds each).

Predictions under H_proxy.
- **P4.** DA-delta retains at least half of the DA increment over RA only: ATop-1 ≥ 39.6% (ten-seed mean), interval on the delta against DA-collapse excluding zero.
- **P5.** DA-gauss is not separated from DA full (interval on the difference includes zero).
- **P6.** DA-spread-only is not separated from DA-collapse.
Under H_spread, P4 fails and P6 succeeds with DA-spread-only above DA-collapse.

Statistics: the paper's cluster bootstrap (28 test passes and 10 seeds, seeds unpaired, 20,000 replicates). The arms are declared exploratory in Sec. V-A.

## What this file does not allow

No reinterpretation of the row-permutation control of Table III beyond its interval (−1.53 pp on [−3.73, +0.56]). No claim of speed invariance. No transport of the Scenario 9 slope to Scenario 31 without repeating E4 there.

---
## Outcome of E4 (run 2026-09-27, after the predictions above were locked)

Script: `CMKD_Project/rev_20260927_mech/e4_bearing_proxy.py`; summary `e4_summary.json`; per-frame arrays `e4_frames.npz`.

| Test | Pre-specified criterion | Result | Verdict |
|---|---|---|---|
| P1 centroid reader residual (test passes) | < 3.60 indices, |r| > 0.978 | **11.7 indices RMS, r = 0.85** (angle-peak reader on the same maps: 6.8, r = 0.95; joint reader 6.6) | fails |
| P2 slope ∝ speed | same sign on all passes; R² > 0.5 | same sign 136/136; **R² = 0.08**, corr 0.42; slope/speed CV 0.19 ≈ slope CV 0.21 | fails |
| P3 joint reader better than either | descriptive | 6.6 < 6.8 | holds, marginal |

Diagnostics: within-pass median r = 0.94 for the centroid but 0.99 for the angle peak; rescaling by oracle per-pass speed leaves the residual at 11.4, so speed spread is not the cause; the residual is largest at the codebook edges (14.1) and smallest mid-range (7.1); zero-Doppler crossing sits at beam 33 ± 2.5 across passes.

Decision (per the rule above): H_proxy is **not supported** at the information level. The manuscript keeps the distributional wording of Sec. VII-C and reports this test (main Sec. VII-C, one paragraph; supplement, new section). T8 remains scheduled as the encoder-level test; prior for P4 lowered, P4–P6 unchanged as written.

Side finding (codebook grid, F5): a single-ray 16-element fit of the measured profiles has relative residual 0.41 for the paper's sinθ ∈ [−1, 1] grid against 0.28 for a 90° field of view (sinθ-uniform or angle-uniform indistinguishable, 0.278 vs 0.277); the profile-reader calibration implies Δ(sinθ) ≈ 0.021 per beam (span ±41°); measured half-power beam width 6.75 codebook steps (median). All angular figures move to the 90° grid; degree figures that depend on cached predictions (mean misalignment 3.64°/3.03°/2.95°, S8) are recomputed in Wave 1.

Not reproduced: the Table S11 calibration (`beam = 1.6643·angle − 28.198`, r = 0.978, residual 3.60 on the training split). A re-implementation summing the raw log map over Doppler gives slope 1.50, r = 0.94, RMS 7.2 on the same split. The m7a/b/c scripts that produced the published figures are not on the local machine; to be re-derived in Wave 1.

---
## T8 interim (2026-09-27, 8 of 10 DA-delta seeds, 1 of 10 DA-gauss seeds) and one added arm

Interim, read before the remaining seeds finished: DA-delta 37.0 ± 0.8 % ATop-1, 1.03 dB, against DA-collapse 38.0 ± 1.3 / 0.88 dB (−1.0 pp on [−3.5, +1.6]) and DA full 41.8 / 0.67 (−4.7 pp on [−7.3, −2.2]). **P4 fails.** The first DA-gauss seed sits at 37.1 %, i.e. at the collapse level; P5 is heading to fail.

Confound noticed at this point and recorded before any result of the added arm exists: the three moment maps zero the column floor (the median over rows), whereas the collapse control keeps it (it repeats the column mean, floor included). A fourth arm, **DA-gauss+floor** (`MAP_TRANSFORM=dagaussfl`: the Gaussian moment map with each column's median added back), ten seeds, separates the two readings:
- if DA-gauss+floor recovers the increment (not separated from DA full), the cue is the floor texture plus two moments, not the finer Doppler distribution;
- if it stays at the collapse level, the encoder needs structure of the per-column value distribution beyond its first two moments and its floor, which is what the row-permutation result (multiset preserved, order destroyed, small loss) already suggested.
This arm is exploratory; the prediction written here is the second outcome.

## T8 outcome, DA-delta complete (2026-09-28, ten seeds)
DA-delta 37.17 ± 0.75 % ATop-1, 1.018 ± 0.051 dB, 9.7 % of slots above 3 dB.
- vs DA-collapse (38.0 / 0.880): −0.86 pp on [−3.36, +1.70]; −0.14 dB on [−0.37, +0.08] → not separated; **P4 fails** (required ≥ +1.6 pp with an interval excluding zero).
- vs DA full (41.8 / 0.673): −4.59 pp on [−7.18, −2.07]; −0.35 dB on [−0.55, −0.17] → the delta map forfeits the whole Doppler increment.
- vs RA only (37.3 / 0.914): −0.16 pp on [−2.79, +2.31] → at the range–angle-only level.
Keeping each angle's total moving energy and its Doppler centroid, and nothing else, leaves the encoder no better than having no Doppler map at all. P5 (DA-gauss) and the added DA-gauss+floor arm are pending.

## T8 outcome, DA-gauss complete and DA-gauss+floor at eight seeds (2026-09-28)
- DA-gauss (ten seeds): 38.69 ± 1.58 %, 0.855 dB. vs DA full −3.08 pp on [−5.45, −0.68], −0.18 dB on [−0.35, −0.02] → separated; **P5 fails**. vs DA-collapse +0.66 pp on [−1.32, +2.67] → not separated. Seeds spread widely (37.0–41.3 %): a two-moment summary sometimes recovers most of the increment and usually does not.
- DA-gauss+floor (eight of ten seeds): 38.41 ± 2.05 %, 0.878 dB. vs DA full −3.35 pp on [−5.78, −1.05]; vs collapse +0.38 pp on [−1.66, +2.39]; vs DA-gauss −0.28 pp on [−2.30, +1.62]. Adding the floor changes nothing → the confound noted above is excluded; the second reading holds: the encoder needs structure of each column's value distribution beyond its total, centroid, spread and floor.
- P6 (DA-spread, Gaussian at zero Doppler) is no longer informative once DA-gauss sits at the collapse level; it is queued as optional and will be reported only if it runs before submission.
Verdict for the manuscript (Sec. VII-C): the row permutation, which keeps each column's multiset of values, cost −1.5 pp (unresolved); every summary that replaces the multiset cost the whole increment. The cue is the shape of the per-angle magnitude distribution, not its location or its two leading moments.

## T8 outcome, DA-gauss+floor complete (2026-09-28 13:15 KST, ten seeds)
- DA-gauss+floor: 38.79 ± 1.97 %, 0.847 dB. vs DA full −2.97 pp on [−5.32, −0.67], −0.17 dB on [−0.34, −0.02] (separated); vs DA-gauss +0.10 pp on [−1.70, +1.76], +0.01 dB on [−0.13, +0.14]; vs DA-collapse +0.76 pp on [−1.12, +2.62], +0.03 dB on [−0.10, +0.17].
- Reading (the second of the two outcomes written above): restoring the column floor to the two-moment Gaussian recovers nothing, so the increment lives in the per-column value distribution beyond its total, centroid, spread and floor. Manuscript Table III row, footnote and Sec. VII-C updated from the eight-seed interim.

## T8 outcome, DA-spread-only complete (2026-09-29, ten seeds)
- DA-spread (Gaussian centred at zero Doppler; keeps total and spread, removes the centroid): 37.77 ± 1.00 %, 0.915 dB.
- vs DA-collapse (38.03 / 0.880): −0.26 pp on [−1.80, +1.32], −0.04 dB on [−0.14, +0.08] → not separated; **P6 holds as written** (spread alone adds nothing over the per-angle total).
- vs DA-gauss: −0.91 pp on [−2.35, +0.56]; vs DA full: −3.99 pp on [−6.46, −1.61], −0.24 dB on [−0.42, −0.08].
- Reading: with P4 and P5 failed and P6 holding, none of total, centroid, spread (or the floor) carries the Doppler increment; the encoder uses the finer shape of each column's value distribution. Reported in Table III (fourth pre-specified row) and Sec. VII-C.
