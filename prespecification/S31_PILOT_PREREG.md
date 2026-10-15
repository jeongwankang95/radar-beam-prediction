# S31 RA-only pilot — pre-registration

**Written 2026-08-30, before any RA-only run on Scenario 31 was launched.**
Nothing below may be changed after the first run starts. If any of it is changed,
the change and its reason are recorded here with a date and the pilot is restarted.

## Question

Does the Doppler gain of contribution C2, measured on Scenario 9, reproduce at a
second site? Scenario 31 is the only other site that qualifies: its radar arm reaches
`16.8%` current-slot top-1 against a `7.2%` constant-beam baseline (2.3x), so it
learns, whereas Scenario 32's radar arm sits exactly on its constant baseline
(`16.9%` vs `16.9%`) and Scenario 33 has no training split at all.

This is a **weak-site replication**, not a generalization test. Both S31 arms are
weaker than oracle-fed persistence (`39.7%`), so a positive result would say the
Doppler effect reproduces inside the model, not that the predictor is useful there.

## What is held fixed

Everything in `cmkd_results/_s31rd_s42/phase0_0807_0206/args.json` except the radar
input. Specifically: `epochs 70`, `patience 20`, `lr 1e-4`, `weight_decay 1e-4`,
`batch_size 32`, `T_0 10`, `T_mult 2`, `eta_min 1e-6`, `grad_clip 10.0`,
`seq_len 8`, `num_pred 6`, `feature_size 64`, `gru_hidden 64`,
`gru_layers_student 1`, `num_heads_mha 8`, `radar_h 128`, `radar_w 64`,
`phase 0`, splits `train_seqs_RA.csv / val_seqs_RA.csv / test_seqs_RA.csv`
(35/6/11 passes), same checkpoint-selection metric and same early stopping.

**Only change:** `radar_channels 2 --second_map RD` becomes `radar_channels 1`
(range--angle only). No other architecture or recipe change.

**Seeds:** 42, 1, 2 — the same three the existing S31 RD arm has, so every
comparison is paired per seed.

## Metric

Primary: paired

    dL(seed) = L_RA(seed) - L_RA+RD(seed)

where `L` is mean power lost against the per-frame oracle, computed directly from
that site's own recorded 64-beam sweeps, over all seven predicted slots, under the
same oracle-fed windowed protocol Paper 1 uses. A positive `dL` means Doppler helps.

The Scenario-9 exchange rate (`0.062` dB/pp) is **not** transported; S31's decibels
are computed from S31's sweeps.

Secondary, recorded but not decisive: the same quantity over the horizon proper
(`t+1` to `t+6`), and ATop-1.

## Where the decision is made

**On the six validation passes only.** The eleven test passes are not looked at until
the go/no-go call below has been made and written down.

## Decision rule (locked)

| validation outcome | action |
|---|---|
| `dL > 0` on 3/3 seeds **and** mean `dL >= 0.10` dB | expand RA **and** RD to 10 seeds each |
| signs mixed **and** `abs(mean dL) < 0.05` dB | stop; record as a weak-site null |
| anything else | expand to 5 seeds, re-apply this table |

## Reporting commitment

Scenario 31 is reported in the paper **whatever the outcome**, positive or null. It is
not conditional on the sign. This is fixed now so that the later search for further
DeepSense scenarios cannot be read as selection.

## Statistics

At three seeds we quote **all three paired deltas individually** and no seed-level
95% interval. A pass bootstrap over eleven test passes would describe pass-to-pass
variation only and would not represent training randomness, so it is not used as the
uncertainty statement for a three-seed result.

## Recorded deviation (2026-08-30, ~35 minutes after launch)

The launched runs carry `second_map = DA` (the training script's default) rather than
an absent second map. This does **not** change the model input. With
`radar_channels = 1` and `single_channel = RA`, `prepare_batch` in `train_cmkd.py`
takes the `else` branch and builds the tensor as `ra.unsqueeze(2)`, range--angle only;
`radar_DA` is never read on that path. The only effect is that the loader reads a map
it then discards. The runs were not restarted for it.

The three other differences against `_s31rd_s42`'s stored arguments are
`radar_channels` (2 to 1, intended), `save_dir` (intended), and `single_channel` /
`extra_map` appearing explicitly in the newer argument schema where the 2026-08 run
recorded them as absent. No hyperparameter, split, seed, epoch budget, early-stopping
rule or checkpoint metric differs.

## Go/no-go call (2026-08-30, recorded before the test passes were read)

Validation, six passes, 781 windows, 5,467 scored slots. Paired per seed:

| seed | L(RA) | L(RA+RD) | dL = RA - RD |
|---|---|---|---|
| 42 | 0.7797 | 0.6972 | **+0.0825** |
| 1 | 0.7868 | 0.6200 | **+0.1668** |
| 2 | 0.8343 | 0.6744 | **+0.1599** |

`dL > 0` on **3/3** seeds; mean **+0.1364** dB, which is at or above the pre-registered
`0.10` dB. Over the horizon proper (`t+1` to `t+6`) the three deltas are `+0.0905`,
`+0.1795`, `+0.1572`, the same sign and the same order of magnitude.

**Decision, by the locked table: expand RA and RD to ten seeds each on Scenario 31.**
Seeds 3 through 9 are added to both arms; the three already run are kept unchanged.
The eleven test passes remain unread until those runs finish.

The reporting commitment stands: Scenario 31 goes into the paper whatever the ten-seed
result turns out to be.

## Note on the five-seed branch (2026-08-31, added after the call above)

The locked table's middle branch reads "expand to 5 seeds, re-apply this table", and the
table it points back at is phrased in terms of `3/3`, so at five seeds it has no defined
reading. That is a real defect in the document.

**It was never invoked.** The pilot met the top branch outright: `dL > 0` on 3/3 seeds
with a mean of `+0.1364` dB, so the run went directly from three seeds to ten and the
five-seed path was never on the table. The defect therefore does not touch any decision
that was actually made here, and the ten-seed runs already on disk are not affected by it.

Recording it rather than repairing it is deliberate. Amending the rule now, after the
data that selected the branch has been seen, would weaken the pre-registration more than
the unreachable branch does: the value of the document is that its text is fixed before
the outcome is known. A future pilot that adopts this template should define the
five-seed branch up front, for example as "same sign on at least 4/5 seeds and
`|mean dL| >= 0.10` dB to continue to ten, otherwise stop at five".

## Test result (2026-09-22, ten seeds per arm, the eleven test passes read for the first time)

Computed by `s31_test.py` on the dgx container (copy in `CMKD_Project/rev_20260922_paper1/`), scoring
exactly as `s31_val.py` did for the go/no-go call. 1,259 windows, 11 passes, 8,813 scored slots.

| quantity | paired point | 95% cluster-bootstrap interval (11 passes x 10 seeds, unpaired) | seeds positive |
|---|---|---|---|
| dL7 = L_RA - L_RA+RD (primary) | **+0.142 dB** | [+0.052, +0.264] | 10/10 |
| dL6, t+1..t+6 (secondary) | +0.149 dB | [+0.055, +0.274] | 10/10 |
| dATop-1, seven slots (secondary) | +2.95 pp | [+0.48, +5.68] | 8/10 |

Per-seed dL7: +0.114, +0.087, +0.082, +0.318, +0.222, +0.067, +0.086, +0.265, +0.088, +0.095.
Arm means L7: RA 0.858 dB, RA+RD 0.716 dB; ATop-1 14.73+-2.36 vs 17.68+-1.59.
Oracle-fed persistence on the same test passes: 0.346 dB over t+1..t+6, ATop-1 30.0%, so both
learned arms remain weaker than holding the last measured beam, as the "weak-site" framing above
anticipated. Reported in the paper whatever the outcome, per the commitment above.

---
## Addendum 2026-09-28 — label correction and rerun (written after the rerun was read)
The Scenario 31 preparation used `np.argmax` on power vectors that contain missing entries, so about 1 % of training labels pointed at an unmeasured beam (evaluation had always excluded those entries). Both arms were retrained with `nanargmax` labels, ten seeds each, nothing else changed. Outcome on the eleven test passes: ΔL7 = +0.202 dB [+0.083, +0.386] (10/10 seeds positive; t+1..t+6: +0.211 [+0.087, +0.398]); ATop-1 13.6 ± 2.4 → 17.7 ± 1.0 % (+4.1 pp [+1.6, +7.3]); arm means L7 0.904 / 0.702 dB. The pre-correction arms had given +0.142 [+0.052, +0.264]. The go/no-go decision of this file is unaffected; the manuscript reports the corrected-label arms and states both figures.
