# Pre-specification files

These five files fixed the decision rule (and, for three of them, a prediction) of the pre-specified analyses in the
paper before each analysis was run. They are the authors' working notes. Every file is reproduced unchanged: its text
matches the SHA-256 hash in `SHA256SUMS` (check with `shasum -a 256 -c SHA256SUMS`). Sections appended after an analysis
are dated and marked as outcomes, deviations or addenda.

As the paper states (Sec. V-A), "pre-specified" means that the rule preceded the analysis, not that the data were unseen.
For example, the Scenario 31 site had been screened on the range-Doppler arm's test accuracy before its rule was written.
Paths, run names and host names in the notes refer to the authors' own storage. "Review" refers to internal reviews before
submission, and "published" to earlier internal drafts.

| File | Analysis (paper section) |
|---|---|
| `S31_PILOT_PREREG.md` | Scenario 31 replication of the Doppler contrast (V-B; supplement S6), with the label-correction addendum |
| `PREREG_KD_T1_20260922.md` | Distillation at temperature 1 over ten seeds (V-A; supplement S10) |
| `MECHANISM_PRESPEC_2026-09-27.md` | Doppler-centroid test and moment-map controls (VII-C; supplement S13) |
| `RADAR_TEACHER_CONTROL_PRESPEC_2026-09-29.md` | Radar-only teacher in the learning curve (VII-B; supplement S11) |
| `TEACHER_SEED_REPLICATION_PRESPEC_2026-09-30.md` | Distillation from two further teachers (VII-A; supplement S10) |
