# DeepSense 6G Scenario 9: data and split definitions

The DeepSense 6G data are not redistributed here. Download Scenario 9 from the DeepSense 6G website
(<https://www.deepsense6g.net>) under its terms of use and extract it into this folder, so that `scenario9.csv` and
`unit1/`, `unit2/` sit next to the files below. Then build the radar maps from the repository root:

```bash
python src/preprocess_radar.py      # range-angle and Doppler-angle maps; writes scenario9_RA.csv and scenario9_DA.csv
python src/preprocess_radar_rd.py   # range-Doppler maps
python src/preprocess_ra_nosub.py   # optional: range-angle maps without mean subtraction (one control)
```

The scripts expect the data in this folder. To keep them elsewhere, link them in, for example
`ln -s /data/scenario9/unit1 dataset/scenario9/unit1` (likewise `unit2` and `scenario9.csv`); the analysis scripts also
accept `CMKD_DATA` set to a folder that holds both the data and the split files below.

## Split definitions (part of the repository)

Each row is one window: eight observed frames (`camera1..8`, `radar1..8`, the power files `beam1..8`), the six
future power files (`future_beam1..6`) and the pass (`seq_index`). Passes never straddle two splits.

| File | Windows | Passes | Use |
|---|---|---|---|
| `train_seqs_RA.csv` | 3,286 | 108 | the first 80% of passes in time (training and validation together) |
| `train_seqs_RA_tr.csv` | 2,787 | 91 | training |
| `val_seqs_RA.csv` | 499 | 17 | validation: the temporally last passes of `train_seqs_RA.csv` |
| `test_seqs_RA.csv` | 774 | 28 | test: the last 20% of passes in time |
| `train_seqs_RA_tr_p10.csv`, `_p20.csv`, `_p45.csv` | 320, 611, 1,419 | 10, 20, 45 | learning curve: nested subsets of the training passes |

`src/gen_data_seq.py` (windows and the train/test split) and `src/make_val_split.py` (validation split) regenerate the
first four files byte for byte from `scenario9_RA.csv`; `src/make_lc_subsets.py` regenerates the subsets from
`train_subsets_provenance.json`, which records the random permutation of the training passes they are cut from.

Scenarios 31, 32 and 33 go to `dataset/scenario31`, `dataset/scenario32` and `dataset/scenario33`; `src/s31_prep.py` and
`src/s3x_prep.py` build their maps and splits.
