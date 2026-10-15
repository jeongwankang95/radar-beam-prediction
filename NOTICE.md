# Third-party code

Part of `src/` builds on the code Mengyuan Ma wrote for

> M. Ma, N. T. Nguyen, N. Shlezinger, Y. C. Eldar, A. L. Swindlehurst and M. Juntti, "Knowledge Distillation for
> Sensing-Assisted Long-Term Beam Tracking in mmWave Communications," *IEEE Trans. Wireless Commun.*, vol. 25,
> pp. 20494-20511, 2026.

It is included here with his permission; `src/DataFeed.py`, `src/MyFunc.py` and `src/model_image.py` keep his author line.

| File | Relation to the original code |
|---|---|
| `src/Radar_KPI.py` | unchanged |
| `src/MyFunc.py`, `src/model_image.py` | unchanged except that the contact line was removed from the header |
| `src/gen_data_seq.py` | unchanged except for the path of the repository root |
| `src/DataFeed.py` | modified: second and third radar maps, the environment switches of the input controls, labels from the arg max over measured entries |
| `src/model_cmkd.py`, `src/model_vision_only.py`, `src/preprocess_radar.py`, `src/preprocess_radar_rd.py` | written for this work after the structure of the original models and preprocessing |

The DeepSense 6G dataset is not redistributed; see `dataset/scenario9/README.md`.
