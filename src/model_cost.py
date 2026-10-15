#!/usr/bin/env python3
"""Parameters by component (supplementary Table S2) and multiply-accumulates per call (Sec. V-B) of the vision-radar
teacher, the radar-only students and the vision-only student.

MACs are counted with thop for one call on one window, batch 1: the seven observed frames followed by the seven slots
to predict (14 inputs, num_past = 7). Parameters are those of the model classes, which are the checkpoints' trainable
parameters (batch-norm running statistics excluded); the feature projector exists only for feature distillation and
is discarded at inference. Weights are random, since neither count depends on them.
Usage, from the repository root: python src/model_cost.py"""
import torch
from thop import profile
from model_cmkd import MultimodalTeacher, RadarOnlyStudent
from model_vision_only import VisionOnlyStudent

T, P = 14, 7
radar = lambda ch: torch.zeros(1, T, ch, 128, 64)
image = torch.zeros(1, T, 1, 224, 224)
ARMS = {
    "teacher": (MultimodalTeacher(64, 64, 64, gru_layers=2, num_heads_ca=4, num_heads_mha=8, radar_h=128, radar_w=64,
                                  radar_channels=2), (image, radar(2), P)),
    "radar-only RA+RD": (RadarOnlyStudent(64, 64, 64, gru_layers=1, num_heads_mha=8, radar_h=128, radar_w=64,
                                          radar_channels=2), (radar(2), P)),
    "radar-only RA": (RadarOnlyStudent(64, 64, 64, gru_layers=1, num_heads_mha=8, radar_h=128, radar_w=64,
                                       radar_channels=1), (radar(1), P)),
    "vision-only": (VisionOnlyStudent(64, 64, 64, gru_layers=1, num_heads_mha=8), (image, P)),
}
COMPONENTS = (("vision encoder (CNN + FC)", ("vision_enc",)), ("radar encoder (CNN + FC)", ("radar_enc",)),
              ("radar-query cross-attention", ("cross_attn",)), ("GRU", ("gru",)),
              ("decoder-encoder attention, positional tables", ("dec2enc_mha",)),
              ("classifier and norms", ("classifier", "layer_norm")))
count = lambda m, names: sum(p.numel() for n in names if hasattr(m, n) for p in getattr(m, n).parameters())

print(f"{'component':46s}" + "".join(f"{a:>18s}" for a in ARMS))
for label, names in COMPONENTS:
    print(f"{label:46s}" + "".join(f"{count(m, names):18,d}" for m, _ in ARMS.values()))
total = {a: sum(p.numel() for p in m.parameters()) for a, (m, _) in ARMS.items()}
proj = {a: count(m, ("feat_projector",)) for a, (m, _) in ARMS.items()}
print(f"{'total at inference':46s}" + "".join(f"{total[a] - proj[a]:18,d}" for a in ARMS))
print(f"{'projector (training only)':46s}" + "".join(f"{proj[a]:18,d}" for a in ARMS))
macs = {}
for a, (m, inputs) in ARMS.items():
    m.eval()
    with torch.no_grad():
        macs[a] = profile(m, inputs=inputs, verbose=False)[0]
print(f"{'MACs per call (millions)':46s}" + "".join(f"{macs[a] / 1e6:18.1f}" for a in ARMS))
