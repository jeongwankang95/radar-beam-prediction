#!/usr/bin/env python3
"""Vision-only student = ImageFeatureExtractor (motion masks) + GRU + Dec2EncMHA + head.
Mirrors RadarOnlyStudent exactly, swapping the radar encoder for the vision encoder, so
it is a same-protocol vision baseline trained on the same pass-level split."""
import torch, torch.nn as nn
from model_image import ImageFeatureExtractor
from model_cmkd import Dec2EncMHA


class ImageFeatureExtractorGAP(nn.Module):
    """Parameter-matched vision encoder (revision control): the same five conv blocks as
    ImageFeatureExtractor, but global average pooling replaces the 3136->512 flatten-FC,
    so the whole vision-only student carries about as many parameters as the radar-only
    student (~93k against 92.5k) instead of 1.76M."""
    def __init__(self, n_feature, in_channel=1):
        super().__init__()
        chans = [in_channel, 4, 8, 16, 32, 64]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=1, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.cnn_layers = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_feature))

    def forward(self, x):
        B, T = x.shape[:2]
        f = self.cnn_layers(x.reshape(B * T, *x.shape[2:]))
        f = self.gap(f).flatten(1)
        return self.fc_layer(f).view(B, T, -1)


class VisionOnlyStudent(nn.Module):
    def __init__(self, feature_size=64, num_classes=64, gru_hidden=64,
                 gru_layers=1, num_heads_mha=8, gap=False):
        super().__init__()
        self.name = 'VisionOnlyStudent'
        self.feature_size = feature_size
        self.vision_enc = (ImageFeatureExtractorGAP if gap else ImageFeatureExtractor)(feature_size, in_channel=1)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.gru = nn.GRU(input_size=feature_size, hidden_size=gru_hidden,
                          num_layers=gru_layers, dropout=0.0, batch_first=True)
        self.dec2enc_mha = Dec2EncMHA(gru_hidden, num_heads=num_heads_mha)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        self.feat_projector = nn.Linear(feature_size, feature_size)

    def forward(self, img_seq, num_past):
        B = img_seq.shape[0]; T_total = img_seq.shape[1]; num_pred_plus1 = T_total - num_past
        F_V = self.vision_enc(img_seq[:, :num_past])                 # (B, num_past, D)
        zero_pad = torch.zeros(B, num_pred_plus1, self.feature_size, device=F_V.device)
        full_seq = self.layer_norm(torch.cat([F_V, zero_pad], dim=1))
        gru_out, _ = self.gru(full_seq)
        enc_states = gru_out[:, :num_past, :]; dec_states = gru_out[:, num_past:, :]
        refined_dec, attn_map = self.dec2enc_mha(dec_states, enc_states)
        logits = self.classifier(torch.cat([enc_states, refined_dec], dim=1))
        return logits, F_V, attn_map
