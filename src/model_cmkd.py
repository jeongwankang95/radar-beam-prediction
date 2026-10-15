#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_cmkd.py: Cross-Modal Knowledge Distillation models
  - RadarFeatureExtractor: 3-layer CNN for Range-Angle maps
  - CrossAttentionFusion: radar-query-centric cross-attention
  - MultimodalTeacher: vision + radar → cross-attention → GRU → Dec-to-Enc MHA → prediction
  - RadarOnlyStudent: radar → GRU → Dec-to-Enc MHA → prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_image import ImageFeatureExtractor  # reuse Ma et al.'s vision CNN


# =============================================================================
# Radar Feature Extractor
# =============================================================================
class RadarFeatureExtractor(nn.Module):
    """
    3-layer CNN encoder for Range-Angle (or Doppler-Angle) radar maps.
    Input:  (B*T, 1, H_r, W_r)  e.g. (B*T, 1, 128, 64)
    Output: (B*T, n_feature)
    """
    def __init__(self, n_feature, in_channels=1, input_h=128, input_w=64,
                 preserve_angle=False):
        super().__init__()
        # The second map axis is angle, and the label is an angular index: the beam
        # codebook and the 64-point angle FFT lie on the same sin(theta) grid to within
        # one bin. The default encoder nevertheless reduces that axis 64 -> 32 -> 16 -> 8
        # and then averages it away, so angular position has to survive implicitly in the
        # channels. With preserve_angle the pooling acts on the range/velocity axis only,
        # all 64 angle bins reach the end, and a learned attention over them replaces the
        # global average.
        self.preserve_angle = preserve_angle
        pool = (2, 1) if preserve_angle else 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )
        if preserve_angle:
            # (B, 64, H/8, W) -> mean over range/velocity -> (B, 64, W) -> attention over W
            self.angle_attn = nn.Conv1d(64, 1, kernel_size=1)
        else:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (B, 64, H/8, W/8) → (B, 64, 1, 1)

        self.fc = nn.Sequential(
            # Flatten is a no-op on the already-2D preserve_angle output; it is kept so
            # the Linear layers hold the same state-dict indices as before this change,
            # and existing checkpoints keep loading.
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, n_feature),
        )

    def forward(self, x, return_attn=False):
        """
        x: (B, T, C, H, W), radar map sequence with channel dim
        returns: (B, T, n_feature), and with return_attn also the angular attention
                 logits (B, T, W) when preserve_angle is set (else None).
        """
        B, T, C, H, W = x.shape
        f = self.cnn(x.reshape(B * T, C, H, W))
        logits = None
        if self.preserve_angle:
            f = f.mean(dim=2)                       # (B*T, 64, W), collapse range only
            logits = self.angle_attn(f).squeeze(1)  # (B*T, W)
            alpha = torch.softmax(logits, dim=-1).unsqueeze(1)
            f = (f * alpha).sum(-1)                 # (B*T, 64)
        else:
            f = self.gap(f).flatten(1)              # (B*T, 64)
        out = self.fc(f).view(B, T, -1)
        if return_attn:
            return out, (logits.view(B, T, -1) if logits is not None else None)
        return out


# =============================================================================
# Radar-query-centric Cross-Attention Fusion
# =============================================================================
class RadarFeatureExtractorTwoStem(nn.Module):
    """Late-fusion control (revision, Sec. IV-A): one CNN stem per input map, each pooled
    to 64 features, concatenated and projected. The two maps share no common pixel grid
    (range-angle vs range-Doppler or Doppler-angle), so this variant never convolves them
    together. Same conv stack per stem as RadarFeatureExtractor; preserve_angle unsupported."""
    def __init__(self, n_feature, in_channels=2, input_h=128, input_w=64, preserve_angle=False):
        super().__init__()
        assert not preserve_angle, "two-stem encoder has no preserve_angle variant"
        def stem():
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.stems = nn.ModuleList([stem() for _ in range(in_channels)])
        self.fc = nn.Sequential(nn.Linear(64 * in_channels, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                nn.Linear(128, n_feature))

    def forward(self, x, return_attn=False):
        B, T, C, H, W = x.shape
        xs = x.reshape(B * T, C, H, W)
        f = torch.cat([s(xs[:, c:c + 1]) for c, s in enumerate(self.stems)], dim=1)
        out = self.fc(f).view(B, T, -1)
        return (out, None) if return_attn else out


class CrossAttentionFusion(nn.Module):
    """
    Radar features pose Q, vision features provide K & V.
    Residual connection preserves radar structure: output = LN(CA + F_R).
    Operates on the full (L-1)-length observation sequence.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, F_R, F_V):
        """
        F_R: (B, L-1, D)  radar features
        F_V: (B, L-1, D)  vision features
        returns: F_fused (B, L-1, D), attn_weights (B, num_heads, L-1, L-1)
        """
        ca_out, attn_weights = self.cross_attn(
            query=F_R, key=F_V, value=F_V, average_attn_weights=False,
        )
        F_fused = self.layer_norm(ca_out + F_R)  # residual on radar
        return F_fused, attn_weights


# =============================================================================
# Decoder-to-Encoder Multi-Head Attention
# =============================================================================
class Dec2EncMHA(nn.Module):
    """
    Cross-attention: Q from decoder states (predictions), K/V from encoder states (past).
    Produces attention map A ∈ R^{(num_pred+1) x (L-1)} per head, the target for
    temporal attention alignment during CMKD.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, max_enc=16, max_dec=16):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads
        # --- Learned positional encodings ---
        # Without absolute position, the zero-padded decoder states are near-identical
        # and the dec-to-enc attention collapses to a uniform map (nothing to distill).
        # Adding PE to Q (decoder slots) and K/V (encoder frames) makes them distinguishable.
        self.pe_enc = nn.Parameter(torch.zeros(1, max_enc, embed_dim))
        self.pe_dec = nn.Parameter(torch.zeros(1, max_dec, embed_dim))
        nn.init.trunc_normal_(self.pe_enc, std=0.02)
        nn.init.trunc_normal_(self.pe_dec, std=0.02)

    def forward(self, dec_states, enc_states):
        """
        dec_states: (B, num_pred+1, D'), decoder GRU outputs for current+future
        enc_states: (B, L-1, D'), encoder GRU outputs for past observations
        returns:
            refined:    (B, num_pred+1, D')
            attn_map:   (B, num_heads, num_pred+1, L-1)
        """
        Le, Ld = enc_states.size(1), dec_states.size(1)
        enc_p = enc_states + self.pe_enc[:, :Le]
        dec_p = dec_states + self.pe_dec[:, :Ld]
        attn_out, attn_map = self.mha(
            query=dec_p, key=enc_p, value=enc_p,
            average_attn_weights=False,
        )
        refined = self.layer_norm(attn_out + dec_states)  # residual on original dec_states
        return refined, attn_map


# =============================================================================
# Multimodal Teacher
# =============================================================================
class MultimodalTeacher(nn.Module):
    """
    Vision + Radar → Cross-Attention Fusion → GRU → Dec-to-Enc MHA → Classifier.
    """
    def __init__(self, feature_size=64, num_classes=64, gru_hidden=64,
                 gru_layers=2, num_heads_ca=4, num_heads_mha=8,
                 radar_h=128, radar_w=64, radar_channels=2, preserve_angle=False):
        super().__init__()
        self.name = 'MultimodalTeacher'
        self.feature_size = feature_size
        self.num_heads_mha = num_heads_mha

        # --- Modality-specific encoders ---
        self.vision_enc = ImageFeatureExtractor(feature_size, in_channel=1)
        self.radar_enc = RadarFeatureExtractor(feature_size, in_channels=radar_channels,  # [RA(,DA)]
                                                input_h=radar_h, input_w=radar_w,
                                                preserve_angle=preserve_angle)

        # --- Cross-attention fusion ---
        self.cross_attn = CrossAttentionFusion(feature_size, num_heads=num_heads_ca)

        # --- Temporal model ---
        self.layer_norm = nn.LayerNorm(feature_size)
        self.gru = nn.GRU(input_size=feature_size, hidden_size=gru_hidden,
                          num_layers=gru_layers, dropout=0.5, batch_first=True)
        self.dec2enc_mha = Dec2EncMHA(gru_hidden, num_heads=num_heads_mha)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, img_seq, radar_seq, num_past):
        """
        img_seq:   (B, T_total, 1, 224, 224), motion masks + zero-padded future
        radar_seq: (B, T_total, 1, H_r, W_r), radar maps + zero-padded future
        num_past:  int, number of valid past observation frames (L-1)

        returns:
            logits:       (B, T_total, C)
            fused_feat:   (B, num_past, D), for feature alignment KD
            attn_map:     (B, heads, num_pred+1, num_past), for temporal KD
        """
        B = img_seq.shape[0]
        T_total = img_seq.shape[1]
        num_pred_plus1 = T_total - num_past  # num_pred + 1

        # --- Modality embeddings (only for past frames) ---
        F_V = self.vision_enc(img_seq[:, :num_past])   # (B, num_past, D)
        F_R = self.radar_enc(radar_seq[:, :num_past])   # (B, num_past, D)

        # --- Cross-attention fusion ---
        F_fused, _ = self.cross_attn(F_R, F_V)          # (B, num_past, D)

        # --- Build full sequence: fused past + zero-padded future ---
        zero_pad = torch.zeros(B, num_pred_plus1, self.feature_size,
                               device=F_fused.device)
        full_seq = torch.cat([F_fused, zero_pad], dim=1)  # (B, T_total, D)

        # --- GRU ---
        full_seq = self.layer_norm(full_seq)
        gru_out, _ = self.gru(full_seq)                 # (B, T_total, D')

        # --- Split encoder / decoder states ---
        enc_states = gru_out[:, :num_past, :]            # (B, num_past, D')
        dec_states = gru_out[:, num_past:, :]            # (B, num_pred+1, D')

        # --- Decoder-to-Encoder MHA ---
        refined_dec, attn_map = self.dec2enc_mha(dec_states, enc_states)
        # attn_map: (B, heads, num_pred+1, num_past)

        # --- Reconstruct full output for classifier ---
        full_out = torch.cat([enc_states, refined_dec], dim=1)  # (B, T_total, D')

        # --- Predict ---
        logits = self.classifier(full_out)               # (B, T_total, C)

        return logits, F_fused, attn_map


# =============================================================================
# Radar-Only Student
# =============================================================================
class RadarOnlyStudent(nn.Module):
    """
    Radar CNN → GRU (1-layer) → Dec-to-Enc MHA → Classifier.
    Shares the same radar encoder architecture as the teacher.
    """
    def __init__(self, feature_size=64, num_classes=64, gru_hidden=64,
                 gru_layers=1, num_heads_mha=8,
                 radar_h=128, radar_w=64, radar_channels=2, preserve_angle=False,
                 two_stem=False):
        super().__init__()
        self.name = 'RadarOnlyStudent'
        self.feature_size = feature_size
        self.num_heads_mha = num_heads_mha

        # --- Radar encoder (same architecture as teacher) ---
        enc_cls = RadarFeatureExtractorTwoStem if two_stem else RadarFeatureExtractor
        self.radar_enc = enc_cls(feature_size, in_channels=radar_channels,  # [RA(,DA)]
                                 input_h=radar_h, input_w=radar_w,
                                 preserve_angle=preserve_angle)

        # --- Temporal model (lighter: 1-layer GRU) ---
        self.layer_norm = nn.LayerNorm(feature_size)
        self.gru = nn.GRU(input_size=feature_size, hidden_size=gru_hidden,
                          num_layers=gru_layers, dropout=0.0, batch_first=True)
        self.dec2enc_mha = Dec2EncMHA(gru_hidden, num_heads=num_heads_mha)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        # --- Projection for feature alignment KD (discarded at inference) ---
        _pk = globals().get("FEAT_PROJECTOR", "linear")
        if _pk == "mlp":
            self.feat_projector = nn.Sequential(
                nn.Linear(feature_size, 2 * feature_size),
                nn.ReLU(inplace=True),
                nn.Linear(2 * feature_size, feature_size))
        else:
            self.feat_projector = nn.Linear(feature_size, feature_size)

    def forward(self, radar_seq, num_past):
        """
        radar_seq: (B, T_total, 1, H_r, W_r)
        num_past:  int

        returns:
            logits:      (B, T_total, C)
            radar_feat:  (B, num_past, D), for feature alignment KD
            attn_map:    (B, heads, num_pred+1, num_past), for temporal KD
        """
        B = radar_seq.shape[0]
        T_total = radar_seq.shape[1]
        num_pred_plus1 = T_total - num_past

        # --- Radar embedding ---
        F_R = self.radar_enc(radar_seq[:, :num_past])  # (B, num_past, D)

        # --- Build full sequence ---
        zero_pad = torch.zeros(B, num_pred_plus1, self.feature_size,
                               device=F_R.device)
        full_seq = torch.cat([F_R, zero_pad], dim=1)

        # --- GRU ---
        full_seq = self.layer_norm(full_seq)
        gru_out, _ = self.gru(full_seq)

        # --- Split & Dec-to-Enc MHA ---
        enc_states = gru_out[:, :num_past, :]
        dec_states = gru_out[:, num_past:, :]
        refined_dec, attn_map = self.dec2enc_mha(dec_states, enc_states)

        # --- Reconstruct & predict ---
        full_out = torch.cat([enc_states, refined_dec], dim=1)
        logits = self.classifier(full_out)

        return logits, F_R, attn_map

    def project_features(self, feat):
        """Apply the learnable projection ψ for feature alignment KD."""
        return self.feat_projector(feat)


# =============================================================================
# Concat Fusion Teacher (baseline variant, Park et al. style)
# =============================================================================
class ConcatFusionTeacher(nn.Module):
    """
    Vision + Radar → Concatenation + Linear projection → GRU → Dec-to-Enc MHA.
    Modality-agnostic fusion (no cross-attention). Used as ablation baseline
    to demonstrate the advantage of radar-query-centric cross-attention.
    Same forward signature as MultimodalTeacher for drop-in comparison.
    """
    def __init__(self, feature_size=64, num_classes=64, gru_hidden=64,
                 gru_layers=2, num_heads_mha=8,
                 radar_h=128, radar_w=64, radar_channels=2, preserve_angle=False):
        super().__init__()
        self.name = 'ConcatFusionTeacher'
        self.feature_size = feature_size
        self.num_heads_mha = num_heads_mha

        # --- Modality-specific encoders ---
        self.vision_enc = ImageFeatureExtractor(feature_size, in_channel=1)
        self.radar_enc = RadarFeatureExtractor(feature_size, in_channels=radar_channels,  # [RA(,DA)]
                                                input_h=radar_h, input_w=radar_w,
                                                preserve_angle=preserve_angle)

        # --- Concat fusion: [F_V; F_R] → Linear → D ---
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
        )

        # --- Temporal model ---
        self.layer_norm = nn.LayerNorm(feature_size)
        self.gru = nn.GRU(input_size=feature_size, hidden_size=gru_hidden,
                          num_layers=gru_layers, dropout=0.5, batch_first=True)
        self.dec2enc_mha = Dec2EncMHA(gru_hidden, num_heads=num_heads_mha)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, img_seq, radar_seq, num_past):
        B = img_seq.shape[0]
        T_total = img_seq.shape[1]
        num_pred_plus1 = T_total - num_past

        F_V = self.vision_enc(img_seq[:, :num_past])   # (B, num_past, D)
        F_R = self.radar_enc(radar_seq[:, :num_past])   # (B, num_past, D)

        # --- Concat fusion ---
        F_cat = torch.cat([F_V, F_R], dim=-1)           # (B, num_past, 2D)
        F_fused = self.fusion_proj(F_cat)                # (B, num_past, D)

        # --- GRU ---
        zero_pad = torch.zeros(B, num_pred_plus1, self.feature_size,
                               device=F_fused.device)
        full_seq = torch.cat([F_fused, zero_pad], dim=1)
        full_seq = self.layer_norm(full_seq)
        gru_out, _ = self.gru(full_seq)

        enc_states = gru_out[:, :num_past, :]
        dec_states = gru_out[:, num_past:, :]
        refined_dec, attn_map = self.dec2enc_mha(dec_states, enc_states)

        full_out = torch.cat([enc_states, refined_dec], dim=1)
        logits = self.classifier(full_out)

        return logits, F_fused, attn_map


# =============================================================================
# Utility: parameter count
# =============================================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Quick sanity check
    device = 'cpu'
    B, L, J = 2, 8, 6
    num_past = L - 1  # 7 (after frame differencing)
    T_total = num_past + (J + 1)  # 7 + 7 = 14

    img = torch.randn(B, T_total, 1, 224, 224)
    radar = torch.randn(B, T_total, 1, 128, 64)

    teacher = MultimodalTeacher()
    t_logits, t_fused, t_attn = teacher(img, radar, num_past)
    print(f"Teacher logits: {t_logits.shape}")        # (2, 14, 64)
    print(f"Teacher fused:  {t_fused.shape}")          # (2, 7, 64)
    print(f"Teacher attn:   {t_attn.shape}")           # (2, 8, 7, 7)
    t_total, t_train = count_params(teacher)
    print(f"Teacher params:  {t_total:,} total, {t_train:,} trainable")

    concat_teacher = ConcatFusionTeacher()
    c_logits, c_fused, c_attn = concat_teacher(img, radar, num_past)
    c_total, c_train = count_params(concat_teacher)
    print(f"\nConcat Teacher params: {c_total:,} total, {c_train:,} trainable")

    student = RadarOnlyStudent()
    s_logits, s_feat, s_attn = student(radar, num_past)
    print(f"\nStudent logits: {s_logits.shape}")
    print(f"Student feat:   {s_feat.shape}")
    print(f"Student attn:   {s_attn.shape}")
    s_total, s_train = count_params(student)
    print(f"Student params:  {s_total:,} total, {s_train:,} trainable")
    print(f"Compression:     {t_total / s_total:.1f}x")
