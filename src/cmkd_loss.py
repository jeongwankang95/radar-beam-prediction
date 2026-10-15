#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cmkd_loss.py: Composite Cross-Modal Knowledge Distillation Loss

Three distillation levels:
  1. Logit KD        : KL divergence on softened output distributions
  2. Feature alignment: MSE between teacher's fused features and projected student features
  3. Temporal attention: KL divergence on decoder-to-encoder attention maps (per head, per step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMKDLoss(nn.Module):
    """
    Total loss = L_task + λ1 * L_logit + λ2 * L_feat + λ3 * L_temp

    Args:
        task_criterion:  base task loss (e.g., FocalLoss or CrossEntropyLoss)
        lambda1:         weight for logit distillation
        lambda2:         weight for feature alignment
        lambda3:         weight for temporal attention alignment
        temperature:     softmax temperature for logit KD (Γ)
    """
    def __init__(self, task_criterion, lambda1=1.0, lambda2=1.0, lambda3=1.0,
                 temperature=3.0):
        super().__init__()
        self.task_criterion = task_criterion
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.temperature = temperature

    # -----------------------------------------------------------------
    # 1. Logit Distillation Loss
    # -----------------------------------------------------------------
    def logit_distillation_loss(self, student_logits, teacher_logits):
        """
        KL divergence between temperature-scaled softmax distributions.

        Args:
            student_logits: (N, C), flattened over batch and prediction steps
            teacher_logits: (N, C)
        Returns:
            scalar loss (scaled by Γ²)
        """
        T = self.temperature
        s_log_soft = F.log_softmax(student_logits / T, dim=-1)
        t_soft = F.softmax(teacher_logits / T, dim=-1)
        kl = F.kl_div(s_log_soft, t_soft, reduction='batchmean')
        return kl * (T ** 2)

    # -----------------------------------------------------------------
    # 2. Feature Alignment Loss
    # -----------------------------------------------------------------
    @staticmethod
    def feature_alignment_loss(student_projected, teacher_fused):
        """
        MSE between the teacher's fused multimodal features and the
        student's projected radar features.

        Args:
            student_projected: (B, num_past, D), ψ(f_S)
            teacher_fused:     (B, num_past, D), f_fused
        Returns:
            scalar MSE loss
        """
        return F.mse_loss(student_projected, teacher_fused)

    # -----------------------------------------------------------------
    # 3. Temporal Attention Alignment Loss
    # -----------------------------------------------------------------
    @staticmethod
    def temporal_attention_loss(student_attn, teacher_attn):
        """
        KL divergence between teacher and student decoder-to-encoder
        attention maps, averaged over heads and prediction steps.

        Handles L_S < L: if student has fewer past frames, align with
        the LAST L_S-1 columns of the teacher's attention map
        (most recent observation frames).

        Args:
            student_attn: (B, H, num_pred+1, L_S-1)
            teacher_attn: (B, H, num_pred+1, L-1)  where L-1 >= L_S-1
        Returns:
            scalar loss
        """
        B, H, J1, L_t = teacher_attn.shape
        L_s = student_attn.shape[-1]
        eps = 1e-8

        # Truncate teacher to match student's temporal dimension
        if L_s < L_t:
            teacher_attn = teacher_attn[:, :, :, -L_s:]  # take last L_s columns
            # Renormalize teacher attention over truncated dimension
            teacher_attn = teacher_attn / (teacher_attn.sum(dim=-1, keepdim=True) + eps)

        # Reshape to (B*H*J1, L_s) for KL computation
        t_flat = teacher_attn.reshape(-1, L_s)
        s_flat = student_attn.reshape(-1, L_s)

        # Ensure valid probability distributions (add eps, renormalize)
        t_flat = (t_flat + eps)
        t_flat = t_flat / t_flat.sum(dim=-1, keepdim=True)
        s_flat = (s_flat + eps)
        s_flat = s_flat / s_flat.sum(dim=-1, keepdim=True)

        # KL(teacher || student); PyTorch kl_div expects log-input
        kl = F.kl_div(s_flat.log(), t_flat, reduction='batchmean')
        return kl

    # -----------------------------------------------------------------
    # Forward: composite loss
    # -----------------------------------------------------------------
    def forward(self, student_logits, teacher_logits, targets,
                student_feat_projected=None, teacher_fused_feat=None,
                student_attn=None, teacher_attn=None):
        """
        Args:
            student_logits: (N, C), flattened student predictions
            teacher_logits: (N, C), flattened teacher predictions
            targets:        (N,), ground-truth beam indices
            student_feat_projected: (B, num_past, D), ψ(f_S)  [optional]
            teacher_fused_feat:     (B, num_past, D), f_fused  [optional]
            student_attn:   (B, H, J+1, L-1)                   [optional]
            teacher_attn:   (B, H, J+1, L-1)                   [optional]

        Returns:
            total_loss, task_loss, logit_loss, feat_loss, temp_loss
        """
        # --- Task loss ---
        task_loss = self.task_criterion(student_logits, targets)

        # --- Logit KD ---
        logit_loss = self.logit_distillation_loss(student_logits, teacher_logits)

        # --- Feature alignment ---
        if student_feat_projected is not None and teacher_fused_feat is not None:
            feat_loss = self.feature_alignment_loss(
                student_feat_projected, teacher_fused_feat.detach()
            )
        else:
            feat_loss = torch.tensor(0.0, device=student_logits.device)

        # --- Temporal attention alignment ---
        if student_attn is not None and teacher_attn is not None:
            temp_loss = self.temporal_attention_loss(
                student_attn, teacher_attn.detach()
            )
        else:
            temp_loss = torch.tensor(0.0, device=student_logits.device)

        # --- Composite ---
        total_loss = (task_loss
                      + self.lambda1 * logit_loss
                      + self.lambda2 * feat_loss
                      + self.lambda3 * temp_loss)

        return total_loss, task_loss, logit_loss, feat_loss, temp_loss


# =============================================================================
# Self-KD loss (for teacher refinement, Phase 2)
# =============================================================================
class SelfKDLoss(nn.Module):
    """
    L_self = (1 - β) * L_task + β * Γ_s² * KL(prev || current)
    """
    def __init__(self, task_criterion, beta=0.3, temperature=2.0):
        super().__init__()
        self.task_criterion = task_criterion
        self.beta = beta
        self.temperature = temperature

    def forward(self, logits, prev_logits, targets):
        """
        logits:      (N, C), current model output
        prev_logits: (N, C), previous checkpoint output (detached)
        targets:     (N,)
        """
        task_loss = self.task_criterion(logits, targets)

        T = self.temperature
        s_log = F.log_softmax(logits / T, dim=-1)
        t_soft = F.softmax(prev_logits / T, dim=-1)
        kd_loss = F.kl_div(s_log, t_soft, reduction='batchmean') * (T ** 2)

        total = (1 - self.beta) * task_loss + self.beta * kd_loss
        return total, task_loss, kd_loss


if __name__ == '__main__':
    from MyFunc import FocalLoss
    # Quick test
    B, J1, C = 4, 7, 64
    num_past, H = 7, 8

    criterion = CMKDLoss(
        task_criterion=FocalLoss(alpha=1, gamma=2),
        lambda1=1.0, lambda2=1.0, lambda3=1.0, temperature=3.0,
    )

    s_logits = torch.randn(B * J1, C)
    t_logits = torch.randn(B * J1, C)
    targets = torch.randint(0, C, (B * J1,))
    s_feat = torch.randn(B, num_past, 64)
    t_feat = torch.randn(B, num_past, 64)
    s_attn = torch.softmax(torch.randn(B, H, J1, num_past), dim=-1)
    t_attn = torch.softmax(torch.randn(B, H, J1, num_past), dim=-1)

    total, task, logit, feat, temp = criterion(
        s_logits, t_logits, targets, s_feat, t_feat, s_attn, t_attn
    )
    print(f"total={total:.4f}  task={task:.4f}  logit={logit:.4f}  "
          f"feat={feat:.4f}  temp={temp:.4f}")
