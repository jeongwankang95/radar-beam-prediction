#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cmkd.py: training of the radar-only students and the vision-radar teacher (--phase)

Phase 0: radar-only student, task loss only (no distillation)
Phase 1: multimodal teacher, task loss only
Phase 2: teacher refined by self-distillation (--teacher_ckpt: its phase-1 checkpoint)
Phase 3: radar-only student distilled with the composite loss (--teacher_ckpt: a phase-2 teacher)
Each run writes <save_dir>/phase<k>_<date>/ with args.json, the checkpoint and eval_results.npz.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os, sys, json, shutil, time, argparse, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transf

from MyFunc import FocalLoss, calculate_topk_accuracy, calculate_dba_score, set_seed, plot_training_curves
from model_cmkd import MultimodalTeacher, RadarOnlyStudent, ConcatFusionTeacher, count_params
from cmkd_loss import CMKDLoss, SelfKDLoss
from DataFeed import DataFeed


class RadarTeacher(RadarOnlyStudent):
    """Radar-only teacher control: a radar-only student used as the distillation source. Same call
    signature as MultimodalTeacher (the image input is ignored); loads a phase-0 radar_baseline.pth."""
    def forward(self, img, radar_seq, num_past):
        return super().forward(radar_seq, num_past)


# =============================================================================
# Argument parser
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description='CMKD Training')

    # --- Phase selection ---
    p.add_argument('--phase', type=int, default=3, choices=[0, 1, 2, 3],
                   help='Training phase: 0=radar-only baseline (no KD), 1=teacher, 2=self-KD, 3=CMKD student')

    # --- Common training params ---
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--two_stem', action='store_true',
                   help='one CNN stem per radar map, fused after pooling (late-fusion control)')

    # --- Scheduler ---
    p.add_argument('--T_0', type=int, default=10)
    p.add_argument('--T_mult', type=int, default=2)
    p.add_argument('--eta_min', type=float, default=1e-6)

    # --- Model ---
    p.add_argument('--feature_size', type=int, default=64)
    p.add_argument('--gru_hidden', type=int, default=64)
    p.add_argument('--num_classes', type=int, default=64)
    p.add_argument('--num_heads_ca', type=int, default=4, help='Cross-attn heads')
    p.add_argument('--num_heads_mha', type=int, default=8, help='Dec-to-Enc MHA heads')
    p.add_argument('--gru_layers_teacher', type=int, default=2)
    p.add_argument('--gru_layers_student', type=int, default=1)
    p.add_argument('--teacher_type', type=str, default='cross_attn',
                   choices=['cross_attn', 'concat', 'radar'],
                   help='Teacher fusion type: cross_attn (proposed) or concat (baseline)')

    # --- Sequence ---
    p.add_argument('--seq_len', type=int, default=8, help='Raw observation window L')
    p.add_argument('--student_seq_len', type=int, default=0,
                   help='Student observation window L_S (0 = same as seq_len)')
    p.add_argument('--num_pred', type=int, default=6, help='Future prediction steps J')

    # --- Radar ---
    p.add_argument('--radar_h', type=int, default=128,
                   help='Radar map height after range-bin clipping (DataFeed clips 256→128)')
    p.add_argument('--radar_w', type=int, default=64, help='Radar map width (angle bins)')
    p.add_argument('--radar_channels', type=int, default=2, choices=[1, 2, 3],
                   help='1 = range-angle only (ablation baseline); 2 = [RA, DA] (Doppler)')
    p.add_argument('--student_radar_channels', type=int, default=0, choices=[0, 1, 2],
                   help='Radar channels for the STUDENT only (0 = same as --radar_channels). '
                        'Use 1 with --radar_channels 2 to distil a 2-ch teacher into a '
                        'range-angle-only student; the batch stays 2-ch and the student is fed '
                        'its first channel.')
    p.add_argument('--preserve_angle', action='store_true',
                   help='Pool the radar maps along range/velocity only, keeping all 64 '
                        'angle bins to the end, and replace the global average with a '
                        'learned attention over them. The beam index is an angular index '
                        'on the same grid, which the default encoder averages away.')
    p.add_argument('--single_channel', type=str, default='RA',
                   choices=['RA', 'second'],
                   help="with radar_channels=1, which map to use")
    p.add_argument('--extra_map', type=str, default='', choices=['', 'DA', 'RD'],
                   help='third radar map, stacked after --second_map; requires '
                        '--radar_channels 3')
    p.add_argument('--second_map', type=str, default='DA', choices=['DA', 'RD'],
                   help='Second radar map paired with the range-angle map: DA '
                        '(Doppler-angle; angle-resolved) or RD (range-Doppler, the reference '
                        'student; angle collapsed). Both are 128x64.')

    # --- CMKD (Phase 3) ---
    p.add_argument('--lambda1', type=float, default=1.0, help='Logit KD weight')
    p.add_argument('--lambda2', type=float, default=1.0, help='Feature alignment weight')
    p.add_argument('--lambda3', type=float, default=1.0, help='Temporal attention weight')
    p.add_argument('--temperature', type=float, default=3.0, help='CMKD temperature Γ')

    # --- Self-KD (Phase 2) ---
    p.add_argument('--beta', type=float, default=0.3, help='Self-KD mixing weight')
    p.add_argument('--temperature_self', type=float, default=2.0, help='Self-KD temperature Γ_s')

    # --- Paths ---
    p.add_argument('--data_root', type=str, default='dataset/scenario9')
    p.add_argument('--train_csv', type=str, default='train_seqs_RA_tr.csv')
    p.add_argument('--val_csv', type=str, default='val_seqs_RA.csv',
                   help='Held-out validation split (early stopping / model selection). '
                        'Never the test set.')
    p.add_argument('--test_csv', type=str, default='test_seqs_RA.csv')
    p.add_argument('--save_dir', type=str, default='cmkd_results')
    p.add_argument('--teacher_ckpt', type=str, default='', help='Path to teacher .pth (for Phase 2/3)')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--debug', action='store_true', default=False)

    p.add_argument("--feat_projector", type=str, default="linear",
                   choices=["linear", "mlp"],
                   help="shape of the feature-alignment projector (supplementary S10)")
    p.add_argument('--zero_second', action='store_true',
                   help='matched-capacity control: keep the two-channel graph but feed an exactly zero second channel')
    return p.parse_args()


# =============================================================================
# Data preparation helpers
# =============================================================================
def prepare_batch(img, radar_RA, radar_DA, beam, label, args, device):
    """
    Prepare a training batch: extract motion masks & radar maps,
    concatenate with zero padding for future prediction steps.

    Returns:
        img_batch:   (B, T_total, 1, 224, 224)
        radar_batch: (B, T_total, 1, H_r, W_r)
        targets:     (B, num_pred+1)
        num_past:    int
    """
    B = img.shape[0]
    num_past = args.seq_len - 1  # L-1 motion masks
    num_pred_plus1 = args.num_pred + 1  # current + J future
    T_total = num_past + num_pred_plus1

    # --- Vision: motion masks (B, L-1, 224, 224) → add channel dim ---
    img = img.unsqueeze(2)  # (B, L-1, 1, 224, 224)
    img_pad = torch.zeros(B, num_pred_plus1, 1, 224, 224)
    img_batch = torch.cat([img, img_pad], dim=1).to(device)  # (B, T_total, 1, 224, 224)

    # --- Radar: [RA] or [RA, DA], frames 1..L-1 (aligned with motion masks) + pad ---
    C = getattr(args, 'radar_channels', 2)
    ra = radar_RA[:, 1:, :, :]  # (B, L-1, H, W), drop first frame for temporal alignment
    if C == 3:
        # radar_DA arrives as (B, L, 2, H, W) when DataFeed was given an extra map
        radar = torch.cat([ra.unsqueeze(2), radar_DA[:, 1:]], dim=2)   # (B, L-1, 3, H, W)
    elif C == 2:
        da = radar_DA[:, 1:, :, :]  # (B, L-1, H, W)
        if getattr(args, 'zero_second', False):
            da = torch.zeros_like(da)   # matched-capacity control
        radar = torch.stack([ra, da], dim=2)  # (B, L-1, 2, H, W), RA & DA as channels
    elif getattr(args, 'single_channel', 'RA') == 'second':
        # the second map alone: velocity REPLACING range-angle, not augmenting it
        radar = radar_DA[:, 1:, :, :].unsqueeze(2)
    else:
        radar = ra.unsqueeze(2)  # (B, L-1, 1, H, W), range-angle only (ablation baseline)
    radar_pad = torch.zeros(B, num_pred_plus1, C, args.radar_h, args.radar_w)
    radar_batch = torch.cat([radar, radar_pad], dim=1).to(device)  # (B, T_total, C, H, W)

    # --- Labels: [current_beam, future_beam_1, ..., future_beam_J] ---
    beam_last = beam[:, -1:]  # last past beam = current beam
    targets = torch.cat([beam_last, label[:, :args.num_pred]], dim=-1).to(device)

    return img_batch, radar_batch, targets, num_past


def prepare_batch_student(radar_RA, args, device, student_seq_len):
    """
    Prepare a shorter radar batch for student with L_S < L.
    Takes the last L_S frames from radar_RA, drops first for alignment → L_S-1 past.

    Args:
        radar_RA: (B, L, H, W), full radar sequence from DataFeed
        student_seq_len: L_S, student's observation window
    Returns:
        radar_batch: (B, T_total_S, 1, H_r, W_r)
        num_past_s:  int (L_S - 1)
    """
    B = radar_RA.shape[0]
    num_past_s = student_seq_len - 1
    num_pred_plus1 = args.num_pred + 1

    # Take last L_S frames, then drop first for alignment → L_S-1
    radar = radar_RA[:, -(student_seq_len):, :, :]  # (B, L_S, H, W)
    radar = radar[:, 1:, :, :]                       # (B, L_S-1, H, W)
    radar = radar.unsqueeze(2)                        # (B, L_S-1, 1, H, W)
    radar_pad = torch.zeros(B, num_pred_plus1, 1, args.radar_h, args.radar_w)
    radar_batch = torch.cat([radar, radar_pad], dim=1).to(device)

    return radar_batch, num_past_s


# =============================================================================
# Phase 1: Train multimodal teacher
# =============================================================================
def train_teacher(model, dataloaders, args, device, save_path):
    print("=" * 60)
    print("PHASE 1: Training Multimodal Teacher (task loss only)")
    print("=" * 60)

    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)

    best_val_adba = -1.0
    epochs_no_improve = 0
    num_past = args.seq_len - 1

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0

        with tqdm(dataloaders['train'], unit='batch', file=sys.stdout) as pbar:
            for i, (img, radar_RA, radar_DA, beam, label) in enumerate(pbar):
                pbar.set_description(f"Phase1 Epoch {epoch}")
                img_b, radar_b, targets, np_ = prepare_batch(img, radar_RA, radar_DA, beam, label, args, device)

                optimizer.zero_grad()
                logits, _, _ = model(img_b, radar_b, np_)

                # Extract prediction logits: last (num_pred+1) steps
                pred_logits = logits[:, -(args.num_pred + 1):, :]
                pred_flat = pred_logits.reshape(-1, args.num_classes)
                tgt_flat = targets.flatten()

                loss = criterion(pred_flat, tgt_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                acc = (pred_logits.argmax(-1) == targets).float().mean().item()
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_acc = (acc + i * running_acc) / (i + 1)
                pbar.set_postfix(loss=running_loss, acc=running_acc)

        scheduler.step()

        # --- Validation (selection on ADBA) ---
        val_loss, val_adba = validate(model, dataloaders['val'], criterion, args, device, is_teacher=True)
        print(f"  Val loss: {val_loss:.4f}  Val ADBA: {100*val_adba:.2f}% (best: {100*best_val_adba:.2f}%)")

        if val_adba > best_val_adba + 1e-5:
            best_val_adba = val_adba
            torch.save(model.state_dict(), os.path.join(save_path, 'teacher_phase1.pth'))
            print(f"  → Saved best teacher (Phase 1)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    return best_val_adba


# =============================================================================
# Phase 2: Self-KD refinement
# =============================================================================
def train_teacher_selfkd(model, dataloaders, args, device, save_path):
    print("=" * 60)
    print("PHASE 2: Teacher Self-KD Refinement")
    print("=" * 60)

    task_criterion = FocalLoss(alpha=1, gamma=2)
    selfkd_criterion = SelfKDLoss(task_criterion, beta=args.beta, temperature=args.temperature_self)

    # Load Phase 1 checkpoint for generating soft targets
    if args.teacher_type == 'concat':
        prev_model = ConcatFusionTeacher(
            feature_size=args.feature_size, num_classes=args.num_classes,
            gru_hidden=args.gru_hidden, gru_layers=args.gru_layers_teacher,
            num_heads_mha=args.num_heads_mha,
            radar_h=args.radar_h, radar_w=args.radar_w,
            radar_channels=args.radar_channels,
            preserve_angle=args.preserve_angle,
        ).to(device)
    else:
        prev_model = MultimodalTeacher(
            feature_size=args.feature_size, num_classes=args.num_classes,
            gru_hidden=args.gru_hidden, gru_layers=args.gru_layers_teacher,
            num_heads_ca=args.num_heads_ca, num_heads_mha=args.num_heads_mha,
            radar_h=args.radar_h, radar_w=args.radar_w,
            radar_channels=args.radar_channels,
            preserve_angle=args.preserve_angle,
        ).to(device)
    phase1_path = os.path.join(save_path, 'teacher_phase1.pth')
    prev_model.load_state_dict(torch.load(phase1_path, map_location=device))
    prev_model.eval()
    for p in prev_model.parameters():
        p.requires_grad = False
    print(f"  Loaded Phase 1 teacher from {phase1_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)

    best_val_adba = -1.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        with tqdm(dataloaders['train'], unit='batch', file=sys.stdout) as pbar:
            for i, (img, radar_RA, radar_DA, beam, label) in enumerate(pbar):
                pbar.set_description(f"Phase2 Epoch {epoch}")
                img_b, radar_b, targets, np_ = prepare_batch(img, radar_RA, radar_DA, beam, label, args, device)

                optimizer.zero_grad()
                logits, _, _ = model(img_b, radar_b, np_)
                with torch.no_grad():
                    prev_logits, _, _ = prev_model(img_b, radar_b, np_)

                pred_logits = logits[:, -(args.num_pred + 1):, :].reshape(-1, args.num_classes)
                prev_pred = prev_logits[:, -(args.num_pred + 1):, :].reshape(-1, args.num_classes)
                tgt_flat = targets.flatten()

                total, task, kd = selfkd_criterion(pred_logits, prev_pred, tgt_flat)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                running_loss = (total.item() + i * running_loss) / (i + 1)
                pbar.set_postfix(total=running_loss, task=task.item(), kd=kd.item())

        scheduler.step()

        val_loss, val_adba = validate(model, dataloaders['val'], task_criterion, args, device, is_teacher=True)
        print(f"  Val loss: {val_loss:.4f}  Val ADBA: {100*val_adba:.2f}% (best: {100*best_val_adba:.2f}%)")

        if val_adba > best_val_adba + 1e-5:
            best_val_adba = val_adba
            torch.save(model.state_dict(), os.path.join(save_path, 'teacher_selfkd.pth'))
            print(f"  → Saved best teacher (Phase 2 self-KD)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    return best_val_adba


# =============================================================================
# Phase 3: Train radar-only student via CMKD
# =============================================================================
def train_student_cmkd(teacher, student, dataloaders, args, device, save_path):
    student_seq_len = args.student_seq_len if args.student_seq_len > 0 else args.seq_len
    short_seq = (student_seq_len < args.seq_len)
    if short_seq and args.radar_channels != 1:
        raise NotImplementedError(
            'prepare_batch_student() builds a range-angle-only batch; it would silently '
            'drop the Doppler channel. Use --radar_channels 1, or extend it to take radar_DA.')

    print("=" * 60)
    print("PHASE 3: Student Training via CMKD")
    print(f"  λ1={args.lambda1}, λ2={args.lambda2}, λ3={args.lambda3}, Γ={args.temperature}")
    if short_seq:
        print(f"  Student seq_len: L_S={student_seq_len} (teacher L={args.seq_len})")
    print("=" * 60)

    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    task_criterion = FocalLoss(alpha=1, gamma=2)
    cmkd_criterion = CMKDLoss(
        task_criterion=task_criterion,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        temperature=args.temperature,
    )

    # Optimizer includes student params + projection layer
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)

    best_val_adba = -1.0
    epochs_no_improve = 0
    history = {'total': [], 'task': [], 'logit': [], 'feat': [], 'temp': [], 'val': [], 'val_adba': []}

    for epoch in range(args.epochs):
        student.train()
        r_total, r_task, r_logit, r_feat, r_temp = 0., 0., 0., 0., 0.

        with tqdm(dataloaders['train'], unit='batch', file=sys.stdout) as pbar:
            for i, (img, radar_RA, radar_DA, beam, label) in enumerate(pbar):
                pbar.set_description(f"Phase3 Epoch {epoch}")
                img_b, radar_b, targets, np_ = prepare_batch(img, radar_RA, radar_DA, beam, label, args, device)

                optimizer.zero_grad()

                # --- Teacher forward (no grad, full sequence) ---
                with torch.no_grad():
                    t_logits, t_fused, t_attn = teacher(img_b, radar_b, np_)
                    t_pred = t_logits[:, -(args.num_pred + 1):, :]

                # --- Student forward (possibly shorter sequence) ---
                if short_seq:
                    radar_b_s, np_s = prepare_batch_student(radar_RA, args, device, student_seq_len)
                    s_logits, s_feat, s_attn = student(radar_b_s, np_s)
                else:
                    s_ch = args.student_radar_channels or args.radar_channels
                    s_in = radar_b if s_ch == radar_b.shape[2] else radar_b[:, :, :s_ch]
                    s_logits, s_feat, s_attn = student(s_in, np_)
                s_pred = s_logits[:, -(args.num_pred + 1):, :]

                # --- Projected student features for feature alignment ---
                s_feat_proj = student.project_features(s_feat)

                # --- Feature alignment: align with last L_S-1 teacher features ---
                if short_seq:
                    num_past_s = student_seq_len - 1
                    t_fused_aligned = t_fused[:, -num_past_s:, :]  # last L_S-1 frames
                else:
                    t_fused_aligned = t_fused

                # --- Flatten for loss ---
                s_flat = s_pred.reshape(-1, args.num_classes)
                t_flat = t_pred.reshape(-1, args.num_classes)
                tgt_flat = targets.flatten()

                # --- Composite CMKD loss ---
                total, task, logit, feat, temp = cmkd_criterion(
                    s_flat, t_flat, tgt_flat,
                    s_feat_proj, t_fused_aligned,
                    s_attn, t_attn,
                )

                total.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                optimizer.step()

                # Running averages
                r_total = (total.item() + i * r_total) / (i + 1)
                r_task = (task.item() + i * r_task) / (i + 1)
                r_logit = (logit.item() + i * r_logit) / (i + 1)
                r_feat = (feat.item() + i * r_feat) / (i + 1)
                r_temp = (temp.item() + i * r_temp) / (i + 1)

                pbar.set_postfix(OrderedDict(
                    tot=f'{r_total:.3f}', tsk=f'{r_task:.3f}',
                    lgt=f'{r_logit:.3f}', fea=f'{r_feat:.3f}', tmp=f'{r_temp:.3f}'))

        scheduler.step()
        history['total'].append(r_total)
        history['task'].append(r_task)
        history['logit'].append(r_logit)
        history['feat'].append(r_feat)
        history['temp'].append(r_temp)

        # --- Validation ---
        val_loss, val_adba = validate(student, dataloaders['val'],
                                      FocalLoss(alpha=1, gamma=2), args, device, is_teacher=False)
        history['val'].append(val_loss)
        history['val_adba'].append(val_adba)
        print(f"  Val loss: {val_loss:.4f}  Val ADBA: {100*val_adba:.2f}% (best: {100*best_val_adba:.2f}%)")

        if val_adba > best_val_adba + 1e-5:
            best_val_adba = val_adba
            torch.save(student.state_dict(), os.path.join(save_path, 'student_cmkd_best.pth'))
            print(f"  → Saved best student (CMKD)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save history
    np.savez(os.path.join(save_path, 'cmkd_history.npz'),
             **{k: np.array(v) for k, v in history.items()})

    return best_val_adba, history


# =============================================================================
# Validation
# =============================================================================
@torch.no_grad()
def validate(model, dataloader, criterion, args, device, is_teacher=False):
    """Returns (mean task loss, mean ADBA) on the validation split.

    Model selection and early stopping use ADBA, one of the reported metrics,
    rather than the focal loss. Logit distillation trains the student to match the
    teacher's softened distribution, which raises the focal loss even when beam
    *ranking* improves; selecting on the loss would therefore systematically penalise
    the distilled students relative to the no-KD baselines.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_outputs, all_labels = [], []

    for img, radar_RA, radar_DA, beam, label in dataloader:
        img_b, radar_b, targets, np_ = prepare_batch(img, radar_RA, radar_DA, beam, label, args, device)

        if is_teacher:
            logits, _, _ = model(img_b, radar_b, np_)
        else:
            s_ch = getattr(args, 'student_radar_channels', 0) or args.radar_channels
            r_in = radar_b if s_ch == radar_b.shape[2] else radar_b[:, :, :s_ch]
            logits, _, _ = model(r_in, np_)

        pred3 = logits[:, -(args.num_pred + 1):, :]          # (B, J+1, C)
        loss = criterion(pred3.reshape(-1, args.num_classes), targets.flatten())
        total_loss += loss.item()
        n_batches += 1
        all_outputs.append(pred3.cpu())
        all_labels.append(targets.cpu())

    dba = calculate_dba_score(torch.cat(all_outputs, 0), torch.cat(all_labels, 0), delta=5)
    return total_loss / max(n_batches, 1), float(np.mean(dba))


# =============================================================================
# Full evaluation with Top-K and DBA
# =============================================================================
@torch.no_grad()
def evaluate_full(model, dataloader, args, device, is_teacher=False, save_path=None):
    model.eval()
    all_outputs, all_labels = [], []

    for img, radar_RA, radar_DA, beam, label in dataloader:
        img_b, radar_b, targets, np_ = prepare_batch(img, radar_RA, radar_DA, beam, label, args, device)

        if is_teacher:
            logits, _, _ = model(img_b, radar_b, np_)
        else:
            s_ch = getattr(args, 'student_radar_channels', 0) or args.radar_channels
            r_in = radar_b if s_ch == radar_b.shape[2] else radar_b[:, :, :s_ch]
            logits, _, _ = model(r_in, np_)

        pred = logits[:, -(args.num_pred + 1):, :]
        all_outputs.append(pred.cpu())
        all_labels.append(targets.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)  # (M, J+1, C)
    all_labels = torch.cat(all_labels, dim=0)    # (M, J+1)

    # Top-K accuracy: Ma et al.'s function expects (M, J+1, C) and (M, J+1)
    topk_acc, _ = calculate_topk_accuracy(
        all_outputs,   # (M, J+1, C), keep 3D!
        all_labels,    # (M, J+1)
        k_values=[1, 3, 5, 10],
    )

    # DBA score: same shape requirement
    dba = calculate_dba_score(
        all_outputs,   # (M, J+1, C)
        all_labels,    # (M, J+1)
        delta=5,
    )

    print("\n=== Evaluation Results ===")
    for k in [1, 3, 5]:
        acc_str = ', '.join([f't{j}={topk_acc[k][j]*100:.1f}%' for j in range(len(topk_acc[k]))])
        avg = np.mean(topk_acc[k]) * 100
        print(f"  ATop-{k}: {avg:.2f}%  [{acc_str}]")

    avg_dba = np.mean(dba) * 100
    dba_str = ', '.join([f't{j}={dba[j]*100:.1f}%' for j in range(len(dba))])
    print(f"  ADBA:    {avg_dba:.2f}%  [{dba_str}]")

    if save_path:
        results = {'topk_acc': topk_acc, 'dba': dba}
        np.savez(os.path.join(save_path, 'eval_results.npz'), **{
            f'top{k}': np.array(topk_acc[k]) for k in topk_acc
        }, dba=dba)

    return topk_acc, dba


# =============================================================================
# Utility: find most recent checkpoint in save_dir subdirectories
# =============================================================================
def find_latest_checkpoint(save_dir, filename):
    """
    Search save_dir and its subdirectories for the given filename.
    Returns the most recently modified match.
    """
    candidates = []
    # Check flat path first
    flat_path = os.path.join(save_dir, filename)
    if os.path.isfile(flat_path):
        candidates.append(flat_path)
    # Check subdirectories (phase1_MMDD_HHMM/, phase2_MMDD_HHMM/, etc.)
    if os.path.isdir(save_dir):
        for subdir in sorted(os.listdir(save_dir), reverse=True):
            sub_path = os.path.join(save_dir, subdir, filename)
            if os.path.isfile(sub_path):
                candidates.append(sub_path)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint '{filename}' found in {save_dir}. "
            f"Please specify --teacher_ckpt explicitly.")
    # Return most recent by modification time
    best = max(candidates, key=os.path.getmtime)
    return best


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    import model_cmkd as _mc
    _mc.FEAT_PROJECTOR = getattr(args, "feat_projector", "linear")
    set_seed(args.seed)

    # --- Paths ---
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
    data_root = os.path.join(current_dir, args.data_root)
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    save_path = os.path.join(current_dir, args.save_dir, f'phase{args.phase}_{timestamp}')
    os.makedirs(save_path, exist_ok=True)

    # Debug mode overrides
    if args.debug:
        args.epochs = 3
        print("[DEBUG] Epochs reduced to 3")

    # Save args
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # --- Data ---
    proc_pipe = transf.Compose([transf.ToPILImage(), transf.Resize((224, 224))])

    train_loader = DataLoader(
        DataFeed(data_root, os.path.join(data_root, args.train_csv),
                 seq_len=args.seq_len, transform=proc_pipe,
                 clipped_range=args.radar_h, second_map=args.second_map,
                 extra_map=getattr(args, 'extra_map', '')),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        DataFeed(data_root, os.path.join(data_root, args.val_csv),
                 seq_len=args.seq_len, transform=proc_pipe,
                 clipped_range=args.radar_h, second_map=args.second_map,
                 extra_map=getattr(args, 'extra_map', '')),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        DataFeed(data_root, os.path.join(data_root, args.test_csv),
                 seq_len=args.seq_len, transform=proc_pipe,
                 clipped_range=args.radar_h, second_map=args.second_map,
                 extra_map=getattr(args, 'extra_map', '')),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    # Model selection / early stopping uses 'val' only; 'test' is touched exclusively
    # by evaluate_full() after training completes.
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # --- Common teacher kwargs ---
    teacher_kwargs = dict(
        feature_size=args.feature_size, num_classes=args.num_classes,
        gru_hidden=args.gru_hidden, gru_layers=args.gru_layers_teacher,
        num_heads_ca=args.num_heads_ca, num_heads_mha=args.num_heads_mha,
        radar_h=args.radar_h, radar_w=args.radar_w,
        radar_channels=args.radar_channels,
        preserve_angle=args.preserve_angle,
    )
    concat_kwargs = dict(
        feature_size=args.feature_size, num_classes=args.num_classes,
        gru_hidden=args.gru_hidden, gru_layers=args.gru_layers_teacher,
        num_heads_mha=args.num_heads_mha,
        radar_h=args.radar_h, radar_w=args.radar_w,
        radar_channels=args.radar_channels,
        preserve_angle=args.preserve_angle,
    )

    def build_teacher():
        if args.teacher_type == 'radar':
            return RadarTeacher(
                feature_size=args.feature_size, num_classes=args.num_classes,
                gru_hidden=args.gru_hidden, gru_layers=args.gru_layers_student,
                num_heads_mha=args.num_heads_mha, radar_h=args.radar_h, radar_w=args.radar_w,
                radar_channels=args.radar_channels, preserve_angle=args.preserve_angle)
        if args.teacher_type == 'concat':
            return ConcatFusionTeacher(**concat_kwargs)
        else:
            return MultimodalTeacher(**teacher_kwargs)

    # =================================================================
    # PHASE 0: Radar-only baseline (no KD, task loss only)
    # =================================================================
    if args.phase == 0:
        student = RadarOnlyStudent(
            feature_size=args.feature_size, num_classes=args.num_classes,
            gru_hidden=args.gru_hidden, gru_layers=args.gru_layers_student,
            num_heads_mha=args.num_heads_mha,
            radar_h=args.radar_h, radar_w=args.radar_w,
            radar_channels=(args.student_radar_channels or args.radar_channels),
            preserve_angle=args.preserve_angle,
            two_stem=getattr(args, 'two_stem', False),
        ).to(device)
        s_total, s_train = count_params(student)
        print(f"Radar-only baseline params: {s_total:,}")

        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)

        best_val_adba = -1.0
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            student.train()
            running_loss, running_acc = 0.0, 0.0

            with tqdm(dataloaders['train'], unit='batch', file=sys.stdout) as pbar:
                for i, (img, radar_RA, radar_DA, beam, label) in enumerate(pbar):
                    pbar.set_description(f"Phase0 Epoch {epoch}")
                    _, radar_b, targets, np_ = prepare_batch(img, radar_RA, radar_DA, beam, label, args, device)

                    optimizer.zero_grad()
                    logits, _, _ = student(radar_b, np_)
                    pred_logits = logits[:, -(args.num_pred + 1):, :]
                    pred_flat = pred_logits.reshape(-1, args.num_classes)
                    tgt_flat = targets.flatten()

                    loss = criterion(pred_flat, tgt_flat)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                    optimizer.step()

                    acc = (pred_logits.argmax(-1) == targets).float().mean().item()
                    running_loss = (loss.item() + i * running_loss) / (i + 1)
                    running_acc = (acc + i * running_acc) / (i + 1)
                    pbar.set_postfix(loss=running_loss, acc=running_acc)

            scheduler.step()
            val_loss, val_adba = validate(student, dataloaders['val'], criterion, args, device, is_teacher=False)
            print(f"  Val loss: {val_loss:.4f}  Val ADBA: {100*val_adba:.2f}% (best: {100*best_val_adba:.2f}%)")

            if val_adba > best_val_adba + 1e-5:
                best_val_adba = val_adba
                torch.save(student.state_dict(), os.path.join(save_path, 'radar_baseline.pth'))
                print(f"  → Saved best radar-only baseline")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        print("\n=== Radar-Only Baseline Evaluation ===")
        student.load_state_dict(torch.load(os.path.join(save_path, 'radar_baseline.pth'), map_location=device))
        evaluate_full(student, test_loader, args, device, is_teacher=False, save_path=save_path)

    # =================================================================
    # PHASE 1: Teacher pre-training
    # =================================================================
    elif args.phase == 1:
        teacher = build_teacher().to(device)
        t_total, t_train = count_params(teacher)
        print(f"Teacher params: {t_total:,}")

        train_teacher(teacher, dataloaders, args, device, save_path)
        print("\n=== Teacher Evaluation (Phase 1) ===")
        teacher.load_state_dict(torch.load(os.path.join(save_path, 'teacher_phase1.pth'), map_location=device))
        evaluate_full(teacher, test_loader, args, device, is_teacher=True, save_path=save_path)

    # =================================================================
    # PHASE 2: Self-KD
    # =================================================================
    elif args.phase == 2:
        teacher = build_teacher().to(device)

        # Load Phase 1 checkpoint
        if args.teacher_ckpt:
            ckpt_path = args.teacher_ckpt
        else:
            # Search for most recent Phase 1 checkpoint
            ckpt_path = find_latest_checkpoint(
                os.path.join(current_dir, args.save_dir), 'teacher_phase1.pth')
        teacher.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded teacher from {ckpt_path}")

        # Copy Phase 1 checkpoint to save_path for self-KD reference
        shutil.copy(ckpt_path, os.path.join(save_path, 'teacher_phase1.pth'))

        train_teacher_selfkd(teacher, dataloaders, args, device, save_path)
        print("\n=== Teacher Evaluation (Phase 2 self-KD) ===")
        teacher.load_state_dict(torch.load(os.path.join(save_path, 'teacher_selfkd.pth'), map_location=device))
        evaluate_full(teacher, test_loader, args, device, is_teacher=True, save_path=save_path)

    # =================================================================
    # PHASE 3: Student CMKD
    # =================================================================
    elif args.phase == 3:
        # Load frozen teacher
        teacher = build_teacher().to(device)
        if args.teacher_ckpt:
            ckpt_path = args.teacher_ckpt
        else:
            # Search for most recent Phase 2 (or Phase 1) checkpoint
            ckpt_path = find_latest_checkpoint(
                os.path.join(current_dir, args.save_dir), 'teacher_selfkd.pth')
        teacher.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded teacher from {ckpt_path}")

        # Create student
        student = RadarOnlyStudent(
            feature_size=args.feature_size, num_classes=args.num_classes,
            gru_hidden=args.gru_hidden, gru_layers=args.gru_layers_student,
            num_heads_mha=args.num_heads_mha,
            radar_h=args.radar_h, radar_w=args.radar_w,
            radar_channels=(args.student_radar_channels or args.radar_channels),
            preserve_angle=args.preserve_angle,
            two_stem=getattr(args, 'two_stem', False),
        ).to(device)
        s_total, s_train = count_params(student)
        t_total, _ = count_params(teacher)
        print(f"Student params: {s_total:,}  (compression: {t_total/s_total:.1f}x)")

        _, history = train_student_cmkd(teacher, student, dataloaders, args, device, save_path)

        print("\n=== Student Evaluation (CMKD) ===")
        student.load_state_dict(torch.load(os.path.join(save_path, 'student_cmkd_best.pth'), map_location=device))
        evaluate_full(student, test_loader, args, device, is_teacher=False, save_path=save_path)

        print("\n=== Teacher Reference ===")
        evaluate_full(teacher, test_loader, args, device, is_teacher=True)

    print(f"\nAll results saved to: {save_path}")


if __name__ == '__main__':
    main()
