#!/bin/bash
# Logit + feature distillation into the reference student through a two-layer MLP projector (_v3rd_featmlp_s{42,1,2};
# supplementary S10); identical to _v3rd_feat_s* except --feat_projector mlp. Usage: bash scripts/run_e2.sh
cd "$(dirname "$0")/.." || exit 1
for S in 42 1 2; do
  echo "######## _v3rd_featmlp_s$S  $(date +%m-%d\ %H:%M) ########"
  ${PYTHON:-python} -u src/train_cmkd.py \
    --phase 3 --data_root dataset/scenario9 \
    --radar_channels 2 --second_map RD \
    --feat_projector mlp \
    --lambda1 1.0 --lambda2 1.0 --lambda3 0.0 --temperature 3.0 \
    --teacher_type cross_attn \
    --teacher_ckpt "$(ls cmkd_results/_v3rd_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)" \
    --epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 \
    --eta_min 1e-06 --grad_clip 10.0 \
    --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 \
    --gru_layers_student 1 --gru_layers_teacher 2 \
    --num_heads_mha 8 --num_heads_ca 4 --num_classes 64 --num_workers 16 \
    --seed $S --save_dir cmkd_results/_v3rd_featmlp_s$S
done
echo E2_DONE
