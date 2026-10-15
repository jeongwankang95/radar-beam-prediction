#!/bin/bash
# Students distilled from the concatenation range-Doppler teacher _cc_teacher_rd, seeds 3 to 9 (_cckd_s*; the teacher and
# seeds 42, 1 and 2 come from scripts/run_all.sh), then any MLP-projector student of seeds 4 to 9 not yet trained
# (_v3rd_featmlp_s*; see scripts/run_e2.sh). Usage: bash scripts/run_cckd.sh
cd "$(dirname "$0")/.." || exit 1
for S in 3 4 5 6 7 8 9; do
  echo "######## _cckd_s$S  $(date +%m-%d\ %H:%M) ########"
  ${PYTHON:-python} -u src/train_cmkd.py \
    --phase 3 --data_root dataset/scenario9 \
    --radar_channels 2 --second_map RD \
    --lambda1 1.0 --lambda2 0.0 --lambda3 0.0 --temperature 3.0 \
    --teacher_type concat \
    --teacher_ckpt "$(ls cmkd_results/_cc_teacher_rd/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)" \
    --epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 \
    --eta_min 1e-06 --grad_clip 10.0 \
    --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 \
    --gru_layers_student 1 --gru_layers_teacher 2 \
    --num_heads_mha 8 --num_heads_ca 4 --num_classes 64 --num_workers 16 \
    --seed $S --save_dir cmkd_results/_cckd_s$S
done
echo CCKD_DONE

# ---- MLP-projector students not yet trained (see scripts/run_e2.sh)
for S in 4 5 6 7 8 9; do
  [ -d cmkd_results/_v3rd_featmlp_s$S ] && continue
  echo "######## _v3rd_featmlp_s$S  $(date +%m-%d\ %H:%M) ########"
  ${PYTHON:-python} -u src/train_cmkd.py \
    --phase 3 --data_root dataset/scenario9 \
    --radar_channels 2 --second_map RD \
    --lambda1 1.0 --lambda2 1.0 --lambda3 0.0 --temperature 3.0 \
    --feat_projector mlp \
    --teacher_type cross_attn \
    --teacher_ckpt "$(ls cmkd_results/_v3rd_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)" \
    --epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 \
    --eta_min 1e-06 --grad_clip 10.0 \
    --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 \
    --gru_layers_student 1 --gru_layers_teacher 2 \
    --num_heads_mha 8 --num_heads_ca 4 --num_classes 64 --num_workers 16 \
    --seed $S --save_dir cmkd_results/_v3rd_featmlp_s$S
done
echo FEATMLP_RESUME_DONE
