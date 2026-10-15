#!/bin/bash
# Seeds 3 to 9 of the MLP-projector students (_v3rd_featmlp_s*; see scripts/run_e2.sh). Seed 9 did not finish and is not
# reported. Usage: bash scripts/run_e2fill.sh
cd "$(dirname "$0")/.." || exit 1
run () {  # $1 tag, $2 seed, $3.. extra flags
  local TAG=$1 S=$2; shift 2
  echo "######## ${TAG}_s${S}  $(date +%m-%d\ %H:%M) ########"
  ${PYTHON:-python} -u src/train_cmkd.py \
    --phase 3 --data_root dataset/scenario9 \
    --radar_channels 2 --second_map RD \
    --lambda1 1.0 --temperature 3.0 \
    --teacher_type cross_attn \
    --teacher_ckpt "$(ls cmkd_results/_v3rd_teacher/phase2_*/teacher_selfkd.pth 2>/dev/null | tail -1)" \
    --epochs 70 --patience 20 --lr 0.0001 --weight_decay 0.0001 --batch_size 32 \
    --eta_min 1e-06 --grad_clip 10.0 \
    --seq_len 8 --num_pred 6 --feature_size 64 --gru_hidden 64 \
    --gru_layers_student 1 --gru_layers_teacher 2 \
    --num_heads_mha 8 --num_heads_ca 4 --num_classes 64 --num_workers 16 \
    "$@" --seed $S --save_dir cmkd_results/${TAG}_s${S}
}

# MLP projector, seeds 3 to 9 (seeds 42, 1 and 2: scripts/run_e2.sh)
for S in 3 4 5 6 7 8 9; do run _v3rd_featmlp $S --lambda2 1.0 --lambda3 0.0 --feat_projector mlp; done
echo FEATMLP_FILL_DONE
