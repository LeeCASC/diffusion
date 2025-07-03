#!/bin/bash

# 快速训练脚本 - ShapeNet Car 扩散模型

echo "🚀 开始训练 ShapeNet Car 扩散模型..."

# 训练命令
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config /home/lizihao/diffusion/config/exp-shapenet-car/specs_shapenet_all.json \
  --train_path /home/lizihao/MMD3D_JS/data_split/shapenet_car/train.txt \
  --tri_dir /home/lizihao/MMD3D_pr15_vae/exp-shapenet-all/05_15-18_55_55_finetune/triplane_256/model_geometry_opt-1st/train/02958343 \
  --eval_interval 10 \
  --eval_samples 16 \
  --patience 20 \
  --min_delta 1e-6

echo "✅ 训练完成！" 