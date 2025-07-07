#!/bin/bash

# 文本控制扩散模型训练启动脚本
# 基于Cross-Attention的文本控制扩散模型训练

echo "🚀 开始文本控制扩散模型训练..."

# 数据路径配置（需要根据实际情况调整）
TRAIN_PATH="/home/lizihao/MMD3D_JS/data_split/shapenet/train.txt"
VAL_PATH="/home/lizihao/MMD3D_JS/data_split/shapenet/val.txt"
TRI_DIR="/home/lizihao/MMD3D_pr15_vae/exp-shapenet-all/05_15-18_55_55_finetune/triplane_256/model_geometry_opt-1st/train"
TEXT_JSON_PATH="/path/to/text_descriptions.json"  # 需要指定实际的文本描述文件

# 模型参数
IN_CHANNELS=256
OUT_CHANNELS=256
FEATURE_SIZE="16 48"
CROSS_ATTENTION_DIM=1024
CONTROL_STRENGTH=1.0
USE_CLIP=true
MAX_TEXT_LENGTH=77

# 训练参数
BATCH_SIZE=8
NUM_EPOCHS=100
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-6
NUM_WORKERS=4

# 输出配置
OUTPUT_DIR="./text_controlled_experiments"
EXP_NAME="text_controlled_$(date +%m%d_%H%M%S)"

# 其他参数
DEVICE="auto"
SEED=42
SAVE_FREQ=10
EARLY_STOPPING_PATIENCE=15
RANDOM_TEXT=true  # 推荐：随机选择文本描述以提高泛化能力

# 检查必要文件是否存在
echo "🔍 检查数据路径..."
if [ ! -f "$TRAIN_PATH" ]; then
    echo "❌ 训练数据列表文件不存在: $TRAIN_PATH"
    exit 1
fi

if [ ! -d "$TRI_DIR" ]; then
    echo "❌ 特征数据目录不存在: $TRI_DIR"
    exit 1
fi

if [ ! -f "$TEXT_JSON_PATH" ]; then
    echo "❌ 文本描述json文件不存在: $TEXT_JSON_PATH"
    echo "💡 请确保已准备好文本描述文件，格式为 {\"sample_id\": \"text_description\", ...}"
    exit 1
fi

echo "✅ 数据路径检查通过"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动训练
echo "🎯 启动文本控制扩散模型训练..."
echo "实验名称: $EXP_NAME"
echo "输出目录: $OUTPUT_DIR/$EXP_NAME"

python train_text_controlled.py \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --tri_dir "$TRI_DIR" \
    --text_json_path "$TEXT_JSON_PATH" \
    --in_channels $IN_CHANNELS \
    --out_channels $OUT_CHANNELS \
    --feature_size $FEATURE_SIZE \
    --cross_attention_dim $CROSS_ATTENTION_DIM \
    --control_strength $CONTROL_STRENGTH \
    --use_clip $USE_CLIP \
    --max_text_length $MAX_TEXT_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --num_workers $NUM_WORKERS \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --device "$DEVICE" \
    --seed $SEED \
    --save_freq $SAVE_FREQ \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --use_categories false \
    --feature_ext ".pt" \
    --random_text $RANDOM_TEXT

echo "🎉 训练完成！"
echo "实验结果保存在: $OUTPUT_DIR/$EXP_NAME"

# 显示训练结果目录结构
echo "📁 训练结果目录结构:"
ls -la "$OUTPUT_DIR/$EXP_NAME/" 2>/dev/null || echo "目录尚未创建或训练失败"

# 提示如何查看tensorboard
echo ""
echo "📊 查看训练日志:"
echo "tensorboard --logdir=$OUTPUT_DIR/$EXP_NAME/tensorboard"
echo ""
echo "📝 查看训练配置:"
echo "cat $OUTPUT_DIR/$EXP_NAME/config.json" 