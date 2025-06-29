# DDIM扩散模型项目

基于diffusers库实现的DDIM扩散算法，支持无条件生成和条件生成（图控+文控），生成分辨率为(b, 256, 16, 48)的特征。

## 功能特性

### 🎯 核心功能
- **无条件生成**: 生成随机特征
- **图控生成**: 基于输入图像(3, 256, 256)生成特征
- **文控生成**: 基于文本描述生成特征
- **混合条件**: 同时使用图像和文本条件

### 🚀 高级特性
- **CFG支持**: Classifier-Free Guidance，提高生成质量
- **多种微调**: 全参数微调、LoRA微调、Adapter微调
- **高效训练**: DDPM训练策略，DDIM推理策略
- **可视化**: 特征可视化和对比分析

## 项目结构

```
diffusion/
├── ddim_model.py              # 基础DDIM模型
├── conditional_diffusion.py   # 条件扩散模型（图控+文控）
├── lora_adapter.py           # LoRA和Adapter实现
├── data_generator.py         # 数据生成器
├── train.py                  # 无条件训练脚本
├── conditional_train.py      # 条件训练脚本
├── finetune.py              # 微调脚本
├── inference.py             # 无条件推理脚本
├── conditional_inference.py # 条件推理脚本
├── example_usage.py         # 基础使用示例
├── conditional_example.py   # 条件模型使用示例
├── requirements.txt         # 依赖包
└── README.md               # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 无条件生成

```python
from ddim_model import DDIMDiffusionModel

# 创建模型
model = DDIMDiffusionModel(device="cuda")

# 训练
model.train_step(clean_features, optimizer)

# 推理
samples = model.sample(batch_size=4, num_inference_steps=50)
```

### 2. 图控生成

```python
from conditional_diffusion import ConditionalDDIMDiffusionModel

# 创建条件模型
model = ConditionalDDIMDiffusionModel(device="cuda")

# 图像条件 (3, 256, 256)
image_condition = torch.randn(1, 3, 256, 256)

# 图控生成
samples = model.sample(
    batch_size=2,
    image_condition=image_condition,
    cfg_scale=7.5
)
```

### 3. 文控生成

```python
# 文本条件
text_condition = ["A beautiful landscape with mountains"]

# 文控生成
samples = model.sample(
    batch_size=2,
    text_condition=text_condition,
    cfg_scale=7.5
)
```

### 4. 混合条件生成

```python
# 同时使用图像和文本条件
samples = model.sample(
    batch_size=2,
    image_condition=image_condition,
    text_condition=text_condition,
    cfg_scale=7.5
)
```

## 训练指南

### 无条件训练

```bash
python train.py --num_epochs 100 --batch_size 8 --learning_rate 1e-4
```

### 条件训练

```bash
# 从头训练
python conditional_train.py --mode train --num_epochs 50 --batch_size 4

# 使用真实数据
python conditional_train.py --mode train \
    --image_dir ./images \
    --text_file ./texts.txt \
    --num_epochs 50
```

### 微调

```bash
# LoRA微调
python conditional_train.py --mode finetune \
    --base_model_path ./models/conditional_model.pth \
    --finetune_type lora \
    --num_epochs 10

# 全参数微调
python conditional_train.py --mode finetune \
    --base_model_path ./models/conditional_model.pth \
    --finetune_type full \
    --num_epochs 5
```

## 推理指南

### 无条件推理

```bash
python inference.py --model_path ./models/ddim_model.pth --num_samples 10
```

### 条件推理

```bash
# 图控推理
python conditional_inference.py \
    --model_path ./models/conditional_model.pth \
    --image_condition ./input_image.jpg \
    --num_samples 5 \
    --cfg_scale 7.5

# 文控推理
python conditional_inference.py \
    --model_path ./models/conditional_model.pth \
    --text_condition "A futuristic city with flying cars" \
    --num_samples 5 \
    --cfg_scale 7.5

# 混合条件推理
python conditional_inference.py \
    --model_path ./models/conditional_model.pth \
    --image_condition ./input_image.jpg \
    --text_condition "Transform into cyberpunk style" \
    --num_samples 5 \
    --cfg_scale 7.5
```

## 示例代码

### 运行完整示例

```bash
# 基础示例
python example_usage.py

# 条件模型示例
python conditional_example.py
```

### 代码示例

```python
# 1. 基础训练和推理
from example_usage import basic_training_and_inference
basic_training_and_inference()

# 2. 条件模型完整流程
from conditional_example import main
main()
```

## 模型架构

### 条件UNet
- **输入**: 特征 (256, 16, 48) + 条件特征 (256, 16, 48)
- **条件编码器**: 
  - 图像编码器: CLIP Vision + 投影层
  - 文本编码器: CLIP Text + 投影层
- **UNet**: 标准UNet架构，支持注意力机制
- **输出**: 预测噪声 (256, 16, 48)

### 训练策略
- **训练**: 使用DDPM调度器添加噪声，更稳定
- **推理**: 使用DDIM调度器，更快更可控
- **CFG**: Classifier-Free Guidance，提高生成质量

## 微调技术

### LoRA (Low-Rank Adaptation)
- 只训练低秩矩阵，参数效率高
- 保持预训练知识，适应新任务
- 适合快速微调和个性化

### Adapter
- 在模型中插入适配器层
- 冻结原始参数，只训练适配器
- 模块化设计，易于扩展

## 参数说明

### 训练参数
- `num_epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `save_interval`: 保存间隔

### 推理参数
- `num_inference_steps`: 推理步数 (20-100)
- `cfg_scale`: CFG强度 (0-20)
- `eta`: DDIM eta参数 (0-1)

### 模型参数
- `condition_dim`: 条件特征维度 (默认256)
- `rank`: LoRA秩 (默认16)
- `alpha`: LoRA缩放因子 (默认32)

## 性能优化

### 训练优化
- 使用混合精度训练
- 梯度累积
- 学习率调度
- 早停机制

### 推理优化
- 批处理推理
- 模型量化
- 缓存优化
- 并行采样

## 常见问题

### Q: 为什么训练用DDPM，推理用DDIM？
A: DDPM训练更稳定，DDIM推理更快且可控。这是标准做法。

### Q: CFG scale如何选择？
A: 通常7.5-15.0效果较好。值越大，条件控制越强，但可能降低多样性。

### Q: 如何提高生成质量？
A: 增加训练数据、调整CFG scale、使用更多推理步数、微调模型。

### Q: LoRA和全参数微调如何选择？
A: LoRA适合快速适应和资源受限场景，全参数微调适合充分训练。

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 更新日志

### v2.0.0 (2024-01-XX)
- ✨ 新增条件扩散模型（图控+文控）
- ✨ 支持CFG (Classifier-Free Guidance)
- ✨ 新增LoRA和Adapter微调
- ✨ 完整的训练和推理流程
- ✨ 可视化工具和示例代码

### v1.0.0 (2024-01-XX)
- 🎉 基础DDIM扩散模型
- 🎉 无条件生成功能
- 🎉 训练和推理脚本 