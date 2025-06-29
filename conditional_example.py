#!/usr/bin/env python3
"""
条件扩散模型使用示例
支持图控生成和文控生成，包含训练、微调和推理
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# 导入我们的模块
from conditional_diffusion import ConditionalDDIMDiffusionModel, create_conditional_dataset
from conditional_train import train_conditional_model, finetune_conditional_model
from conditional_inference import conditional_inference, visualize_features


def example_1_basic_training():
    """示例1: 基础条件训练"""
    print("=" * 50)
    print("示例1: 基础条件训练")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("创建条件数据集...")
    dataset = create_conditional_dataset(
        image_dir=None,  # 使用随机图像
        text_file=None,  # 使用随机文本
        num_samples=1000  # 小数据集用于演示
    )
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建模型
    print("创建条件扩散模型...")
    model = ConditionalDDIMDiffusionModel(device=device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    
    # 训练（只训练几个epoch用于演示）
    print("开始训练...")
    losses = train_conditional_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=2,  # 只训练2个epoch
        device=device,
        save_dir="./example_models",
        save_interval=100,
        log_interval=50
    )
    
    print(f"训练完成，最终损失: {losses[-1]:.4f}")
    
    # 保存模型
    model.save_model("./example_models/conditional_model_final.pth")
    print("模型已保存到: ./example_models/conditional_model_final.pth")


def example_2_image_conditioned_generation():
    """示例2: 图控生成"""
    print("\n" + "=" * 50)
    print("示例2: 图控生成")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检查是否有训练好的模型
    model_path = "./example_models/conditional_model_final.pth"
    if not os.path.exists(model_path):
        print("未找到训练好的模型，请先运行示例1")
        return
    
    # 加载模型
    print("加载条件模型...")
    model = ConditionalDDIMDiffusionModel(device=device)
    model.load_model(model_path)
    
    # 创建随机图像条件
    print("创建图像条件...")
    image_condition = torch.randn(1, 3, 256, 256).to(device)
    
    # 生成样本
    print("开始图控生成...")
    samples = model.sample(
        batch_size=2,
        num_inference_steps=20,  # 快速推理
        image_condition=image_condition,
        cfg_scale=7.5
    )
    
    print(f"生成样本形状: {samples.shape}")
    
    # 可视化结果
    os.makedirs("./example_outputs", exist_ok=True)
    for i in range(samples.shape[0]):
        viz_path = f"./example_outputs/image_conditioned_sample_{i}.png"
        visualize_features(samples[i], viz_path, f"Image Conditioned Sample {i}")
    
    print("图控生成完成，结果保存在 ./example_outputs/")


def example_3_text_conditioned_generation():
    """示例3: 文控生成"""
    print("\n" + "=" * 50)
    print("示例3: 文控生成")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检查是否有训练好的模型
    model_path = "./example_models/conditional_model_final.pth"
    if not os.path.exists(model_path):
        print("未找到训练好的模型，请先运行示例1")
        return
    
    # 加载模型
    print("加载条件模型...")
    model = ConditionalDDIMDiffusionModel(device=device)
    model.load_model(model_path)
    
    # 文本条件
    text_conditions = [
        "A beautiful landscape with mountains and lakes",
        "A modern city skyline at sunset",
        "A peaceful forest scene with sunlight filtering through trees"
    ]
    
    print("开始文控生成...")
    for i, text in enumerate(text_conditions):
        print(f"生成文本: {text}")
        
        samples = model.sample(
            batch_size=1,
            num_inference_steps=20,
            text_condition=[text],
            cfg_scale=7.5
        )
        
        # 可视化结果
        os.makedirs("./example_outputs", exist_ok=True)
        viz_path = f"./example_outputs/text_conditioned_sample_{i}.png"
        visualize_features(samples[0], viz_path, f"Text: {text[:30]}...")
    
    print("文控生成完成，结果保存在 ./example_outputs/")


def example_4_cfg_comparison():
    """示例4: CFG效果对比"""
    print("\n" + "=" * 50)
    print("示例4: CFG效果对比")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检查是否有训练好的模型
    model_path = "./example_models/conditional_model_final.pth"
    if not os.path.exists(model_path):
        print("未找到训练好的模型，请先运行示例1")
        return
    
    # 加载模型
    print("加载条件模型...")
    model = ConditionalDDIMDiffusionModel(device=device)
    model.load_model(model_path)
    
    # 相同的条件
    image_condition = torch.randn(1, 3, 256, 256).to(device)
    text_condition = ["A futuristic city with flying cars"]
    
    # 不同的CFG scale
    cfg_scales = [0.0, 3.0, 7.5, 15.0]
    
    print("对比不同CFG scale的效果...")
    all_samples = []
    
    for cfg_scale in cfg_scales:
        print(f"CFG scale: {cfg_scale}")
        
        samples = model.sample(
            batch_size=1,
            num_inference_steps=20,
            image_condition=image_condition,
            text_condition=text_condition,
            cfg_scale=cfg_scale
        )
        
        all_samples.append(samples[0])
    
    # 可视化对比
    fig, axes = plt.subplots(1, len(cfg_scales), figsize=(4*len(cfg_scales), 4))
    
    for i, (samples, cfg_scale) in enumerate(zip(all_samples, cfg_scales)):
        # 显示均值特征
        mean_features = samples.mean(dim=0)
        im = axes[i].imshow(mean_features.cpu().numpy(), cmap='viridis')
        axes[i].set_title(f'CFG={cfg_scale}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle('CFG Scale Comparison')
    plt.tight_layout()
    
    os.makedirs("./example_outputs", exist_ok=True)
    plt.savefig("./example_outputs/cfg_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("CFG对比完成，结果保存在 ./example_outputs/cfg_comparison.png")


def example_5_lora_finetuning():
    """示例5: LoRA微调"""
    print("\n" + "=" * 50)
    print("示例5: LoRA微调")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检查是否有基础模型
    base_model_path = "./example_models/conditional_model_final.pth"
    if not os.path.exists(base_model_path):
        print("未找到基础模型，请先运行示例1")
        return
    
    # 创建微调数据集
    print("创建微调数据集...")
    finetune_dataset = create_conditional_dataset(
        image_dir=None,
        text_file=None,
        num_samples=500  # 小数据集用于微调
    )
    
    from torch.utils.data import DataLoader
    finetune_loader = DataLoader(finetune_dataset, batch_size=4, shuffle=True)
    
    # LoRA微调
    print("开始LoRA微调...")
    from conditional_diffusion import ConditionalLoRAModel
    
    lora_model = ConditionalLoRAModel(base_model_path, device=device)
    optimizer = torch.optim.Adam(lora_model.get_lora_parameters(), lr=1e-5)
    
    losses = finetune_conditional_model(
        base_model_path=base_model_path,
        train_loader=finetune_loader,
        optimizer=optimizer,
        num_epochs=1,  # 只微调1个epoch
        device=device,
        save_dir="./example_models",
        finetune_type="lora"
    )
    
    # 保存LoRA权重
    lora_model.save_lora_weights("./example_models/conditional_lora_final.pth")
    print("LoRA微调完成，权重已保存")


def example_6_comprehensive_inference():
    """示例6: 综合推理示例"""
    print("\n" + "=" * 50)
    print("示例6: 综合推理示例")
    print("=" * 50)
    
    # 使用推理脚本进行综合推理
    model_path = "./example_models/conditional_model_final.pth"
    if not os.path.exists(model_path):
        print("未找到模型，请先运行示例1")
        return
    
    print("执行综合推理...")
    
    # 图控推理
    print("1. 图控推理...")
    conditional_inference(
        model_path=model_path,
        output_dir="./example_outputs/image_conditioned",
        image_condition=torch.randn(1, 3, 256, 256),  # 随机图像
        num_samples=3,
        num_inference_steps=20,
        cfg_scale=7.5
    )
    
    # 文控推理
    print("2. 文控推理...")
    conditional_inference(
        model_path=model_path,
        output_dir="./example_outputs/text_conditioned",
        text_condition="A magical forest with glowing mushrooms and fairy lights",
        num_samples=3,
        num_inference_steps=20,
        cfg_scale=7.5
    )
    
    # 混合条件推理
    print("3. 混合条件推理...")
    conditional_inference(
        model_path=model_path,
        output_dir="./example_outputs/mixed_conditioned",
        image_condition=torch.randn(1, 3, 256, 256),
        text_condition="Transform this image into a cyberpunk style",
        num_samples=3,
        num_inference_steps=20,
        cfg_scale=7.5
    )
    
    print("综合推理完成！")


def main():
    """主函数，运行所有示例"""
    print("条件扩散模型完整示例")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs("./example_models", exist_ok=True)
    os.makedirs("./example_outputs", exist_ok=True)
    
    try:
        # 运行示例
        example_1_basic_training()
        example_2_image_conditioned_generation()
        example_3_text_conditioned_generation()
        example_4_cfg_comparison()
        example_5_lora_finetuning()
        example_6_comprehensive_inference()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        print("生成的文件:")
        print("- 模型文件: ./example_models/")
        print("- 输出结果: ./example_outputs/")
        print("- 可视化图像: ./example_outputs/*.png")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 