#!/usr/bin/env python3
"""
DDIM扩散模型使用示例
演示如何使用项目的主要功能
"""

import torch
import os
from ddim_model import DDIMDiffusionModel
from data_generator import create_dataloader, visualize_features


def example_1_basic_training():
    """示例1: 基础训练"""
    print("=== 示例1: 基础训练 ===")
    
    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DDIMDiffusionModel(device=device)
    
    # 创建数据加载器（使用小数据集进行快速演示）
    dataloader = create_dataloader("synthetic", batch_size=4, num_samples=100)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    
    print(f"开始训练，设备: {device}")
    print(f"数据加载器长度: {len(dataloader)}")
    
    # 训练几个epoch
    model.model.train()
    for epoch in range(3):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, features in enumerate(dataloader):
            features = features.to(device)
            loss = model.train_step(features, optimizer)
            epoch_loss += loss
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.6f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.6f}")
    
    print("基础训练完成！\n")


def example_2_feature_generation():
    """示例2: 特征生成"""
    print("=== 示例2: 特征生成 ===")
    
    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DDIMDiffusionModel(device=device)
    
    print(f"使用设备: {device}")
    print("生成特征样本...")
    
    # 生成特征
    model.model.eval()
    with torch.no_grad():
        samples = model.sample(batch_size=2, num_inference_steps=20)
    
    print(f"生成完成！特征形状: {samples.shape}")
    print(f"特征统计 - 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")
    
    # 可视化特征
    os.makedirs("examples", exist_ok=True)
    visualize_features(samples, "examples/generated_features.png")
    print("特征可视化已保存到: examples/generated_features.png\n")


def example_3_data_exploration():
    """示例3: 数据探索"""
    print("=== 示例3: 数据探索 ===")
    
    # 创建不同类型的数据加载器
    synthetic_dataloader = create_dataloader("synthetic", batch_size=4, num_samples=100)
    realistic_dataloader = create_dataloader("realistic", batch_size=4, num_samples=100)
    
    print("探索合成数据...")
    synthetic_batch = next(iter(synthetic_dataloader))
    print(f"合成数据形状: {synthetic_batch.shape}")
    print(f"合成数据统计 - 均值: {synthetic_batch.mean():.4f}, 标准差: {synthetic_batch.std():.4f}")
    
    print("探索现实数据...")
    realistic_batch = next(iter(realistic_dataloader))
    print(f"现实数据形状: {realistic_batch.shape}")
    print(f"现实数据统计 - 均值: {realistic_batch.mean():.4f}, 标准差: {realistic_batch.std():.4f}")
    
    # 可视化数据
    os.makedirs("examples", exist_ok=True)
    visualize_features(synthetic_batch, "examples/synthetic_data.png")
    visualize_features(realistic_batch, "examples/realistic_data.png")
    print("数据可视化已保存到 examples/ 目录\n")


def example_4_model_save_load():
    """示例4: 模型保存和加载"""
    print("=== 示例4: 模型保存和加载 ===")
    
    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DDIMDiffusionModel(device=device)
    
    # 保存模型
    os.makedirs("examples", exist_ok=True)
    save_path = "examples/test_model.pth"
    model.save_model(save_path)
    print(f"模型已保存到: {save_path}")
    
    # 创建新模型并加载
    new_model = DDIMDiffusionModel(device=device)
    new_model.load_model(save_path)
    print("模型加载成功！")
    
    # 验证模型是否相同
    model.model.eval()
    new_model.model.eval()
    
    with torch.no_grad():
        test_input = torch.randn(1, 256, 16, 48).to(device)
        test_timesteps = torch.tensor([500]).to(device)
        
        output1 = model.model(test_input, test_timesteps).sample
        output2 = new_model.model(test_input, test_timesteps).sample
        
        diff = torch.abs(output1 - output2).max().item()
        print(f"模型输出差异: {diff:.8f} (应该接近0)")
    
    print("模型保存和加载测试完成！\n")


def example_5_ddim_parameters():
    """示例5: DDIM参数探索"""
    print("=== 示例5: DDIM参数探索 ===")
    
    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DDIMDiffusionModel(device=device)
    
    # 测试不同的推理步数
    inference_steps_list = [10, 20, 50, 100]
    
    for steps in inference_steps_list:
        print(f"测试推理步数: {steps}")
        model.model.eval()
        
        with torch.no_grad():
            samples = model.sample(batch_size=1, num_inference_steps=steps)
            print(f"  生成样本统计 - 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")
    
    # 测试不同的eta值
    eta_values = [0.0, 0.5, 1.0]
    
    for eta in eta_values:
        print(f"测试eta值: {eta}")
        model.model.eval()
        
        with torch.no_grad():
            samples = model.sample(batch_size=1, num_inference_steps=20, eta=eta)
            print(f"  生成样本统计 - 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")
    
    print("DDIM参数探索完成！\n")


def main():
    """运行所有示例"""
    print("DDIM扩散模型使用示例")
    print("=" * 50)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU进行计算")
    
    print()
    
    try:
        # 运行示例
        example_1_basic_training()
        example_2_feature_generation()
        example_3_data_exploration()
        example_4_model_save_load()
        example_5_ddim_parameters()
        
        print("所有示例运行完成！")
        print("生成的文件保存在 examples/ 目录中")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 