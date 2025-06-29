#!/usr/bin/env python3
"""
扩散模型微调使用示例
演示完整的微调流程
"""

import torch
import os
from ddim_model import DDIMDiffusionModel
from data_generator import create_dataloader, visualize_features
from finetune import FineTuneDiffusionModel, finetune_model


def example_1_basic_finetuning():
    """示例1: 基础微调流程"""
    print("=== 示例1: 基础微调流程 ===\n")
    
    # 假设我们已经有一个预训练模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建预训练模型（如果没有，先训练一个）
    print("1. 创建预训练模型...")
    pretrained_model = DDIMDiffusionModel(device=device)
    
    # 保存预训练模型
    os.makedirs("pretrained_models", exist_ok=True)
    pretrained_path = "pretrained_models/pretrained_model.pth"
    pretrained_model.save_model(pretrained_path)
    print(f"预训练模型已保存到: {pretrained_path}")
    
    # 创建微调数据（1k个样本）
    print("\n2. 创建微调数据集...")
    finetune_dataloader = create_dataloader(
        dataset_type="realistic",
        batch_size=4,
        num_samples=1000  # 1k个样本
    )
    print(f"微调数据集大小: {len(finetune_dataloader)} 批次")
    
    # 开始微调
    print("\n3. 开始微调...")
    finetuned_model = finetune_model(
        pretrained_model_path=pretrained_path,
        finetune_dataloader=finetune_dataloader,
        num_epochs=5,  # 微调轮数较少
        learning_rate=1e-5,  # 较小的学习率
        save_dir="./finetuned_models",
        log_dir="./finetune_logs",
        device=device
    )
    
    print("基础微调完成！\n")


def example_2_finetuning_comparison():
    """示例2: 微调前后对比"""
    print("=== 示例2: 微调前后对比 ===\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 假设我们已经有了预训练和微调模型
    pretrained_path = "pretrained_models/pretrained_model.pth"
    finetuned_path = "finetuned_models/best_finetuned_model.pth"
    
    if not os.path.exists(pretrained_path) or not os.path.exists(finetuned_path):
        print("请先运行示例1生成预训练和微调模型")
        return
    
    print("1. 加载模型...")
    pretrained_model = DDIMDiffusionModel(device=device)
    pretrained_model.load_model(pretrained_path)
    
    finetuned_model = DDIMDiffusionModel(device=device)
    finetuned_model.load_model(finetuned_path)
    
    print("2. 生成对比样本...")
    # 生成预训练模型样本
    pretrained_model.model.eval()
    with torch.no_grad():
        pretrained_samples = pretrained_model.sample(batch_size=4, num_inference_steps=50)
    
    # 生成微调模型样本
    finetuned_model.model.eval()
    with torch.no_grad():
        finetuned_samples = finetuned_model.sample(batch_size=4, num_inference_steps=50)
    
    print("3. 保存对比可视化...")
    os.makedirs("comparison_results", exist_ok=True)
    visualize_features(pretrained_samples, "comparison_results/pretrained_samples.png")
    visualize_features(finetuned_samples, "comparison_results/finetuned_samples.png")
    
    print("4. 计算统计对比...")
    pretrained_mean = pretrained_samples.mean().item()
    pretrained_std = pretrained_samples.std().item()
    finetuned_mean = finetuned_samples.mean().item()
    finetuned_std = finetuned_samples.std().item()
    
    print(f"预训练模型 - 均值: {pretrained_mean:.4f}, 标准差: {pretrained_std:.4f}")
    print(f"微调模型 - 均值: {finetuned_mean:.4f}, 标准差: {finetuned_std:.4f}")
    print(f"均值差异: {abs(finetuned_mean - pretrained_mean):.4f}")
    print(f"标准差差异: {abs(finetuned_std - pretrained_std):.4f}")
    
    print("微调对比完成！\n")


def example_3_parameter_analysis():
    """示例3: 参数变化分析"""
    print("=== 示例3: 参数变化分析 ===\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 假设我们已经有了预训练和微调模型
    pretrained_path = "pretrained_models/pretrained_model.pth"
    finetuned_path = "finetuned_models/best_finetuned_model.pth"
    
    if not os.path.exists(pretrained_path) or not os.path.exists(finetuned_path):
        print("请先运行示例1生成预训练和微调模型")
        return
    
    print("1. 创建微调分析器...")
    finetune_analyzer = FineTuneDiffusionModel(pretrained_path, device=device)
    
    # 加载微调后的模型
    finetune_analyzer.model.load_model(finetuned_path)
    
    print("2. 分析参数变化...")
    changes = finetune_analyzer.get_parameter_changes()
    
    # 统计变化
    total_params = len(changes)
    significant_changes = sum(1 for stats in changes.values() if stats['relative_change'] > 0.01)
    
    print(f"总参数数量: {total_params}")
    print(f"显著变化参数数量 (>1%): {significant_changes}")
    print(f"显著变化比例: {significant_changes/total_params*100:.2f}%")
    
    # 找出变化最大的参数
    sorted_changes = sorted(changes.items(), key=lambda x: x[1]['relative_change'], reverse=True)
    print("\n变化最大的5个参数:")
    for i, (key, stats) in enumerate(sorted_changes[:5]):
        print(f"{i+1}. {key}: {stats['relative_change']:.4f}")
    
    print("参数分析完成！\n")


def example_4_custom_finetuning():
    """示例4: 自定义微调策略"""
    print("=== 示例4: 自定义微调策略 ===\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 假设我们已经有了预训练模型
    pretrained_path = "pretrained_models/pretrained_model.pth"
    
    if not os.path.exists(pretrained_path):
        print("请先运行示例1生成预训练模型")
        return
    
    print("1. 创建自定义微调数据...")
    # 创建特定领域的微调数据
    custom_dataloader = create_dataloader(
        dataset_type="synthetic",  # 使用合成数据模拟特定领域
        batch_size=2,  # 更小的批量大小
        num_samples=500  # 更少的样本
    )
    
    print("2. 自定义微调参数...")
    # 使用更保守的微调参数
    custom_finetuned = finetune_model(
        pretrained_model_path=pretrained_path,
        finetune_dataloader=custom_dataloader,
        num_epochs=3,  # 更少的轮数
        learning_rate=5e-6,  # 更小的学习率
        save_dir="./custom_finetuned_models",
        log_dir="./custom_finetune_logs",
        device=device
    )
    
    print("3. 测试自定义微调效果...")
    custom_finetuned.model.model.eval()
    with torch.no_grad():
        custom_samples = custom_finetuned.model.sample(batch_size=4, num_inference_steps=50)
    
    os.makedirs("custom_results", exist_ok=True)
    visualize_features(custom_samples, "custom_results/custom_finetuned_samples.png")
    
    print("自定义微调完成！\n")


def example_5_inference_with_finetuned_model():
    """示例5: 使用微调模型进行推理"""
    print("=== 示例5: 使用微调模型进行推理 ===\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 假设我们已经有了微调模型
    finetuned_path = "finetuned_models/best_finetuned_model.pth"
    
    if not os.path.exists(finetuned_path):
        print("请先运行示例1生成微调模型")
        return
    
    print("1. 加载微调模型...")
    model = DDIMDiffusionModel(device=device)
    model.load_model(finetuned_path)
    model.model.eval()
    
    print("2. 使用不同参数生成样本...")
    
    # 不同推理步数
    inference_steps_list = [20, 50, 100]
    for steps in inference_steps_list:
        print(f"  推理步数: {steps}")
        with torch.no_grad():
            samples = model.sample(batch_size=2, num_inference_steps=steps)
            print(f"    样本统计 - 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")
    
    # 不同eta值
    eta_values = [0.0, 0.5, 1.0]
    for eta in eta_values:
        print(f"  DDIM eta: {eta}")
        with torch.no_grad():
            samples = model.sample(batch_size=2, num_inference_steps=50, eta=eta)
            print(f"    样本统计 - 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")
    
    print("3. 保存推理结果...")
    os.makedirs("inference_results", exist_ok=True)
    
    with torch.no_grad():
        final_samples = model.sample(batch_size=4, num_inference_steps=50, eta=0.0)
        visualize_features(final_samples, "inference_results/finetuned_inference_samples.png")
    
    print("微调模型推理完成！\n")


def main():
    """运行所有微调示例"""
    print("扩散模型微调使用示例")
    print("=" * 50)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU进行计算")
    
    print()
    
    try:
        # 运行示例
        example_1_basic_finetuning()
        example_2_finetuning_comparison()
        example_3_parameter_analysis()
        example_4_custom_finetuning()
        example_5_inference_with_finetuned_model()
        
        print("所有微调示例运行完成！")
        print("生成的文件保存在相应的目录中")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 