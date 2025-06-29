import torch
import argparse
import os
import numpy as np
from ddim_model import DDIMDiffusionModel
from data_generator import visualize_features


def generate_features(
    model_path,
    batch_size=4,
    num_inference_steps=50,
    eta=0.0,
    output_dir="./generated_features",
    device="cuda",
    model_name="model"
):
    """
    使用训练好的DDIM模型生成特征
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = DDIMDiffusionModel(device=device)
    model.load_model(model_path)
    model.model.eval()
    
    print(f"生成 {batch_size} 个样本...")
    print(f"推理步数: {num_inference_steps}")
    print(f"DDIM eta: {eta}")
    
    # 生成特征
    with torch.no_grad():
        generated_features = model.sample(
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            eta=eta
        )
    
    print(f"生成完成！特征形状: {generated_features.shape}")
    print(f"特征统计 - 均值: {generated_features.mean():.4f}, 标准差: {generated_features.std():.4f}")
    
    # 保存生成的特征
    features_path = os.path.join(output_dir, f"{model_name}_generated_features.pt")
    torch.save(generated_features, features_path)
    print(f"特征已保存到: {features_path}")
    
    # 可视化特征
    viz_path = os.path.join(output_dir, f"{model_name}_generated_features_visualization.png")
    visualize_features(generated_features, viz_path)
    print(f"可视化已保存到: {viz_path}")
    
    return generated_features


def compare_models(
    pretrained_model_path,
    finetuned_model_path,
    batch_size=4,
    num_inference_steps=50,
    eta=0.0,
    output_dir="./model_comparison",
    device="cuda"
):
    """
    比较预训练模型和微调模型的生成效果
    """
    print("=== 模型对比分析 ===\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成预训练模型的样本
    print("1. 生成预训练模型样本...")
    pretrained_samples = generate_features(
        pretrained_model_path,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        eta=eta,
        output_dir=output_dir,
        device=device,
        model_name="pretrained"
    )
    
    # 生成微调模型的样本
    print("\n2. 生成微调模型样本...")
    finetuned_samples = generate_features(
        finetuned_model_path,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        eta=eta,
        output_dir=output_dir,
        device=device,
        model_name="finetuned"
    )
    
    # 计算统计对比
    print("\n3. 统计对比分析...")
    
    # 基础统计
    pretrained_mean = pretrained_samples.mean().item()
    pretrained_std = pretrained_samples.std().item()
    finetuned_mean = finetuned_samples.mean().item()
    finetuned_std = finetuned_samples.std().item()
    
    print(f"预训练模型 - 均值: {pretrained_mean:.4f}, 标准差: {pretrained_std:.4f}")
    print(f"微调模型 - 均值: {finetuned_mean:.4f}, 标准差: {finetuned_std:.4f}")
    print(f"均值差异: {abs(finetuned_mean - pretrained_mean):.4f}")
    print(f"标准差差异: {abs(finetuned_std - pretrained_std):.4f}")
    
    # 计算相似度指标
    similarity = torch.cosine_similarity(
        pretrained_samples.flatten(1),
        finetuned_samples.flatten(1),
        dim=1
    ).mean().item()
    
    print(f"余弦相似度: {similarity:.4f}")
    
    # 保存对比报告
    report_path = os.path.join(output_dir, "model_comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("预训练模型 vs 微调模型对比报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("生成参数:\n")
        f.write(f"  批量大小: {batch_size}\n")
        f.write(f"  推理步数: {num_inference_steps}\n")
        f.write(f"  DDIM eta: {eta}\n\n")
        
        f.write("统计对比:\n")
        f.write(f"  预训练模型 - 均值: {pretrained_mean:.6f}, 标准差: {pretrained_std:.6f}\n")
        f.write(f"  微调模型 - 均值: {finetuned_mean:.6f}, 标准差: {finetuned_std:.6f}\n")
        f.write(f"  均值差异: {abs(finetuned_mean - pretrained_mean):.6f}\n")
        f.write(f"  标准差差异: {abs(finetuned_std - pretrained_std):.6f}\n")
        f.write(f"  余弦相似度: {similarity:.6f}\n\n")
        
        # 分析结论
        if similarity > 0.9:
            f.write("结论: 两个模型生成的特征非常相似，微调效果不明显\n")
        elif similarity > 0.7:
            f.write("结论: 两个模型生成的特征有一定相似性，微调产生了适度变化\n")
        else:
            f.write("结论: 两个模型生成的特征差异较大，微调产生了显著变化\n")
    
    print(f"对比报告已保存到: {report_path}")
    
    return {
        'pretrained_samples': pretrained_samples,
        'finetuned_samples': finetuned_samples,
        'similarity': similarity,
        'stats': {
            'pretrained': {'mean': pretrained_mean, 'std': pretrained_std},
            'finetuned': {'mean': finetuned_mean, 'std': finetuned_std}
        }
    }


def compare_with_training_data(
    generated_features,
    model_path,
    num_samples=100,
    device="cuda"
):
    """
    比较生成的特征与训练数据的分布
    """
    from data_generator import create_dataloader
    
    print("比较生成特征与训练数据分布...")
    
    # 加载一些训练数据
    dataloader = create_dataloader("realistic", batch_size=num_samples, num_samples=num_samples)
    training_features = next(iter(dataloader)).to(device)
    
    # 计算统计信息
    gen_mean = generated_features.mean().item()
    gen_std = generated_features.std().item()
    train_mean = training_features.mean().item()
    train_std = training_features.std().item()
    
    print(f"生成特征 - 均值: {gen_mean:.4f}, 标准差: {gen_std:.4f}")
    print(f"训练特征 - 均值: {train_mean:.4f}, 标准差: {train_std:.4f}")
    print(f"均值差异: {abs(gen_mean - train_mean):.4f}")
    print(f"标准差差异: {abs(gen_std - train_std):.4f}")


def analyze_finetuning_effectiveness(
    pretrained_model_path,
    finetuned_model_path,
    output_dir="./finetuning_analysis",
    device="cuda"
):
    """
    分析微调的有效性
    """
    print("=== 微调有效性分析 ===\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载两个模型
    pretrained_model = DDIMDiffusionModel(device=device)
    pretrained_model.load_model(pretrained_model_path)
    
    finetuned_model = DDIMDiffusionModel(device=device)
    finetuned_model.load_model(finetuned_model_path)
    
    # 分析参数变化
    pretrained_state = pretrained_model.model.state_dict()
    finetuned_state = finetuned_model.model.state_dict()
    
    parameter_changes = {}
    total_params = 0
    changed_params = 0
    
    for key in pretrained_state.keys():
        if key in finetuned_state:
            pretrained_param = pretrained_state[key]
            finetuned_param = finetuned_state[key]
            
            # 计算变化
            diff = torch.abs(finetuned_param - pretrained_param)
            relative_change = diff.mean() / (pretrained_param.abs().mean() + 1e-8)
            
            parameter_changes[key] = {
                'mean_change': diff.mean().item(),
                'max_change': diff.max().item(),
                'relative_change': relative_change.item(),
                'param_shape': list(pretrained_param.shape)
            }
            
            total_params += 1
            if relative_change > 0.001:  # 0.1%的变化阈值
                changed_params += 1
    
    # 生成分析报告
    report_path = os.path.join(output_dir, "finetuning_effectiveness_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("微调有效性分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总参数数量: {total_params}\n")
        f.write(f"发生变化的参数数量: {changed_params}\n")
        f.write(f"变化参数比例: {changed_params/total_params*100:.2f}%\n\n")
        
        # 按变化程度排序
        sorted_changes = sorted(parameter_changes.items(), key=lambda x: x[1]['relative_change'], reverse=True)
        
        f.write("参数变化排名（前20）:\n")
        f.write("-" * 80 + "\n")
        for i, (key, stats) in enumerate(sorted_changes[:20]):
            f.write(f"{i+1:2d}. {key:<40} | 相对变化: {stats['relative_change']:.4f} | 形状: {stats['param_shape']}\n")
        
        # 分析不同层的变化
        layer_changes = {}
        for key, stats in parameter_changes.items():
            layer_name = key.split('.')[0] if '.' in key else key
            if layer_name not in layer_changes:
                layer_changes[layer_name] = []
            layer_changes[layer_name].append(stats['relative_change'])
        
        f.write("\n\n各层平均变化:\n")
        f.write("-" * 40 + "\n")
        for layer_name, changes in layer_changes.items():
            avg_change = np.mean(changes)
            f.write(f"{layer_name:<20}: {avg_change:.4f}\n")
    
    print(f"微调有效性分析报告已保存到: {report_path}")
    
    return parameter_changes


def main():
    parser = argparse.ArgumentParser(description="使用DDIM模型生成特征")
    parser.add_argument("--model_path", type=str, help="模型文件路径")
    parser.add_argument("--pretrained_model", type=str, help="预训练模型路径（用于对比）")
    parser.add_argument("--finetuned_model", type=str, help="微调模型路径（用于对比）")
    parser.add_argument("--batch_size", type=int, default=4, help="生成批量大小")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta参数")
    parser.add_argument("--output_dir", type=str, default="./generated_features", help="输出目录")
    parser.add_argument("--device", type=str, default="auto", help="推理设备")
    parser.add_argument("--compare", action="store_true", help="是否与训练数据比较")
    parser.add_argument("--compare_models", action="store_true", help="是否比较预训练和微调模型")
    parser.add_argument("--analyze_finetuning", action="store_true", help="是否分析微调有效性")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 检查参数
    if args.compare_models and (not args.pretrained_model or not args.finetuned_model):
        print("错误: 比较模型时需要同时提供预训练模型和微调模型路径")
        return
    
    if args.analyze_finetuning and (not args.pretrained_model or not args.finetuned_model):
        print("错误: 分析微调有效性时需要同时提供预训练模型和微调模型路径")
        return
    
    # 执行相应的功能
    if args.compare_models:
        # 比较预训练和微调模型
        comparison_result = compare_models(
            args.pretrained_model,
            args.finetuned_model,
            args.batch_size,
            args.num_inference_steps,
            args.eta,
            args.output_dir,
            device
        )
        
        if args.analyze_finetuning:
            # 分析微调有效性
            parameter_changes = analyze_finetuning_effectiveness(
                args.pretrained_model,
                args.finetuned_model,
                args.output_dir,
                device
            )
    
    elif args.model_path:
        # 生成单个模型的样本
        if not os.path.exists(args.model_path):
            print(f"错误: 模型文件不存在: {args.model_path}")
            return
        
        generated_features = generate_features(
            model_path=args.model_path,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
            output_dir=args.output_dir,
            device=device
        )
        
        # 比较分布（如果请求）
        if args.compare:
            compare_with_training_data(generated_features, args.model_path, device=device)
    
    else:
        print("错误: 请提供模型路径或指定比较模式")
        return
    
    print("推理完成！")


if __name__ == "__main__":
    main() 