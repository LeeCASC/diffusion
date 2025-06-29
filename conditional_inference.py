import torch
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from conditional_diffusion import ConditionalDDIMDiffusionModel, ConditionalLoRAModel


def load_image(image_path, target_size=(256, 256)):
    """加载和预处理图像"""
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return None
    
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)  # 添加batch维度
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None


def save_features(features, save_path, format='npy'):
    """保存生成的特征"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format == 'npy':
        np.save(save_path, features.cpu().numpy())
    elif format == 'pt':
        torch.save(features, save_path)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    print(f"特征已保存到: {save_path}")


def visualize_features(features, save_path=None, title="Generated Features"):
    """可视化特征"""
    # 将特征转换为可可视化的格式
    if features.dim() == 4:  # (batch, channels, height, width)
        features = features[0]  # 取第一个样本
    
    # 计算每个通道的统计信息
    mean_features = features.mean(dim=0)  # (height, width)
    std_features = features.std(dim=0)    # (height, width)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示均值
    im1 = axes[0].imshow(mean_features.cpu().numpy(), cmap='viridis')
    axes[0].set_title('Mean Features')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # 显示标准差
    im2 = axes[1].imshow(std_features.cpu().numpy(), cmap='viridis')
    axes[1].set_title('Std Features')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # 显示第一个通道
    im3 = axes[2].imshow(features[0].cpu().numpy(), cmap='viridis')
    axes[2].set_title('First Channel')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def conditional_inference(
    model_path,
    output_dir,
    image_condition=None,
    text_condition=None,
    batch_size=1,
    num_inference_steps=50,
    eta=0.0,
    cfg_scale=7.5,
    num_samples=1,
    device="cuda",
    save_format='npy',
    visualize=True
):
    """条件推理"""
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    
    # 检查是否是LoRA模型
    if 'lora' in model_path.lower():
        model = ConditionalLoRAModel(model_path, device=str(device))
        print("加载LoRA模型")
    else:
        model = ConditionalDDIMDiffusionModel(device=str(device))
        model.load_model(model_path)
        print("加载标准条件模型")
    
    # 准备条件
    if image_condition is not None:
        if isinstance(image_condition, str):
            image_condition = load_image(image_condition)
            if image_condition is None:
                return
        image_condition = image_condition.to(device)
        print(f"图像条件形状: {image_condition.shape}")
    
    if text_condition is not None:
        if isinstance(text_condition, str):
            text_condition = [text_condition]
        print(f"文本条件: {text_condition}")
    
    # 生成样本
    print(f"开始生成 {num_samples} 个样本...")
    print(f"推理参数: steps={num_inference_steps}, eta={eta}, cfg_scale={cfg_scale}")
    
    all_samples = []
    
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        
        # 调整条件批次大小
        current_image_condition = None
        current_text_condition = None
        
        if image_condition is not None:
            current_image_condition = image_condition.repeat(current_batch_size, 1, 1, 1)
        
        if text_condition is not None:
            current_text_condition = text_condition * current_batch_size
        
        # 采样
        samples = model.sample(
            batch_size=current_batch_size,
            num_inference_steps=num_inference_steps,
            eta=eta,
            image_condition=current_image_condition,
            text_condition=current_text_condition,
            cfg_scale=cfg_scale
        )
        
        all_samples.append(samples)
        print(f"已生成 {i + current_batch_size}/{num_samples} 个样本")
    
    # 合并所有样本
    all_samples = torch.cat(all_samples, dim=0)
    print(f"生成完成，总样本形状: {all_samples.shape}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有样本
    all_samples_path = os.path.join(output_dir, f'all_samples.{save_format}')
    save_features(all_samples, all_samples_path, save_format)
    
    # 保存单个样本
    for i in range(min(5, num_samples)):  # 最多保存5个样本
        sample_path = os.path.join(output_dir, f'sample_{i}.{save_format}')
        save_features(all_samples[i:i+1], sample_path, save_format)
        
        if visualize:
            viz_path = os.path.join(output_dir, f'sample_{i}_visualization.png')
            visualize_features(all_samples[i], viz_path, f"Sample {i}")
    
    # 保存生成配置
    config = {
        'model_path': model_path,
        'num_samples': num_samples,
        'batch_size': batch_size,
        'num_inference_steps': num_inference_steps,
        'eta': eta,
        'cfg_scale': cfg_scale,
        'device': str(device),
        'image_condition': image_condition.shape if image_condition is not None else None,
        'text_condition': text_condition,
        'output_shape': all_samples.shape
    }
    
    config_path = os.path.join(output_dir, 'generation_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"生成配置已保存到: {config_path}")
    print(f"所有结果已保存到: {output_dir}")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='条件扩散模型推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--output_dir', type=str, default='./conditional_outputs', help='输出目录')
    parser.add_argument('--image_condition', type=str, default=None, help='图像条件路径')
    parser.add_argument('--text_condition', type=str, default=None, help='文本条件')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步数')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta参数')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale')
    parser.add_argument('--num_samples', type=int, default=1, help='生成样本数量')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'pt'], help='保存格式')
    parser.add_argument('--no_visualize', action='store_true', help='不生成可视化')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"模型文件不存在: {args.model_path}")
        return
    
    # 执行推理
    conditional_inference(
        model_path=args.model_path,
        output_dir=args.output_dir,
        image_condition=args.image_condition,
        text_condition=args.text_condition,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        eta=args.eta,
        cfg_scale=args.cfg_scale,
        num_samples=args.num_samples,
        device=args.device,
        save_format=args.save_format,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main() 