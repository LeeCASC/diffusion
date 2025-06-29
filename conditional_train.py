import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from conditional_diffusion import ConditionalDDIMDiffusionModel, create_conditional_dataset
from data_generator import FeatureGenerator


def train_conditional_model(
    model,
    train_loader,
    optimizer,
    num_epochs,
    device,
    save_dir,
    save_interval=1000,
    log_interval=100
):
    """训练条件扩散模型"""
    
    model.model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            features = batch['features'].to(device)
            image_condition = batch['image_condition'].to(device)
            text_condition = batch['text_condition']
            
            # 训练步骤
            loss = model.train_step(features, optimizer, image_condition, text_condition)
            epoch_loss += loss
            losses.append(loss)
            
            # 更新进度条
            progress_bar.set_postfix({'Loss': f'{loss:.4f}'})
            
            # 保存检查点
            if batch_idx % save_interval == 0:
                save_path = os.path.join(save_dir, f'conditional_model_epoch_{epoch}_step_{batch_idx}.pth')
                model.save_model(save_path)
                print(f"模型已保存到: {save_path}")
            
            # 记录日志
            if batch_idx % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                print(f"Epoch {epoch+1}, Step {batch_idx}, Average Loss: {avg_loss:.4f}")
        
        # 每个epoch结束后保存
        save_path = os.path.join(save_dir, f'conditional_model_epoch_{epoch+1}.pth')
        model.save_model(save_path)
        print(f"Epoch {epoch+1} 完成，模型已保存到: {save_path}")
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_dir, 'training_loss.png'))
        plt.close()
    
    return losses


def finetune_conditional_model(
    base_model_path,
    train_loader,
    optimizer,
    num_epochs,
    device,
    save_dir,
    finetune_type="full",  # "full", "lora", "adapter"
    save_interval=500,
    log_interval=50
):
    """微调条件扩散模型"""
    
    if finetune_type == "lora":
        from conditional_diffusion import ConditionalLoRAModel
        model = ConditionalLoRAModel(base_model_path, device=str(device))
        trainable_params = model.get_lora_parameters()
    else:
        # 全参数微调
        model = ConditionalDDIMDiffusionModel(device=str(device))
        model.load_model(base_model_path)
        trainable_params = model.model.parameters()
    
    model.model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            features = batch['features'].to(device)
            image_condition = batch['image_condition'].to(device)
            text_condition = batch['text_condition']
            
            # 训练步骤
            loss = model.train_step(features, optimizer, image_condition, text_condition)
            epoch_loss += loss
            losses.append(loss)
            
            # 更新进度条
            progress_bar.set_postfix({'Loss': f'{loss:.4f}'})
            
            # 保存检查点
            if batch_idx % save_interval == 0:
                if finetune_type == "lora":
                    save_path = os.path.join(save_dir, f'conditional_lora_epoch_{epoch}_step_{batch_idx}.pth')
                    model.save_lora_weights(save_path)
                else:
                    save_path = os.path.join(save_dir, f'conditional_finetuned_epoch_{epoch}_step_{batch_idx}.pth')
                    model.save_model(save_path)
                print(f"模型已保存到: {save_path}")
        
        # 每个epoch结束后保存
        if finetune_type == "lora":
            save_path = os.path.join(save_dir, f'conditional_lora_epoch_{epoch+1}.pth')
            model.save_lora_weights(save_path)
        else:
            save_path = os.path.join(save_dir, f'conditional_finetuned_epoch_{epoch+1}.pth')
            model.save_model(save_path)
        print(f"Epoch {epoch+1} 完成，模型已保存到: {save_path}")
    
    return losses


def main():
    parser = argparse.ArgumentParser(description='训练条件扩散模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'finetune'],
                       help='训练模式: train(从头训练) 或 finetune(微调)')
    parser.add_argument('--base_model_path', type=str, default=None,
                       help='基础模型路径（微调时需要）')
    parser.add_argument('--finetune_type', type=str, default='full', 
                       choices=['full', 'lora', 'adapter'],
                       help='微调类型')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./conditional_models', help='保存目录')
    parser.add_argument('--image_dir', type=str, default=None, help='图像数据目录')
    parser.add_argument('--text_file', type=str, default=None, help='文本数据文件')
    parser.add_argument('--num_samples', type=int, default=10000, help='数据集大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("创建数据集...")
    dataset = create_conditional_dataset(
        image_dir=args.image_dir,
        text_file=args.text_file,
        num_samples=args.num_samples
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"批次数量: {len(train_loader)}")
    
    # 创建优化器
    if args.mode == 'train':
        # 从头训练
        print("创建条件扩散模型...")
        model = ConditionalDDIMDiffusionModel(device=str(device))
        optimizer = optim.Adam(model.model.parameters(), lr=args.learning_rate)
        
        print("开始训练...")
        losses = train_conditional_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            device=device,
            save_dir=args.save_dir
        )
        
    elif args.mode == 'finetune':
        # 微调
        if args.base_model_path is None:
            raise ValueError("微调模式需要指定基础模型路径")
        
        print(f"加载基础模型: {args.base_model_path}")
        
        if args.finetune_type == "lora":
            from conditional_diffusion import ConditionalLoRAModel
            model = ConditionalLoRAModel(args.base_model_path, device=str(device))
            optimizer = optim.Adam(model.get_lora_parameters(), lr=args.learning_rate * 0.1)
        else:
            model = ConditionalDDIMDiffusionModel(device=str(device))
            model.load_model(args.base_model_path)
            optimizer = optim.Adam(model.model.parameters(), lr=args.learning_rate * 0.1)
        
        print("开始微调...")
        losses = finetune_conditional_model(
            base_model_path=args.base_model_path,
            train_loader=train_loader,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            device=device,
            save_dir=args.save_dir,
            finetune_type=args.finetune_type
        )
    
    # 保存训练配置
    config = {
        'mode': args.mode,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': str(device),
        'dataset_size': len(dataset),
        'final_loss': losses[-1] if losses else None
    }
    
    if args.mode == 'finetune':
        config['base_model_path'] = args.base_model_path
        config['finetune_type'] = args.finetune_type
    
    with open(os.path.join(args.save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("训练完成！")
    print(f"模型和配置已保存到: {args.save_dir}")


if __name__ == "__main__":
    main() 