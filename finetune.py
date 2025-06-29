import torch
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import os
import time
from tqdm import tqdm
import argparse
import copy

from ddim_model import DDIMDiffusionModel
from data_generator import create_dataloader, visualize_features


class FineTuneDiffusionModel:
    """扩散模型微调类"""
    
    def __init__(self, pretrained_model_path, device="cuda"):
        self.device = device
        
        # 加载预训练模型
        print(f"加载预训练模型: {pretrained_model_path}")
        self.model = DDIMDiffusionModel(device=device)
        self.model.load_model(pretrained_model_path)
        
        # 保存原始模型参数（用于对比）
        self.original_state = copy.deepcopy(self.model.model.state_dict())
        
        print("预训练模型加载完成")
    
    def finetune_step(self, clean_features, optimizer, learning_rate_scheduler=None):
        """微调单步"""
        batch_size = clean_features.shape[0]
        
        # 添加噪声
        noise = torch.randn_like(clean_features)
        timesteps = torch.randint(0, self.model.train_scheduler.num_train_timesteps, (batch_size,), device=self.device)
        timesteps = timesteps.long()
        
        noisy_features = self.model.train_scheduler.add_noise(clean_features, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.model.model(noisy_features, timesteps).sample
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止过度更新）
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if learning_rate_scheduler:
            learning_rate_scheduler.step()
        
        return loss.item()
    
    def get_parameter_changes(self):
        """获取参数变化统计"""
        current_state = self.model.model.state_dict()
        changes = {}
        
        for key in current_state.keys():
            if key in self.original_state:
                diff = torch.abs(current_state[key] - self.original_state[key])
                changes[key] = {
                    'mean_change': diff.mean().item(),
                    'max_change': diff.max().item(),
                    'relative_change': (diff.mean() / (self.original_state[key].abs().mean() + 1e-8)).item()
                }
        
        return changes
    
    def save_finetuned_model(self, path, save_original_comparison=True):
        """保存微调后的模型"""
        # 保存微调后的模型
        self.model.save_model(path)
        
        if save_original_comparison:
            # 保存参数变化统计
            changes = self.get_parameter_changes()
            comparison_path = path.replace('.pth', '_changes.txt')
            
            with open(comparison_path, 'w', encoding='utf-8') as f:
                f.write("微调前后参数变化统计\n")
                f.write("=" * 50 + "\n\n")
                
                for key, stats in changes.items():
                    f.write(f"参数: {key}\n")
                    f.write(f"  平均变化: {stats['mean_change']:.6f}\n")
                    f.write(f"  最大变化: {stats['max_change']:.6f}\n")
                    f.write(f"  相对变化: {stats['relative_change']:.4f}\n")
                    f.write("-" * 30 + "\n")
        
        print(f"微调模型已保存到: {path}")


def finetune_model(
    pretrained_model_path,
    finetune_dataloader,
    num_epochs=10,
    learning_rate=1e-5,  # 微调时使用更小的学习率
    save_dir="./finetuned_checkpoints",
    log_dir="./finetune_logs",
    save_interval=2,
    device="cuda"
):
    """
    微调扩散模型
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化微调模型
    finetune_model = FineTuneDiffusionModel(pretrained_model_path, device=device)
    
    # 初始化优化器（使用更小的学习率）
    optimizer = torch.optim.AdamW(
        finetune_model.model.model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器（余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(finetune_dataloader),
        eta_min=learning_rate * 0.1
    )
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 微调循环
    global_step = 0
    best_loss = float('inf')
    
    print(f"开始微调，设备: {device}")
    print(f"微调轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"数据加载器长度: {len(finetune_dataloader)}")
    
    for epoch in range(num_epochs):
        finetune_model.model.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # 进度条
        pbar = tqdm(finetune_dataloader, desc=f"微调 Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, features in enumerate(pbar):
            # 移动数据到设备
            features = features.to(device)
            
            # 微调步骤
            loss = finetune_model.finetune_step(features, optimizer, scheduler)
            
            epoch_loss += loss
            num_batches += 1
            global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss:.6f}',
                'Avg Loss': f'{epoch_loss/num_batches:.6f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # 记录到TensorBoard
            if global_step % 5 == 0:
                writer.add_scalar('Loss/FineTune', loss, global_step)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
                
                # 记录参数变化
                if global_step % 50 == 0:
                    changes = finetune_model.get_parameter_changes()
                    avg_change = sum(stats['relative_change'] for stats in changes.values()) / len(changes)
                    writer.add_scalar('Parameters/Average_Change', avg_change, global_step)
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        
        # 记录epoch级别的指标
        writer.add_scalar('Loss/Epoch', avg_loss, epoch)
        
        print(f"微调 Epoch {epoch+1}/{num_epochs} - 平均损失: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            finetune_model.save_finetuned_model(
                os.path.join(save_dir, "best_finetuned_model.pth")
            )
            print(f"保存最佳微调模型，损失: {best_loss:.6f}")
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"finetuned_checkpoint_epoch_{epoch+1}.pth")
            finetune_model.save_finetuned_model(checkpoint_path)
            print(f"保存微调检查点: {checkpoint_path}")
            
            # 生成样本进行可视化
            finetune_model.model.model.eval()
            with torch.no_grad():
                samples = finetune_model.model.sample(batch_size=4, num_inference_steps=50)
                visualize_features(samples, os.path.join(log_dir, f"finetuned_samples_epoch_{epoch+1}.png"))
    
    # 保存最终微调模型
    finetune_model.save_finetuned_model(os.path.join(save_dir, "final_finetuned_model.pth"))
    
    # 保存参数变化报告
    changes = finetune_model.get_parameter_changes()
    report_path = os.path.join(save_dir, "parameter_changes_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("微调参数变化报告\n")
        f.write("=" * 50 + "\n\n")
        
        total_params = len(changes)
        significant_changes = sum(1 for stats in changes.values() if stats['relative_change'] > 0.01)
        
        f.write(f"总参数数量: {total_params}\n")
        f.write(f"显著变化参数数量 (>1%): {significant_changes}\n")
        f.write(f"显著变化比例: {significant_changes/total_params*100:.2f}%\n\n")
        
        # 按变化程度排序
        sorted_changes = sorted(changes.items(), key=lambda x: x[1]['relative_change'], reverse=True)
        
        f.write("参数变化排名（前10）:\n")
        for i, (key, stats) in enumerate(sorted_changes[:10]):
            f.write(f"{i+1}. {key}: {stats['relative_change']:.4f}\n")
    
    print("微调完成！")
    print(f"参数变化报告已保存到: {report_path}")
    
    writer.close()
    return finetune_model


def main():
    parser = argparse.ArgumentParser(description="微调DDIM扩散模型")
    parser.add_argument("--pretrained_model", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--batch_size", type=int, default=4, help="微调批量大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="微调轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="微调学习率")
    parser.add_argument("--num_samples", type=int, default=1000, help="微调样本数量")
    parser.add_argument("--dataset_type", type=str, default="realistic", 
                       choices=["synthetic", "realistic"], help="数据集类型")
    parser.add_argument("--save_dir", type=str, default="./finetuned_checkpoints", help="保存目录")
    parser.add_argument("--log_dir", type=str, default="./finetune_logs", help="日志目录")
    parser.add_argument("--device", type=str, default="auto", help="训练设备")
    parser.add_argument("--save_interval", type=int, default=2, help="保存间隔")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 检查预训练模型是否存在
    if not os.path.exists(args.pretrained_model):
        print(f"错误: 预训练模型文件不存在: {args.pretrained_model}")
        return
    
    # 创建微调数据加载器
    finetune_dataloader = create_dataloader(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    print(f"微调数据集类型: {args.dataset_type}")
    print(f"微调样本数量: {args.num_samples}")
    print(f"批量大小: {args.batch_size}")
    print(f"数据加载器长度: {len(finetune_dataloader)}")
    
    # 开始微调
    start_time = time.time()
    finetuned_model = finetune_model(
        pretrained_model_path=args.pretrained_model,
        finetune_dataloader=finetune_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        device=device
    )
    
    finetune_time = time.time() - start_time
    print(f"微调完成，总耗时: {finetune_time/60:.2f} 分钟")
    
    # 测试微调后的生成
    print("测试微调后的生成...")
    finetuned_model.model.model.eval()
    with torch.no_grad():
        samples = finetuned_model.model.sample(batch_size=4, num_inference_steps=50)
        visualize_features(samples, os.path.join(args.log_dir, "finetuned_final_samples.png"))
        print(f"生成样本形状: {samples.shape}")
        print(f"样本统计 - 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")


if __name__ == "__main__":
    main() 