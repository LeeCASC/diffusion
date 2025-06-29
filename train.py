import torch
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import os
import time
from tqdm import tqdm
import argparse

from ddim_model import DDIMDiffusionModel
from data_generator import create_dataloader, visualize_features


def train_model(
    model,
    train_dataloader,
    num_epochs=100,
    learning_rate=1e-4,
    save_dir="./checkpoints",
    log_dir="./logs",
    save_interval=10,
    device="cuda"
):
    """
    训练DDIM扩散模型
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化优化器和学习率调度器
    optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    global_step = 0
    best_loss = float('inf')
    
    print(f"开始训练，设备: {device}")
    print(f"总epoch数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    
    for epoch in range(num_epochs):
        model.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # 进度条
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, features in enumerate(pbar):
            # 移动数据到设备
            features = features.to(device)
            
            # 训练步骤
            loss = model.train_step(features, optimizer)
            
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
            if global_step % 10 == 0:
                writer.add_scalar('Loss/Train', loss, global_step)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        
        # 更新学习率
        scheduler.step()
        
        # 记录epoch级别的指标
        writer.add_scalar('Loss/Epoch', avg_loss, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} - 平均损失: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_model(os.path.join(save_dir, "best_model.pth"))
            print(f"保存最佳模型，损失: {best_loss:.6f}")
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            model.save_model(checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
            
            # 生成样本进行可视化
            model.model.eval()
            with torch.no_grad():
                samples = model.sample(batch_size=4, num_inference_steps=50)
                visualize_features(samples, os.path.join(log_dir, f"samples_epoch_{epoch+1}.png"))
    
    # 保存最终模型
    model.save_model(os.path.join(save_dir, "final_model.pth"))
    print("训练完成！")
    
    writer.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="训练DDIM扩散模型")
    parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_samples", type=int, default=10000, help="训练样本数量")
    parser.add_argument("--dataset_type", type=str, default="realistic", 
                       choices=["synthetic", "realistic"], help="数据集类型")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志保存目录")
    parser.add_argument("--device", type=str, default="auto", help="训练设备")
    parser.add_argument("--save_interval", type=int, default=10, help="保存间隔")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 创建模型
    model = DDIMDiffusionModel(device=device)
    
    # 创建数据加载器
    train_dataloader = create_dataloader(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    print(f"数据集类型: {args.dataset_type}")
    print(f"训练样本数量: {args.num_samples}")
    print(f"批量大小: {args.batch_size}")
    print(f"数据加载器长度: {len(train_dataloader)}")
    
    # 开始训练
    start_time = time.time()
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        device=device
    )
    
    training_time = time.time() - start_time
    print(f"训练完成，总耗时: {training_time/3600:.2f} 小时")
    
    # 测试生成
    print("测试生成样本...")
    trained_model.model.eval()
    with torch.no_grad():
        samples = trained_model.sample(batch_size=4, num_inference_steps=50)
        visualize_features(samples, os.path.join(args.log_dir, "final_samples.png"))
        print(f"生成样本形状: {samples.shape}")
        print(f"样本统计 - 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")


if __name__ == "__main__":
    main() 