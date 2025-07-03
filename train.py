import os
import time
import argparse
from tqdm import tqdm
from accelerate import Accelerator
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime

from ddim_model import DDIMDiffusionModel
from dataset_diff.shapenet_triplane_dataset import FeatureDatasetSingleClass
from config.config_parser import ConfigParser

class DiffusionTrainer:
    def __init__(self, args):
        self.args = args
        self.config = self.load_config(args.config)
        
        # 创建基于时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(self.config.output.exp_name, f"uncond_{timestamp}")
        self.save_dir = os.path.join(base_dir, "run")
        self.log_dir = os.path.join(base_dir, "logs")
        # 初始化 accelerator，不使用 logging_dir 参数
        self.accelerator = Accelerator()
        self.train_path = args.train_path if args.train_path else self.config.input.split_dir
        self.tri_dir = args.tri_dir
        self.writer = None
        self.model = None
        self.train_dataloader = None
        self.best_step_loss = float('inf')  # 用于step级别的最佳损失保存
        self.best_epoch_loss = float('inf')  # 用于epoch级别的早停机制
        
        # 早停机制参数
        self.patience = getattr(args, 'patience', 20)  # 等待多少个epoch没有改善就停止
        self.min_delta = getattr(args, 'min_delta', 1e-6)  # 最小改善阈值
        self.patience_counter = 0  # 计数器
        self.should_stop = False
        
        # 验证参数
        self.eval_interval = getattr(args, 'eval_interval', 10)  # 每多少epoch进行一次验证
        self.eval_samples = getattr(args, 'eval_samples', 16)  # 验证时生成的样本数量

    def load_config(self, config_path):
        parser = ConfigParser(config_path)
        config = parser.parse_config()
        # if self.accelerator.is_main_process:
        #     print("=== Diffusion相关配置参数 ===")
        #     print(f"扩散模式: {config.diff_specs.sd_mode}")
        #     print(f"扩散配置: {config.diff_specs.sd_cfg}")
        #     print(f"扩散lambda: {config.diff_specs.sd_lambda}")
        #     print(f"扩散训练轮数: {config.diff_specs.num_epochs}")
        #     print(f"扩散批次大小: {config.diff_specs.batch_size}")
        #     print(f"扩散学习率: {config.diff_specs.lr}")
        #     print(f"扩散学习率调度器: {config.diff_specs.lr_scheduler_type}")
        #     print(f"Beta调度: {config.diff_specs.beta_schedule}")
        #     print(f"VAE缩放因子: {config.diff_specs.vae_scaling_factor}")
        #     print(f"噪声偏移: {config.diff_specs.noise_offset}")
        #     print(f"输入扰动: {config.diff_specs.input_perturbation}")
        #     print(f"指导尺度: {config.diff_specs.guidance_scale}")
        #     print(f"扩散零SNR: {config.diff_specs.diffusion_zero_snr}")
        #     print(f"扩散裁剪标志: {config.diff_specs.diffusion_clip_flag}")
        #     print(f"模型名称: {config.diff_specs.model_name}")
        #     print(f"图像分辨率: {config.diff_specs.image_resolution}")
        #     print(f"随机视角: {config.diff_specs.random_view}")
        return config

    def prepare_dataset(self):
        dataset = FeatureDatasetSingleClass(
            train_path=self.train_path,
            tri_dir=self.tri_dir
        )
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.diff_specs.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        if self.accelerator.is_main_process:
            print(f"数据加载器长度: {len(self.train_dataloader)}")

    def prepare_model(self):
        self.model = DDIMDiffusionModel(device=self.accelerator.device)
        self.optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=self.config.diff_specs.lr,
            weight_decay=getattr(self.config.diff_specs, 'decay', 0.0)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.diff_specs.num_epochs)
        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )
        if self.accelerator.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 复制配置文件到输出文件夹
            config_backup_path = os.path.join(os.path.dirname(self.save_dir), "specs_config.json")
            import shutil
            try:
                shutil.copy2(self.args.config, config_backup_path)
                print(f"配置文件已备份到: {config_backup_path}")
            except Exception as e:
                print(f"警告：配置文件备份失败: {e}")
            
            self.writer = SummaryWriter(self.log_dir)
            # 初始化 tensorboard 追踪器
            self.accelerator.init_trackers("diffusion_training", config={
                "lr": self.config.diff_specs.lr,
                "batch_size": self.config.diff_specs.batch_size,
                "num_epochs": self.config.diff_specs.num_epochs
            })

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"未找到checkpoint: {checkpoint_path}")
            return
        state_dict = torch.load(checkpoint_path, map_location=self.accelerator.device)
        self.model.model.load_state_dict(state_dict)
        print(f"已加载checkpoint: {checkpoint_path}")

    def train(self):
        num_epochs = self.config.diff_specs.num_epochs
        global_step = 0
        if self.accelerator.is_main_process:
            print(f"开始训练，设备: {self.accelerator.device}")
            print(f"总epoch数: {num_epochs}")
            print(f"学习率: {self.config.diff_specs.lr}")
        for epoch in range(num_epochs):
            self.model.model.train()
            epoch_loss = 0.0
            num_batches = 0
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not self.accelerator.is_main_process)
            for batch_idx, features in enumerate(pbar):
                features = features.to(self.accelerator.device)
                
                # 手动计算损失（不使用 train_step，因为它包含了优化步骤）
                batch_size = features.shape[0]
                noise = torch.randn_like(features)
                timesteps = torch.randint(0, self.model.train_scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device)
                timesteps = timesteps.long()
                
                noisy_features = self.model.train_scheduler.add_noise(features, noise, timesteps)
                noise_pred = self.model.model(noisy_features, timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                loss_value = loss.item()
                epoch_loss += loss_value
                num_batches += 1
                global_step += 1
                if self.accelerator.is_main_process:
                    pbar.set_postfix({
                        'Loss': f'{loss_value:.6f}',
                        'Avg Loss': f'{epoch_loss/num_batches:.6f}',
                        'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })
                    if self.writer and global_step % 10 == 0:
                        self.writer.add_scalar('Loss/Train', loss_value, global_step)
                        self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], global_step)
                    # loss变优时保存模型
                    if loss_value < self.best_step_loss:
                        self.best_step_loss = loss_value
                        save_path = os.path.join(self.save_dir, "best_step_model.pth")
                        torch.save(self.model.model.state_dict(), save_path)
                        print(f"[Step {global_step}] 保存最佳step模型，损失: {self.best_step_loss:.6f}")
            avg_loss = epoch_loss / num_batches
            if self.accelerator.is_main_process:
                if self.writer:
                    self.writer.add_scalar('Loss/Epoch', avg_loss, epoch)
                    self.writer.flush()
                print(f"Epoch {epoch+1}/{num_epochs} - 平均损失: {avg_loss:.6f}")
                
                # 早停机制检查
                if avg_loss < self.best_epoch_loss - self.min_delta:
                    # 损失有明显改善
                    self.best_epoch_loss = avg_loss
                    self.patience_counter = 0
                    # 保存最佳epoch模型
                    best_epoch_path = os.path.join(self.save_dir, "best_epoch_model.pth")
                    torch.save(self.model.model.state_dict(), best_epoch_path)
                    print(f"[Epoch {epoch+1}] 新的最佳epoch损失: {self.best_epoch_loss:.6f}, 已保存模型")
                else:
                    # 损失没有改善
                    self.patience_counter += 1
                    print(f"[Epoch {epoch+1}] 损失未改善，耐心计数: {self.patience_counter}/{self.patience}")
                    
                    if self.patience_counter >= self.patience:
                        print(f"早停：连续 {self.patience} 个epoch损失未改善，停止训练")
                        self.should_stop = True
                        break
                
                # 定期验证
                if (epoch + 1) % self.eval_interval == 0:
                    self.validate_model(epoch)
        if self.accelerator.is_main_process:
            torch.save(self.model.model.state_dict(), os.path.join(self.save_dir, "final_model.pth"))
            print("训练完成！")
            if self.writer:
                self.writer.close()

    def validate_model(self, epoch):
        """
        验证模型：生成样本并计算质量指标
        """
        if not self.accelerator.is_main_process:
            return
        
        print(f"\n=== Epoch {epoch+1} 验证开始 ===")
        self.model.model.eval()
        
        # 记录真实数据统计信息（用于对比）
        real_features_stats = self.compute_real_data_stats()
        
        # 生成样本
        with torch.no_grad():
            generated_features = self.model.sample(
                batch_size=self.eval_samples,
                num_inference_steps=50,
                eta=0.0
            )
        
        # 计算生成数据统计信息
        generated_stats = self.compute_generated_stats(generated_features)
        
        # 计算验证指标
        metrics = self.compute_validation_metrics(real_features_stats, generated_stats, generated_features)
        
        # 记录到tensorboard
        if self.writer:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
        
        # 打印验证结果
        print(f"验证指标:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")
        
        # 保存验证样本
        val_save_path = os.path.join(self.save_dir, f"validation_epoch_{epoch+1}.pt")
        torch.save(generated_features, val_save_path)
        print(f"验证样本已保存到: {val_save_path}")
        print(f"=== Epoch {epoch+1} 验证结束 ===\n")
        
        self.model.model.train()
        return metrics
    
    def compute_real_data_stats(self):
        """计算真实数据的统计信息"""
        real_features = []
        with torch.no_grad():
            for i, features in enumerate(self.train_dataloader):
                real_features.append(features)
                if i >= 10:  # 只计算前几个batch的统计信息
                    break
        
        real_features = torch.cat(real_features, dim=0)
        return {
            'mean': real_features.mean().item(),
            'std': real_features.std().item(),
            'min': real_features.min().item(),
            'max': real_features.max().item()
        }
    
    def compute_generated_stats(self, generated_features):
        """计算生成数据的统计信息"""
        return {
            'mean': generated_features.mean().item(),
            'std': generated_features.std().item(),
            'min': generated_features.min().item(),
            'max': generated_features.max().item()
        }
    
    def compute_validation_metrics(self, real_stats, gen_stats, generated_features):
        """计算验证指标"""
        metrics = {}
        
        # 1. 统计分布差异
        metrics['mean_diff'] = abs(real_stats['mean'] - gen_stats['mean'])
        metrics['std_diff'] = abs(real_stats['std'] - gen_stats['std'])
        
        # 2. 生成质量指标
        metrics['generated_mean'] = gen_stats['mean']
        metrics['generated_std'] = gen_stats['std']
        
        # 3. 多样性指标（样本间的平均距离）
        if generated_features.shape[0] > 1:
            # 计算所有样本对之间的L2距离
            flat_features = generated_features.view(generated_features.shape[0], -1)
            pairwise_distances = torch.cdist(flat_features, flat_features, p=2)
            # 取上三角矩阵（排除对角线和重复）
            mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1).bool()
            diversity = pairwise_distances[mask].mean().item()
            metrics['diversity'] = diversity
        else:
            metrics['diversity'] = 0.0
        
        # 4. 特征范围合理性
        metrics['feature_range'] = gen_stats['max'] - gen_stats['min']
        
        return metrics




def main():
    parser = argparse.ArgumentParser(description="训练DDIM扩散模型")
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径 (例如: config/exp-shapenet-car/specs_shapenet_all.json)")
    parser.add_argument("--train_path", type=str, default=None, help="训练数据路径（可选，默认使用配置文件中的路径）")
    parser.add_argument("--tri_dir", type=str, default=None, help="三平面数据目录（可选，默认使用配置文件中的路径）")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint路径，继续训练")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值：多少个epoch损失不改善就停止训练")
    parser.add_argument("--min_delta", type=float, default=1e-6, help="早停最小改善阈值")
    parser.add_argument("--eval_interval", type=int, default=10, help="验证间隔：每多少epoch进行一次验证")
    parser.add_argument("--eval_samples", type=int, default=16, help="验证时生成的样本数量")
    args = parser.parse_args()

    trainer = DiffusionTrainer(args)
    trainer.prepare_dataset()
    trainer.prepare_model()
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

if __name__ == "__main__":
    main() 