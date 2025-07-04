#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图控扩散模型训练脚本
支持2D图像控制的特征生成训练
按照train.py的格式编写
支持两种训练模式：原始图像模式 和 预提取特征模式
"""

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime

from image_controlled_unet_v2 import CrossAttentionImageControlledDDIMModel
from dataset_diff.image_controlled_dataset import ImageControlledDataset, ImageControlledDatasetWithCategories, create_dual_mode_dataset
from config.config_parser import ConfigParser


def cleanup_memory():
    """清理显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def validate_model(model, val_loader, device, epoch, writer):
    """验证模型并生成样本"""
    model.model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        # 计算验证损失
        for batch in val_loader:
            features = batch['features'].to(device)
            control_type = batch['control_type'][0]  # 假设batch内类型一致
            
            # 根据控制类型获取控制数据
            if control_type == 'raw_image':
                control_data = batch['control_image'].to(device)
            else:  # image_features
                control_data = batch['control_features'].to(device)
            
            # 手动计算损失（与训练时相同的方式）
            batch_size = features.shape[0]
            noise = torch.randn_like(features)
            timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
            
            noisy_features = model.train_scheduler.add_noise(features, noise, timesteps)
            noise_pred = model.model(noisy_features, timesteps, control_data, control_type).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:  # 限制验证批次数
                break
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # 生成验证样本
    validation_samples = {}
    try:
        # 选择第一个batch的控制数据进行生成
        first_batch = next(iter(val_loader))
        control_type = first_batch['control_type'][0]
        
        if control_type == 'raw_image':
            control_data = first_batch['control_image'][:4].to(device)  # 取前4个
        else:
            control_data = first_batch['control_features'][:4].to(device)
        
        # 生成样本
        generated_features = model.generate(
            control_data=control_data,
            control_type=control_type,
            num_inference_steps=50
        )
        
        validation_samples['generated'] = generated_features.cpu()
        validation_samples['control_data'] = control_data.cpu()
        validation_samples['control_type'] = control_type
        validation_samples['original_features'] = first_batch['features'][:4]
        
        # 计算生成样本的统计指标
        sample_mean = generated_features.mean().item()
        sample_std = generated_features.std().item()
        sample_min = generated_features.min().item()
        sample_max = generated_features.max().item()
        
        # 记录到tensorboard
        if writer:
            writer.add_scalar('Validation/Generated_Mean', sample_mean, epoch)
            writer.add_scalar('Validation/Generated_Std', sample_std, epoch)
            writer.add_scalar('Validation/Generated_Min', sample_min, epoch)
            writer.add_scalar('Validation/Generated_Max', sample_max, epoch)
            
    except Exception as e:
        print(f"警告: 生成验证样本时出错: {e}")
    
    # 记录总体验证损失
    if writer:
        writer.add_scalar('Validation/Loss', avg_loss, epoch)
    
    model.model.train()
    return avg_loss, validation_samples


def save_validation_results(validation_samples, epoch, save_dir):
    """保存验证结果"""
    if validation_samples:
        val_save_path = os.path.join(save_dir, f"validation_epoch_{epoch+1}.pt")
        torch.save(validation_samples, val_save_path)
        print(f"验证样本已保存到: {val_save_path}")


class ImageControlledDiffusionTrainer:
    def __init__(self, args):
        self.args = args
        self.config = self.load_config(args.config)
        
        # 创建基于时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_mode = getattr(args, 'training_mode', 'auto')
        base_dir = os.path.join(self.config.output.exp_name, f"img_ctrl_{training_mode}_{timestamp}")
        self.save_dir = os.path.join(base_dir, "run")
        self.log_dir = os.path.join(base_dir, "logs")
        
        # 初始化 accelerator
        self.accelerator = Accelerator()
        self.train_path = args.train_path if args.train_path else self.config.input.split_dir
        self.tri_dir = args.tri_dir
        self.image_dir = args.image_dir
        self.writer = None
        self.model = None
        self.train_dataloader = None
        self.best_step_loss = float('inf')
        self.best_epoch_loss = float('inf')
        
        # 早停机制参数
        self.patience = getattr(args, 'patience', 20)
        self.min_delta = getattr(args, 'min_delta', 1e-6)
        self.patience_counter = 0
        self.should_stop = False
        
        # 验证参数
        self.eval_interval = getattr(args, 'eval_interval', 10)
        self.eval_samples = getattr(args, 'eval_samples', 16)
        
        # 图控特定参数
        self.image_size = getattr(args, 'image_size', 256)
        self.control_strength = getattr(args, 'control_strength', 1.0)
        self.training_mode = getattr(args, 'training_mode', 'auto')
        self.raw_image_dir = getattr(args, 'raw_image_dir', None)
        self.image_feature_dir = getattr(args, 'image_feature_dir', None)
        
        # Cross-attention参数
        self.cross_attention_dim = getattr(args, 'cross_attention_dim', 1024)
        self.use_clip = getattr(args, 'use_clip', True)

    def load_config(self, config_path):
        parser = ConfigParser(config_path)
        config = parser.parse_config()
        return config

    def prepare_dataset(self):
        # 使用双模式数据集创建函数
        use_categories = getattr(self.args, 'use_categories', False)
        
        dataset = create_dual_mode_dataset(
            train_path=self.train_path,
            tri_dir=self.tri_dir,
            image_dir=self.image_dir,
            raw_image_dir=self.raw_image_dir,
            image_feature_dir=self.image_feature_dir,
            prefer_mode=self.training_mode,
            image_size=self.image_size,
            use_categories=use_categories
        )
        
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.diff_specs.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # 记录实际使用的图像模式
        if hasattr(dataset, 'image_mode'):
            self.actual_image_mode = dataset.image_mode
        else:
            self.actual_image_mode = 'unknown'
        
        if self.accelerator.is_main_process:
            print(f"数据加载器长度: {len(self.train_dataloader)}")
            print(f"图控数据集类型: {'支持类别' if use_categories else '基础图控'}")
            print(f"实际图像模式: {self.actual_image_mode}")

    def prepare_model(self):
        # 使用实际的图像模式创建模型
        model_training_mode = self.actual_image_mode if hasattr(self, 'actual_image_mode') else self.training_mode
        
        self.model = CrossAttentionImageControlledDDIMModel(
            in_channels=256,
            out_channels=256,
            image_size=self.image_size,
            feature_size=(16, 48),
            cross_attention_dim=self.cross_attention_dim,
            control_strength=self.control_strength,
            training_mode=model_training_mode,
            use_clip=self.use_clip,
            device=self.accelerator.device
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=self.config.diff_specs.lr,
            weight_decay=getattr(self.config.diff_specs, 'decay', 0.0)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.diff_specs.num_epochs)
        
        self.model.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model.model, self.optimizer, self.train_dataloader, self.scheduler
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
            self.accelerator.init_trackers("image_controlled_training", config={
                "lr": self.config.diff_specs.lr,
                "batch_size": self.config.diff_specs.batch_size,
                "num_epochs": self.config.diff_specs.num_epochs,
                "image_size": self.image_size,
                "control_strength": self.control_strength
            })

    def save_complete_checkpoint(self, epoch, global_step, avg_loss, filename="final.pth"):
        """保存完整的训练checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint = {
            # 模型状态
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            
            # 训练进度
            'epoch': epoch,
            'global_step': global_step,
            'current_loss': avg_loss,
            
            # 早停和最佳模型追踪
            'best_step_loss': self.best_step_loss,
            'best_epoch_loss': self.best_epoch_loss,
            'patience_counter': self.patience_counter,
            
            # 训练配置
            'config_path': self.args.config if hasattr(self.args, 'config') else None,
            'train_path': self.args.train_path if hasattr(self.args, 'train_path') else None,
            'tri_dir': self.args.tri_dir if hasattr(self.args, 'tri_dir') else None,
            'image_dir': self.args.image_dir if hasattr(self.args, 'image_dir') else None,
            'image_size': self.image_size,
            'control_strength': self.control_strength,
            'cross_attention_dim': self.cross_attention_dim,
            'use_clip': self.use_clip,
            'training_mode': self.training_mode,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'eval_interval': self.eval_interval,
            'eval_samples': self.eval_samples,
            
            # 时间戳
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # 随机状态
            'random_state': torch.get_rng_state().cpu(),
            'numpy_random_state': np.random.get_state(),
        }
        
        # 保存checkpoint
        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"已保存完整checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"未找到checkpoint: {checkpoint_path}")
            return None
        
        print(f"正在加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        
        # 检查是否是完整的checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整checkpoint
            print("检测到完整checkpoint，正在恢复所有训练状态...")
            
            # 恢复模型状态
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 恢复优化器和调度器状态
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("已恢复优化器状态")
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("已恢复学习率调度器状态")
            
            # 恢复训练进度
            start_epoch = checkpoint.get('epoch', 0) + 1
            global_step = checkpoint.get('global_step', 0)
            
            # 恢复早停和最佳模型状态
            self.best_step_loss = checkpoint.get('best_step_loss', float('inf'))
            self.best_epoch_loss = checkpoint.get('best_epoch_loss', float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            
            # 恢复随机状态
            if 'random_state' in checkpoint:
                torch.set_rng_state(checkpoint['random_state'])
                print("已恢复PyTorch随机状态")
            
            if 'numpy_random_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_random_state'])
                print("已恢复NumPy随机状态")
            
            print(f"已恢复训练状态:")
            print(f"  起始epoch: {start_epoch}")
            print(f"  全局步数: {global_step}")
            print(f"  最佳step损失: {self.best_step_loss:.6f}")
            print(f"  最佳epoch损失: {self.best_epoch_loss:.6f}")
            print(f"  耐心计数: {self.patience_counter}/{self.patience}")
            
            return {
                'start_epoch': start_epoch,
                'global_step': global_step,
            }
        else:
            # 旧格式的权重文件
            print("检测到旧格式模型权重，仅加载模型参数...")
            self.model.model.load_state_dict(checkpoint)
            return {
                'start_epoch': 0,
                'global_step': 0,
            }

    def train(self):
        num_epochs = self.config.diff_specs.num_epochs
        start_epoch = 0
        global_step = 0
        
        # 获取实际使用的训练模式
        model_training_mode = self.actual_image_mode if hasattr(self, 'actual_image_mode') else self.training_mode
        
        # 如果有resume checkpoint，加载它
        if hasattr(self.args, 'resume') and self.args.resume:
            resume_info = self.load_checkpoint(self.args.resume)
            if resume_info:
                start_epoch = resume_info['start_epoch']
                global_step = resume_info['global_step']
        
        if self.accelerator.is_main_process:
            print(f"开始图控训练 (Cross-Attention架构)，设备: {self.accelerator.device}")
            print(f"总epoch数: {num_epochs}")
            print(f"起始epoch: {start_epoch + 1}")
            print(f"学习率: {self.config.diff_specs.lr}")
            print(f"图像尺寸: {self.image_size}")
            print(f"控制强度: {self.control_strength}")
            print(f"Cross-attention维度: {self.cross_attention_dim}")
            print(f"使用CLIP编码器: {self.use_clip}")
            print(f"训练模式: {model_training_mode}")
        
        for epoch in range(start_epoch, num_epochs):
            self.model.model.train()
            epoch_loss = 0.0
            num_batches = 0
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not self.accelerator.is_main_process)
            
            for batch_idx, batch in enumerate(pbar):
                features = batch['features'].to(self.accelerator.device)
                
                # 根据控制数据类型获取控制输入
                control_type = batch['control_type'][0]  # 批次中所有样本的控制类型应该相同
                if control_type == 'raw_image':
                    control_data = batch['control_image'].to(self.accelerator.device)
                elif control_type == 'image_features':
                    control_data = batch['control_features'].to(self.accelerator.device)
                else:
                    raise ValueError(f"未知的控制类型: {control_type}")
                
                # 手动计算损失（类似train.py中的方式）
                batch_size = features.shape[0]
                noise = torch.randn_like(features)
                timesteps = torch.randint(0, 1000, (batch_size,), device=self.accelerator.device)
                timesteps = timesteps.long()
                
                noisy_features = self.model.train_scheduler.add_noise(features, noise, timesteps)
                noise_pred = self.model.model(noisy_features, timesteps, control_data, control_type).sample
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
                        # 早停时保存完整checkpoint
                        self.save_complete_checkpoint(epoch, global_step, avg_loss, "final.pth")
                        self.should_stop = True
                        # 早停时也清理显存
                        cleanup_memory()
                        break
                
                # 定期验证
                if (epoch + 1) % self.eval_interval == 0:
                    self.validate_model(epoch)
        
        # 训练结束时保存完整checkpoint
        if self.accelerator.is_main_process:
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            # 保存完整的final checkpoint
            self.save_complete_checkpoint(epoch, global_step, avg_loss, "final.pth")
            # 为了兼容性，也保存纯模型权重
            torch.save(self.model.model.state_dict(), os.path.join(self.save_dir, "final_model.pth"))
            print("图控训练完成！")
            if self.writer:
                self.writer.close()
        
        # 训练结束后清理显存
        cleanup_memory()

    def validate_model(self, epoch):
        """验证模型：生成图控样本并计算质量指标"""
        if not self.accelerator.is_main_process:
            return
        
        print(f"\n=== Epoch {epoch+1} 图控验证开始 ===")
        
        # 计算验证损失并生成样本
        avg_loss, validation_samples = validate_model(
            self.model, self.train_dataloader, self.accelerator.device, epoch, self.writer
        )
        
        print(f"验证损失: {avg_loss:.6f}")
        
        # 保存验证样本
        save_validation_results(validation_samples, epoch, self.save_dir)
        
        print(f"=== Epoch {epoch+1} 图控验证结束 ===\n")
        
        return avg_loss

    def cleanup_memory(self):
        """清理显存"""
        cleanup_memory()


def main():
    parser = argparse.ArgumentParser(description="训练图控DDIM扩散模型")
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径")
    parser.add_argument("--train_path", type=str, default=None,
                       help="训练数据路径（可选，默认使用配置文件中的路径）")
    parser.add_argument("--tri_dir", type=str, default=None,
                       help="三平面数据目录（可选，默认使用配置文件中的路径）")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="控制图像目录")
    parser.add_argument("--resume", type=str, default=None,
                       help="checkpoint路径，继续训练")
    parser.add_argument("--patience", type=int, default=20,
                       help="早停耐心值：多少个epoch损失不改善就停止训练")
    parser.add_argument("--min_delta", type=float, default=1e-6,
                       help="早停最小改善阈值")
    parser.add_argument("--eval_interval", type=int, default=10,
                       help="验证间隔：每多少epoch进行一次验证")
    parser.add_argument("--eval_samples", type=int, default=16,
                       help="验证时生成的样本数量")
    parser.add_argument("--image_size", type=int, default=256,
                       help="控制图像尺寸")
    parser.add_argument("--control_strength", type=float, default=1.0,
                       help="控制强度")
    parser.add_argument("--use_categories", action="store_true",
                       help="是否使用支持类别的数据集")
    parser.add_argument("--cross_attention_dim", type=int, default=1024,
                       help="Cross-attention维度")
    parser.add_argument("--use_clip", action="store_true", default=True,
                       help="是否使用CLIP编码器（原始图像模式）")
    args = parser.parse_args()

    trainer = ImageControlledDiffusionTrainer(args)
    trainer.prepare_dataset()
    trainer.prepare_model()
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main() 