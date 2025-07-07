#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本控制扩散模型训练脚本
支持文本控制的特征生成训练
按照train_image_controlled.py的格式编写
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

from text_controlled_unet import TextControlledDDIMModel
from dataset_diff.text_controlled_dataset import create_text_controlled_dataset
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
            text_prompts = batch['text_prompt']  # 文本列表
            
            # 手动计算损失（与训练时相同的方式）
            batch_size = features.shape[0]
            noise = torch.randn_like(features)
            timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
            
            noisy_features = model.train_scheduler.add_noise(features, noise, timesteps)
            noise_pred = model.model(noisy_features, timesteps, text_prompts).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:  # 限制验证批次数
                break
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # 生成验证样本
    validation_samples = {}
    try:
        # 选择第一个batch的文本进行生成
        first_batch = next(iter(val_loader))
        text_prompts = first_batch['text_prompt'][:4]  # 取前4个文本
        
        # 生成样本
        generated_features = model.generate(
            text_prompts=text_prompts,
            num_inference_steps=50
        )
        
        validation_samples['generated'] = generated_features.cpu()
        validation_samples['text_prompts'] = text_prompts
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


class TextControlledDiffusionTrainer:
    def __init__(self, args):
        self.args = args
        self.config = self.load_config(args.config)
        
        # 创建基于时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(self.config.output.exp_name, f"text_ctrl_{timestamp}")
        self.save_dir = os.path.join(base_dir, "run")
        self.log_dir = os.path.join(base_dir, "logs")
        
        # 初始化 accelerator
        self.accelerator = Accelerator()
        self.train_path = args.train_path if args.train_path else self.config.input.split_dir
        self.tri_dir = args.tri_dir
        self.text_json_path = args.text_json_path
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
        
        # 文本控制特定参数
        self.control_strength = getattr(args, 'control_strength', 1.0)
        self.random_text = getattr(args, 'random_text', True)
        
        # Cross-attention参数
        self.cross_attention_dim = getattr(args, 'cross_attention_dim', 1024)
        self.use_clip = getattr(args, 'use_clip', True)
        self.max_text_length = getattr(args, 'max_text_length', 77)

    def load_config(self, config_path):
        parser = ConfigParser(config_path)
        config = parser.parse_config()
        return config

    def prepare_dataset(self):
        """准备数据集"""
        # 准备训练路径
        if self.train_path.endswith('.txt'):
            train_file = self.train_path
        else:
            train_file = os.path.join(self.train_path, "train.txt")
        
        # 准备验证路径 - 先尝试找验证集，如果没有就用训练集
        val_file = None
        if self.train_path.endswith('.txt'):
            val_file = self.train_path.replace('train.txt', 'val.txt')
        else:
            val_file = os.path.join(self.train_path, "val.txt")
        
        if not os.path.exists(val_file):
            print("警告: 未找到验证集文件，使用训练集作为验证集")
            val_file = train_file

        print(f"训练数据文件: {train_file}")
        print(f"验证数据文件: {val_file}")
        print(f"特征目录: {self.tri_dir}")
        print(f"文本描述文件: {self.text_json_path}")
        print(f"随机选择文本: {self.random_text}")
        
        # 创建文本控制数据集
        train_dataset = create_text_controlled_dataset(
            train_path=train_file,
            tri_dir=self.tri_dir,
            text_json_path=self.text_json_path,
            use_categories=False,
            random_text=self.random_text
        )
        
        # # 创建验证数据集（现在总是有验证文件）
        # val_dataset = create_text_controlled_dataset(
        #     train_path=val_file,
        #     tri_dir=self.tri_dir,
        #     text_json_path=self.text_json_path,
        #     use_categories=False,
        #     random_text=False  # 验证时使用固定文本
        # )
        val_dataset = train_dataset

        # 创建数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.diff_specs.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # 总是创建验证数据加载器
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.diff_specs.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(f"训练数据集大小: {len(train_dataset)}")
        print(f"训练批次数: {len(self.train_dataloader)}")
        print(f"验证数据集大小: {len(val_dataset)}")
        print(f"验证批次数: {len(self.val_dataloader)}")

    def prepare_model(self):
        """准备模型"""
        # 获取特征尺寸信息
        feature_size = (16, 48)  # 默认特征尺寸
        in_channels = self.config.arch_specs.latent_dim
        out_channels = self.config.arch_specs.latent_dim
        
        print(f"使用文本控制模型:")
        print(f"  输入通道数: {in_channels}")
        print(f"  输出通道数: {out_channels}")
        print(f"  特征尺寸: {feature_size}")
        print(f"  Cross-attention维度: {self.cross_attention_dim}")
        print(f"  使用CLIP: {self.use_clip}")
        print(f"  控制强度: {self.control_strength}")
        
        # 创建文本控制扩散模型
        self.model = TextControlledDDIMModel(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            cross_attention_dim=self.cross_attention_dim,
            control_strength=self.control_strength,
            use_clip=self.use_clip,
            max_text_length=self.max_text_length,
            device=str(self.accelerator.device)
        )
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print(f"模型总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")

    def save_complete_checkpoint(self, epoch, global_step, avg_loss, filename="final.pth"):
        """保存完整的检查点"""
        os.makedirs(self.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.model.state_dict(),
            'avg_loss': avg_loss,
            'config': vars(self.args),
            'cross_attention_dim': self.cross_attention_dim,
            'use_clip': self.use_clip,
            'control_strength': self.control_strength,
            'max_text_length': self.max_text_length,
        }
        
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"模型已保存到: {save_path}")
        return save_path

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"警告: 检查点文件不存在: {checkpoint_path}")
            return 0, 0
        
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        
        # 加载模型状态
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        
        print(f"从epoch {start_epoch}, step {global_step} 继续训练")
        return start_epoch, global_step

    def train(self):
        """训练主循环"""
        print("开始训练...")
        
        # 准备数据集和模型
        self.prepare_dataset()
        self.prepare_model()
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.diff_specs.lr,
            weight_decay=1e-6
        )
        
        # 使用accelerator准备模型、优化器和数据加载器
        self.model.model, optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model.model, optimizer, self.train_dataloader
        )
        
        # 准备验证数据加载器
        self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        
        # 设置tensorboard
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # 早停机制
        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)
        
        # 加载检查点（如果有）
        start_epoch = 0
        global_step = 0
        if hasattr(self.args, 'resume') and self.args.resume:
            start_epoch, global_step = self.load_checkpoint(self.args.resume)
        
        num_epochs = self.config.diff_specs.num_epochs
        
        # 训练循环
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            epoch_loss = self.train_epoch(epoch, optimizer, global_step)
            
            # 验证
            val_loss = None
            validation_samples = None
            if (epoch + 1) % self.eval_interval == 0:
                val_loss, validation_samples = validate_model(
                    self.model, self.val_dataloader, 
                    self.accelerator.device, epoch, self.writer
                )
                save_validation_results(validation_samples, epoch, self.save_dir)
            
            # 记录到tensorboard
            self.writer.add_scalar('Train/EpochLoss', epoch_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar('Validation/EpochLoss', val_loss, epoch)
            
            # 保存最佳模型
            current_loss = val_loss if val_loss is not None else epoch_loss
            if current_loss < self.best_epoch_loss:
                self.best_epoch_loss = current_loss
                self.save_complete_checkpoint(epoch, global_step, current_loss, "best.pth")
                print(f"🎉 新的最佳模型! 损失: {current_loss:.6f}")
            
            # 定期保存检查点
            if (epoch + 1) % 50 == 0:
                self.save_complete_checkpoint(epoch, global_step, current_loss, f"checkpoint_epoch_{epoch+1}.pth")
            
            # 早停检查
            if early_stopping(val_loss or epoch_loss, self.model.model, epoch):
                print(f"早停触发，在epoch {epoch} 停止训练")
                break
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} 完成, 耗时: {epoch_time:.2f}s, 训练损失: {epoch_loss:.6f}")
            if val_loss is not None:
                print(f"验证损失: {val_loss:.6f}")
            
            # 清理内存
            cleanup_memory()
        
        # 保存最终模型
        self.save_complete_checkpoint(epoch, global_step, current_loss, "final.pth")
        
        # 关闭tensorboard
        self.writer.close()
        
        print("训练完成!")
        print(f"最佳验证损失: {self.best_epoch_loss:.6f}")
        print(f"模型保存目录: {self.save_dir}")

    def train_epoch(self, epoch, optimizer, global_step):
        """训练一个epoch"""
        self.model.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f'Epoch {epoch}', 
            total=num_batches,
            disable=not self.accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                features = batch['features']
                text_prompts = batch['text_prompt']
                
                # 训练步骤
                optimizer.zero_grad()
                
                # 使用模型的train_step方法
                with self.accelerator.accumulate(self.model.model):
                    loss = self.model.train_step(features, text_prompts, None)
                    
                    # 使用accelerator进行反向传播
                    self.accelerator.backward(loss)
                    
                    # 梯度裁剪
                    if hasattr(self.config.diff_specs, 'clip_grad_norm') and self.config.diff_specs.clip_grad_norm:
                        self.accelerator.clip_grad_norm_(self.model.model.parameters(), self.config.diff_specs.clip_grad_norm)
                    
                    optimizer.step()
                
                # 记录损失
                loss_value = loss.item()
                total_loss += loss_value
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.6f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.6f}'
                })
                
                # 记录到tensorboard
                if self.accelerator.is_main_process:
                    self.writer.add_scalar('Train/BatchLoss', loss_value, global_step)
                
                global_step += 1
                
                # 保存最佳step模型
                if loss_value < self.best_step_loss:
                    self.best_step_loss = loss_value
                    if self.accelerator.is_main_process:
                        self.save_complete_checkpoint(epoch, global_step, loss_value, "best_step.pth")
                
            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / num_batches
        return avg_loss


def main():
    parser = argparse.ArgumentParser(description='文本控制扩散模型训练')
    
    # 必要参数
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--tri_dir', type=str, required=True,
                        help='特征数据目录')
    parser.add_argument('--text_json_path', type=str, required=True,
                        help='文本描述JSON文件路径')
    
    # 可选参数
    parser.add_argument('--train_path', type=str, default=None,
                        help='训练数据路径（可覆盖配置文件）')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    # 文本控制参数
    parser.add_argument('--control_strength', type=float, default=1.0,
                        help='控制强度')
    parser.add_argument('--cross_attention_dim', type=int, default=1024,
                        help='Cross-attention维度')
    parser.add_argument('--use_clip', type=bool, default=True,
                        help='是否使用CLIP文本编码器')
    parser.add_argument('--max_text_length', type=int, default=77,
                        help='最大文本长度')
    parser.add_argument('--random_text', type=bool, default=True,
                        help='是否随机选择文本描述')
    
    # 训练控制参数
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=1e-6,
                        help='早停最小改善值')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='验证间隔（epoch）')
    
    args = parser.parse_args()
    
    # 检查必要文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    if not os.path.exists(args.tri_dir):
        print(f"错误: 特征目录不存在: {args.tri_dir}")
        return
    
    if not os.path.exists(args.text_json_path):
        print(f"错误: 文本描述文件不存在: {args.text_json_path}")
        return
    
    # 创建训练器并开始训练
    trainer = TextControlledDiffusionTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main() 