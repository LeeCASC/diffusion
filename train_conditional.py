#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
import time
from datetime import datetime
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator

# 导入我们创建的模块
from dataset_diff.conditional_dataset import ConditionalFeatureDataset, CATEGORY_NAMES
from class_conditional_unet import ClassConditionalDDIMModel
from config.config_parser import ConfigParser


def cleanup_memory():
    """清理GPU显存"""
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


def validate_model(model, val_loader, device, epoch, writer, num_classes=4):
    """验证模型并生成条件样本"""
    model.model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        # 计算验证损失
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # 创建优化器（仅用于计算损失）
            dummy_optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
            
            loss = model.train_step(features, labels, dummy_optimizer)
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            num_batches += 1
            
            if num_batches >= 10:  # 限制验证批次数
                break
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # 为每个类别生成样本
    validation_samples = {}
    metrics = {}
    
    for class_id in range(num_classes):
        try:
            # 生成该类别的样本
            class_labels = torch.full((4,), class_id, device=device, dtype=torch.long)
            generated_samples = model.generate(
                class_labels, 
                num_inference_steps=50, 
                guidance_scale=7.5
            )
            
            validation_samples[f'class_{class_id}'] = generated_samples.cpu()
            
            # 计算生成样本的统计指标
            sample_mean = generated_samples.mean().item()
            sample_std = generated_samples.std().item()
            sample_min = generated_samples.min().item()
            sample_max = generated_samples.max().item()
            
            # 计算样本间多样性（成对L2距离的平均值）
            diversity = 0.0
            if generated_samples.shape[0] > 1:
                for i in range(generated_samples.shape[0]):
                    for j in range(i+1, generated_samples.shape[0]):
                        diversity += torch.norm(generated_samples[i] - generated_samples[j], p=2).item()
                diversity /= (generated_samples.shape[0] * (generated_samples.shape[0] - 1) / 2)
            
            # 记录指标
            class_name = CATEGORY_NAMES.get(class_id, f"class_{class_id}")
            metrics[f'class_{class_id}'] = {
                'mean': sample_mean,
                'std': sample_std,
                'min': sample_min,
                'max': sample_max,
                'diversity': diversity,
                'name': class_name
            }
            
            # TensorBoard记录
            if writer:
                writer.add_scalar(f'Validation/Class_{class_id}_Mean', sample_mean, epoch)
                writer.add_scalar(f'Validation/Class_{class_id}_Std', sample_std, epoch)
                writer.add_scalar(f'Validation/Class_{class_id}_Diversity', diversity, epoch)
                
        except Exception as e:
            print(f"警告: 生成类别 {class_id} 样本时出错: {e}")
            continue
    
    # 记录总体验证损失
    if writer:
        writer.add_scalar('Validation/Loss', avg_loss, epoch)
    
    model.model.train()
    return avg_loss, validation_samples, metrics


def save_validation_results(validation_samples, metrics, epoch, save_dir):
    """保存验证结果"""
    # 保存生成的样本
    validation_path = save_dir / f"validation_epoch_{epoch}.pt"
    torch.save(validation_samples, validation_path)
    
    # 保存指标
    metrics_path = save_dir / f"validation_metrics_epoch_{epoch}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    return validation_path, metrics_path


class ConditionalDiffusionTrainer:
    def __init__(self, args):
        self.args = args
        self.config = self.load_config(args.config)
        
        # 创建基于时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(self.config.output.exp_name, f"cond_{timestamp}")
        self.save_dir = os.path.join(base_dir, "run")
        self.log_dir = os.path.join(base_dir, "logs")
        # 初始化 accelerator，不使用 logging_dir 参数
        self.accelerator = Accelerator()
        self.train_path = args.train_path if args.train_path else self.config.input.split_dir
        self.tri_dir = args.tri_dir
        self.num_classes = args.num_classes
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
        return config

    def prepare_dataset(self):
        dataset = ConditionalFeatureDataset(
            train_path=self.train_path,
            tri_dir=self.tri_dir,
            num_classes=self.num_classes
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
        self.model = ClassConditionalDDIMModel(num_classes=self.num_classes, device=self.accelerator.device)
        self.optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=self.config.diff_specs.lr,
            weight_decay=getattr(self.config.diff_specs, 'decay', 0.0)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.diff_specs.num_epochs)
        self.model.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(  # type: ignore
            self.model.model, self.optimizer, self.train_dataloader, self.scheduler  # type: ignore
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
            self.accelerator.init_trackers("conditional_diffusion_training", config={
                "lr": self.config.diff_specs.lr,
                "batch_size": self.config.diff_specs.batch_size,
                "num_epochs": self.config.diff_specs.num_epochs,
                "num_classes": self.num_classes
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
            'num_classes': self.num_classes,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'eval_interval': self.eval_interval,
            'eval_samples': self.eval_samples,
            
            # 时间戳
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # 随机状态（用于完全可重现的训练）
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
            
            # 恢复随机状态（可选）
            if 'random_state' in checkpoint:
                torch.set_rng_state(checkpoint['random_state'])
                print("已恢复PyTorch随机状态")
            if 'numpy_random_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_random_state'])
                print("已恢复NumPy随机状态")
            
            print(f"已恢复完整训练状态:")
            print(f"  保存时间: {checkpoint.get('save_time', '未知')}")
            print(f"  起始epoch: {start_epoch}")
            print(f"  全局step: {global_step}")
            print(f"  当前损失: {checkpoint.get('current_loss', '未知'):.6f}")
            print(f"  最佳step损失: {self.best_step_loss:.6f}")
            print(f"  最佳epoch损失: {self.best_epoch_loss:.6f}")
            print(f"  耐心计数: {self.patience_counter}/{self.patience}")
            print(f"  类别数量: {checkpoint.get('num_classes', self.num_classes)}")
            
            return {
                'start_epoch': start_epoch,
                'global_step': global_step,
                'is_complete_checkpoint': True
            }
        else:
            # 旧格式checkpoint（只有模型权重）
            print("检测到旧格式checkpoint，只恢复模型权重...")
            self.model.model.load_state_dict(checkpoint)
            print(f"已加载模型权重: {checkpoint_path}")
            return {
                'start_epoch': 0,
                'global_step': 0,
                'is_complete_checkpoint': False
            }

    def train(self):
        num_epochs = self.config.diff_specs.num_epochs
        start_epoch = 0
        global_step = 0
        
        # 如果有resume checkpoint，加载它
        if hasattr(self.args, 'resume') and self.args.resume:
            resume_info = self.load_checkpoint(self.args.resume)
            if resume_info:
                start_epoch = resume_info['start_epoch']
                global_step = resume_info['global_step']
        
        if self.accelerator.is_main_process:
            print(f"开始条件训练，设备: {self.accelerator.device}")
            print(f"总epoch数: {num_epochs}")
            print(f"起始epoch: {start_epoch + 1}")
            print(f"学习率: {self.config.diff_specs.lr}")
            print(f"类别数量: {self.num_classes}")
        
        for epoch in range(start_epoch, num_epochs):
            self.model.model.train()
            epoch_loss = 0.0
            num_batches = 0
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not self.accelerator.is_main_process)
            
            for batch_idx, batch in enumerate(pbar):
                features = batch['features'].to(self.accelerator.device)
                labels = batch['label'].to(self.accelerator.device)
                
                # 手动计算损失（类似train.py中的方式）
                batch_size = features.shape[0]
                noise = torch.randn_like(features)
                timesteps = torch.randint(0, 1000, (batch_size,), device=self.accelerator.device)
                timesteps = timesteps.long()
                
                noisy_features = self.model.train_scheduler.add_noise(features, noise, timesteps)  # type: ignore
                
                # 10% 概率进行无条件训练（Classifier-free guidance）
                # 改进：在样本级别而非batch级别进行无条件训练
                batch_size = labels.shape[0]
                
                # 方法1：随机mask，确保每个batch都有条件和无条件样本的混合
                uncond_mask = torch.rand(batch_size, device=labels.device) < 0.1
                condition_labels = labels.clone()
                
                # 如果整个batch都是条件样本，强制至少1个无条件
                if not uncond_mask.any() and batch_size > 1:
                    uncond_mask[torch.randint(0, batch_size, (1,))] = True
                
                # 如果整个batch都是无条件，强制至少保留1个条件样本
                if uncond_mask.all() and batch_size > 1:
                    uncond_mask[torch.randint(0, batch_size, (1,))] = False
                
                # 对无条件样本设置为None（但需要在模型中特殊处理）
                # 这里我们使用一个特殊标记值来表示无条件
                condition_labels[uncond_mask] = -1  # 使用-1表示无条件
                
                noise_pred = self.model.model(noisy_features, timesteps, condition_labels).sample
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
                        self.cleanup_memory()
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
            print("条件训练完成！")
            if self.writer:
                self.writer.close()
        
        # 训练结束后清理显存
        self.cleanup_memory()

    def validate_model(self, epoch):
        """
        验证模型：为每个类别生成样本并计算质量指标
        """
        if not self.accelerator.is_main_process:
            return
        
        print(f"\n=== Epoch {epoch+1} 条件验证开始 ===")
        self.model.model.eval()
        
        # 记录真实数据统计信息（按类别）
        real_features_stats = self.compute_real_data_stats_by_class()
        
        # 为每个类别生成样本
        validation_samples = {}
        all_metrics = {}
        
        for class_id in range(self.num_classes):
            class_name = CATEGORY_NAMES.get(class_id, f"class_{class_id}")
            print(f"为类别 {class_id} ({class_name}) 生成验证样本...")
            
            # 生成该类别的样本
            with torch.no_grad():
                class_labels = torch.full((self.eval_samples,), class_id, device=self.accelerator.device, dtype=torch.long)
                generated_features = self.model.generate(
                    class_labels=class_labels,
                    num_inference_steps=50,
                    guidance_scale=7.5
                )
            
            validation_samples[f'class_{class_id}'] = generated_features.cpu()
            
            # 计算生成数据统计信息
            generated_stats = self.compute_generated_stats(generated_features)
            
            # 计算验证指标
            if class_id in real_features_stats:
                metrics = self.compute_validation_metrics(real_features_stats[class_id], generated_stats, generated_features)
            else:
                # 如果没有该类别的真实数据，只计算生成数据的指标
                metrics = {
                    'generated_mean': generated_stats['mean'],
                    'generated_std': generated_stats['std'],
                    'diversity': self.compute_diversity(generated_features),
                    'feature_range': generated_stats['max'] - generated_stats['min']
                }
            
            all_metrics[f'class_{class_id}'] = metrics
            
            # 记录到tensorboard
            if self.writer:
                for metric_name, metric_value in metrics.items():
                    self.writer.add_scalar(f'Validation/Class_{class_id}_{metric_name}', metric_value, epoch)
            
            # 打印该类别的验证结果
            print(f"  {class_name} 验证指标:")
            for metric_name, metric_value in metrics.items():
                print(f"    {metric_name}: {metric_value:.6f}")
        
        # 保存验证样本
        val_save_path = os.path.join(self.save_dir, f"validation_epoch_{epoch+1}.pt")
        torch.save(validation_samples, val_save_path)
        print(f"验证样本已保存到: {val_save_path}")
        print(f"=== Epoch {epoch+1} 条件验证结束 ===\n")
        
        self.model.model.train()
        return all_metrics
    
    def compute_real_data_stats_by_class(self):
        """计算真实数据的统计信息（按类别分组）"""
        class_features = {i: [] for i in range(self.num_classes)}
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_dataloader):
                features = batch['features']
                labels = batch['label']
                
                for class_id in range(self.num_classes):
                    mask = labels == class_id
                    if mask.any():
                        class_features[class_id].append(features[mask])
                
                if i >= 10:  # 只计算前几个batch的统计信息
                    break
        
        # 计算每个类别的统计信息
        class_stats = {}
        for class_id in range(self.num_classes):
            if class_features[class_id]:
                cat_features = torch.cat(class_features[class_id], dim=0)
                class_stats[class_id] = {
                    'mean': cat_features.mean().item(),
                    'std': cat_features.std().item(),
                    'min': cat_features.min().item(),
                    'max': cat_features.max().item()
                }
        
        return class_stats
    
    def compute_generated_stats(self, generated_features):
        """计算生成数据的统计信息"""
        return {
            'mean': generated_features.mean().item(),
            'std': generated_features.std().item(),
            'min': generated_features.min().item(),
            'max': generated_features.max().item()
        }
    
    def compute_diversity(self, generated_features):
        """计算多样性指标（样本间的平均距离）"""
        if generated_features.shape[0] > 1:
            # 计算所有样本对之间的L2距离
            flat_features = generated_features.view(generated_features.shape[0], -1)
            pairwise_distances = torch.cdist(flat_features, flat_features, p=2)
            # 取上三角矩阵（排除对角线和重复）
            mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1).bool()
            diversity = pairwise_distances[mask].mean().item()
            return diversity
        else:
            return 0.0
    
    def compute_validation_metrics(self, real_stats, gen_stats, generated_features):
        """计算验证指标"""
        metrics = {}
        
        # 1. 统计分布差异
        metrics['mean_diff'] = abs(real_stats['mean'] - gen_stats['mean'])
        metrics['std_diff'] = abs(real_stats['std'] - gen_stats['std'])
        
        # 2. 生成质量指标
        metrics['generated_mean'] = gen_stats['mean']
        metrics['generated_std'] = gen_stats['std']
        
        # 3. 多样性指标
        metrics['diversity'] = self.compute_diversity(generated_features)
        
        # 4. 特征范围合理性
        metrics['feature_range'] = gen_stats['max'] - gen_stats['min']
        
        return metrics
    
    def cleanup_memory(self):
        """清理所有GPU设备的显存"""
        try:
            print("清理显存...")
            
            # 删除模型引用
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            
            # 删除优化器引用
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                del self.optimizer
            
            # 删除数据加载器引用
            if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                del self.train_dataloader
            
            # 删除scheduler引用
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                del self.scheduler
            
            # 关闭writer
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.close()
                del self.writer
            
            # 清理accelerator
            if hasattr(self, 'accelerator') and self.accelerator is not None:
                # 等待所有进程
                self.accelerator.wait_for_everyone()
                
                # 结束accelerator追踪
                try:
                    self.accelerator.end_training()
                except:
                    pass
            
            # 强制清理所有GPU显存
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                # 清理所有可见的GPU
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                print(f"已清理 {torch.cuda.device_count()} 个GPU设备的显存")
            
            print("显存清理完成")
            
        except Exception as e:
            print(f"显存清理时出现错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="训练条件DDIM扩散模型")
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径 (例如: config/exp-shapenet-car/specs_shapenet_all.json)")
    parser.add_argument("--train_path", type=str, default=None, help="训练数据路径（可选，默认使用配置文件中的路径）")
    parser.add_argument("--tri_dir", type=str, default=None, help="三平面数据目录（可选，默认使用配置文件中的路径）")
    parser.add_argument("--num_classes", type=int, default=4, help="类别数量")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint路径，继续训练")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值：多少个epoch损失不改善就停止训练")
    parser.add_argument("--min_delta", type=float, default=1e-6, help="早停最小改善阈值")
    parser.add_argument("--eval_interval", type=int, default=10, help="验证间隔：每多少epoch进行一次验证")
    parser.add_argument("--eval_samples", type=int, default=16, help="验证时生成的样本数量")
    args = parser.parse_args()

    trainer = ConditionalDiffusionTrainer(args)
    trainer.prepare_dataset()
    trainer.prepare_model()
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

if __name__ == "__main__":
    main() 