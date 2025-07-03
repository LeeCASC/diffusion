import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
import numpy as np
from typing import Optional, Tuple


class FeatureUNet(nn.Module):
    """
    专门用于特征生成的UNet模型
    输入输出尺寸: (batch_size, 256, 16, 48)
    """
    
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        
        # 使用diffusers的UNet2DModel作为基础
        self.unet = UNet2DModel(
            sample_size=(16, 48),  # 高度16，宽度48
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(256, 256, 512, 1024),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2D",
            norm_num_groups=32,
            add_attention=True,
        )
    
    def forward(self, x, timesteps, return_dict=True):
        return self.unet(x, timesteps, return_dict=return_dict)


class DDIMDiffusionModel:
    """
    DDIM扩散模型训练和推理类
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = FeatureUNet().to(device)
        
        # 初始化DDIM调度器
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        
        # 初始化DDPM调度器用于训练
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
    
    def train_step(self, clean_features, optimizer):
        """
        单步训练
        Args:
            clean_features: 干净的特征张量 (batch_size, 256, 16, 48)
            optimizer: 优化器
        """
        batch_size = clean_features.shape[0]
        
        # 添加噪声
        noise = torch.randn_like(clean_features)
        timesteps = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,), device=self.device)
        timesteps = timesteps.long()
        
        noisy_features = self.train_scheduler.add_noise(clean_features, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.model(noisy_features, timesteps).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, batch_size=1, num_inference_steps=50, eta=0.0):
        """
        使用DDIM采样生成特征
        Args:
            batch_size: 批量大小
            num_inference_steps: 推理步数
            eta: DDIM的eta参数，0为确定性采样
        Returns:
            生成的特征张量 (batch_size, 256, 16, 48)
        """
        # 设置DDIM调度器参数
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # 初始化随机噪声
        x = torch.randn(
            (batch_size, 256, 16, 48),
            device=self.device,
            dtype=torch.float32
        )
        
        # DDIM采样循环
        for i, t in enumerate(self.noise_scheduler.timesteps):
            # 预测噪声
            noise_pred = self.model(x, t).sample
            
            # DDIM步骤
            x = self.noise_scheduler.step(
                noise_pred,
                t,
                x,
                eta=eta
            ).prev_sample
        
        return x
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'noise_scheduler': self.noise_scheduler,
            'train_scheduler': self.train_scheduler,
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.noise_scheduler = checkpoint['noise_scheduler']
        self.train_scheduler = checkpoint['train_scheduler'] 