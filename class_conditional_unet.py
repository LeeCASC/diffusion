import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import numpy as np
from typing import Optional, Tuple


class ClassEmbedding(nn.Module):
    """类别嵌入层"""
    
    def __init__(self, num_classes=4, embed_dim=256, max_seq_length=16*48):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, embed_dim)
        
        # 位置嵌入（用于空间条件）
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, class_labels, height=16, width=48):
        """
        Args:
            class_labels: 类别标签 (batch_size,)
            height: 特征图高度
            width: 特征图宽度
        Returns:
            class_condition: 类别条件特征 (batch_size, embed_dim, height, width)
        """
        batch_size = class_labels.shape[0]
        device = class_labels.device
        
        # 获取类别嵌入
        class_embed = self.class_embedding(class_labels)  # (batch_size, embed_dim)
        
        # 创建位置索引
        positions = torch.arange(height * width, device=device)
        pos_embed = self.position_embedding(positions)  # (H*W, embed_dim)
        
        # 广播类别嵌入到所有位置
        class_embed = class_embed.unsqueeze(1)  # (batch_size, 1, embed_dim)
        class_embed = class_embed.expand(batch_size, height * width, self.embed_dim)  # (batch_size, H*W, embed_dim)
        
        # 添加位置嵌入
        combined_embed = class_embed + pos_embed.unsqueeze(0)  # (batch_size, H*W, embed_dim)
        
        # 投影
        condition_embed = self.projection(combined_embed)  # (batch_size, H*W, embed_dim)
        
        # 重塑为特征图
        condition_embed = condition_embed.view(batch_size, height, width, self.embed_dim)
        condition_embed = condition_embed.permute(0, 3, 1, 2)  # (batch_size, embed_dim, height, width)
        
        return condition_embed


class ClassConditionalUNet(nn.Module):
    """支持类别条件的UNet模型"""
    
    def __init__(self, 
                 in_channels=256, 
                 out_channels=256, 
                 num_classes=4,
                 condition_dim=128):
        super().__init__()
        
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        
        # 类别嵌入层
        self.class_embedding = ClassEmbedding(
            num_classes=num_classes, 
            embed_dim=condition_dim,
            max_seq_length=16*48
        )
        
        # 条件融合网络
        self.condition_fusion = nn.Sequential(
            nn.Conv2d(condition_dim, condition_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(condition_dim // 2, condition_dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(condition_dim // 4, in_channels, kernel_size=1),
        )
        
        # 基础UNet（输入通道翻倍以容纳条件）
        self.unet = UNet2DModel(
            sample_size=(16, 48),
            in_channels=in_channels * 2,  # 原始输入 + 条件特征
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
            class_embed_type="timestep",  # 启用类别嵌入
            num_class_embeds=num_classes,  # 简化：直接使用num_classes
        )
        
        # 注意：我们使用简化的条件方法，直接通过通道拼接和UNet内置的class_labels
    
    def forward(self, x, timesteps, class_labels=None, return_dict=True):
        """
        Args:
            x: 输入特征 (batch_size, 256, 16, 48)
            timesteps: 时间步 (batch_size,)
            class_labels: 类别标签 (batch_size,) 或 None
                         可包含-1表示无条件样本
            return_dict: 是否返回字典格式
        """
        batch_size, channels, height, width = x.shape
        
        if class_labels is not None:
            # 处理混合条件：有些样本有条件(-1表示无条件)
            uncond_mask = (class_labels == -1)
            cond_mask = ~uncond_mask
            
            # 初始化条件特征
            condition_features = torch.zeros(batch_size, channels, height, width, device=x.device)
            
            # 为有条件的样本生成条件特征
            if cond_mask.any():
                cond_labels = class_labels[cond_mask]
                cond_class_condition = self.class_embedding(cond_labels, height, width)
                cond_condition_features = self.condition_fusion(cond_class_condition)
                condition_features[cond_mask] = cond_condition_features
            
            # 无条件样本保持零条件特征
            # condition_features[uncond_mask] 已经是零
            
            # 拼接原始输入和条件特征
            x_with_condition = torch.cat([x, condition_features], dim=1)
            
            # 为UNet准备class_labels（将-1替换为有效标签）
            unet_class_labels = class_labels.clone()
            unet_class_labels[uncond_mask] = 0  # 无条件样本使用类别0作为占位
            
        else:
            # 完全无条件生成，使用零条件
            zero_condition = torch.zeros(batch_size, channels, height, width, device=x.device)
            x_with_condition = torch.cat([x, zero_condition], dim=1)
            unet_class_labels = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # 确保timesteps是正确的格式（1D张量）
        if timesteps.dim() > 1:
            timesteps = timesteps.squeeze()
        
        # UNet前向传播
        return self.unet(
            x_with_condition, 
            timesteps, 
            class_labels=unet_class_labels,
            return_dict=return_dict
        )


class ClassConditionalDDIMModel:
    """类别条件DDIM扩散模型"""
    
    def __init__(self, num_classes=4, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.num_classes = num_classes
        
        # 初始化模型
        self.model = ClassConditionalUNet(
            in_channels=256,
            out_channels=256,
            num_classes=num_classes,
            condition_dim=128
        ).to(device)
        
        # 初始化调度器
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
    
    def train_step(self, clean_features, class_labels, optimizer):
        """
        条件训练步骤
        Args:
            clean_features: 干净特征 (batch_size, 256, 16, 48)
            class_labels: 类别标签 (batch_size,)
            optimizer: 优化器
        """
        batch_size = clean_features.shape[0]
        
        # 添加噪声
        noise = torch.randn_like(clean_features)
        timesteps = torch.randint(
            0, self.train_scheduler.num_train_timesteps, 
            (batch_size,), device=self.device, dtype=torch.long
        )
        
        noisy_features = self.train_scheduler.add_noise(clean_features, noise, timesteps)
        
        # 10% 概率进行无条件训练（Classifier-free guidance）
        if torch.rand(1) < 0.1:
            class_labels = None
        
        # 预测噪声
        noise_pred = self.model(noisy_features, timesteps, class_labels).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def generate(self, class_labels, num_inference_steps=50, guidance_scale=7.5):
        """
        条件生成
        Args:
            class_labels: 类别标签 (batch_size,)
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
        """
        batch_size = class_labels.shape[0]
        device = class_labels.device
        
        # 初始化随机噪声
        features = torch.randn(batch_size, 256, 16, 48, device=device)
        
        # 设置推理时间步
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps
        
        # 去噪过程
        for t in timesteps:
            # 确保timestep是正确的格式
            t_batch = t.unsqueeze(0).expand(batch_size)
            
            # 条件预测
            noise_pred_cond = self.model(
                features, 
                t_batch, 
                class_labels
            ).sample
            
            # 无条件预测（用于classifier-free guidance）
            noise_pred_uncond = self.model(
                features,
                t_batch,
                None
            ).sample
            
            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 更新特征
            features = self.noise_scheduler.step(noise_pred, t, features).prev_sample
        
        return features
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {path}")


if __name__ == "__main__":
    # 测试模型
    print("测试类别条件UNet模型...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClassConditionalDDIMModel(num_classes=4, device=device)
    
    # 创建测试数据
    batch_size = 2
    features = torch.randn(batch_size, 256, 16, 48, device=device)
    class_labels = torch.randint(0, 4, (batch_size,), device=device)
    
    print(f"输入特征形状: {features.shape}")
    print(f"类别标签: {class_labels}")
    
    # 测试前向传播
    with torch.no_grad():
        output = model.model(features, torch.randint(0, 1000, (batch_size,), device=device), class_labels)
        print(f"输出形状: {output.sample.shape}")
    
    # 测试生成
    print("\n测试条件生成...")
    with torch.no_grad():
        generated = model.generate(class_labels, num_inference_steps=10)
        print(f"生成特征形状: {generated.shape}")
    
    print("✅ 模型测试完成！") 