#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图控UNet模型 V2 - 基于Cross-Attention的改进版本
使用UNet2DConditionalModel和cross-attention机制实现图像控制
支持两种训练模式：原始图像模式 和 预提取特征模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor


class ImageEncoder(nn.Module):
    """
    图像编码器 - 将控制图像编码为序列特征用于cross-attention
    """
    
    def __init__(self, 
                 input_channels=3,      # RGB图像
                 output_dim=1024,       # 输出特征维度（匹配cross-attention）
                 image_size=256,        # 输入图像尺寸
                 patch_size=16,         # patch大小
                 use_clip=True):        # 是否使用CLIP编码器
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_clip = use_clip
        
        if use_clip:
            # 使用预训练的CLIP视觉编码器
            self.clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
            # 投影层：CLIP输出维度 -> 目标维度
            clip_dim = self.clip_encoder.config.hidden_size
            self.projection = nn.Linear(clip_dim, output_dim)
            
            # 冻结CLIP参数（可选）
            for param in self.clip_encoder.parameters():
                param.requires_grad = False
        else:
            # 自定义vision transformer编码器
            num_patches = (image_size // patch_size) ** 2
            patch_dim = input_channels * patch_size * patch_size
            
            self.patch_embedding = nn.Linear(patch_dim, output_dim)
            self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, output_dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
            
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=8,
                dim_feedforward=output_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.num_patches = (image_size // patch_size) ** 2
    
    def forward(self, images):
        """
        前向传播
        Args:
            images: 输入图像 (batch_size, channels, height, width)
        Returns:
            features: 图像特征序列 (batch_size, seq_len, output_dim)
        """
        batch_size = images.shape[0]
        
        if self.use_clip:
            # 使用CLIP编码器
            with torch.no_grad():
                clip_features = self.clip_encoder(images).last_hidden_state  # (B, seq_len, clip_dim)
            
            # 投影到目标维度
            features = self.projection(clip_features)  # (B, seq_len, output_dim)
            
        else:
            # 自定义编码器
            # 分割为patches
            patches = self._image_to_patches(images)  # (B, num_patches, patch_dim)
            
            # Patch embedding
            patch_embeddings = self.patch_embedding(patches)  # (B, num_patches, output_dim)
            
            # 添加CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (B, num_patches+1, output_dim)
            
            # 添加位置编码
            embeddings = embeddings + self.position_embedding
            
            # Transformer编码
            features = self.transformer(embeddings)  # (B, num_patches+1, output_dim)
        
        return features
    
    def _image_to_patches(self, images):
        """将图像转换为patches"""
        batch_size, channels, height, width = images.shape
        patch_size = self.patch_size
        
        # 重塑为patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(batch_size, -1, channels * patch_size * patch_size)
        
        return patches





class CrossAttentionImageControlledUNet(nn.Module):
    """
    基于Cross-Attention的图控UNet模型
    使用UNet2DConditionalModel和cross-attention机制
    """
    
    def __init__(self,
                 in_channels=256,           # 输入特征通道数
                 out_channels=256,          # 输出特征通道数
                 image_channels=3,          # 控制图像通道数
                 image_size=256,            # 控制图像尺寸
                 feature_size=(16, 48),     # 特征空间尺寸
                 cross_attention_dim=1024,  # Cross-attention维度（匹配图控特征维度）
                 control_strength=1.0,      # 控制强度
                 use_image_encoder=True,    # 是否使用图像编码器
                 use_clip=True):            # 是否使用CLIP编码器
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim
        self.control_strength = control_strength
        self.use_image_encoder = use_image_encoder
        
        # 图像编码器（用于原始图像输入）
        if use_image_encoder:
            self.image_encoder = ImageEncoder(
                input_channels=image_channels,
                output_dim=cross_attention_dim,
                image_size=image_size,
                use_clip=use_clip
            )
        else:
            self.image_encoder = None
        
        # 图控特征直接使用，不需要投影器（因为已经是序列格式 (257, 1024)）
        
        # UNet2DConditionalModel
        self.unet = UNet2DConditionModel(
            sample_size=feature_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(256, 256, 512, 1024),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            mid_block_type="UNetMidBlock2DCrossAttn",
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=32,
            use_linear_projection=True,
            attention_head_dim=8,
        )
    
    def forward(self, x, timesteps, control_data, control_type=None, return_dict=True):
        """
        前向传播
        Args:
            x: 输入特征 (batch_size, in_channels, h, w)
            timesteps: 时间步 (batch_size,)
            control_data: 控制数据，可能是图像或特征
            control_type: 控制类型 'raw_image' 或 'image_features'
            return_dict: 是否返回字典格式
        """
        batch_size = x.shape[0]
        
        # 根据控制类型处理控制数据
        if control_type == 'raw_image' or (control_type is None and self.image_encoder is not None):
            # 原始图像模式：通过图像编码器编码
            if self.image_encoder is None:
                raise ValueError("模型未配置图像编码器，但收到原始图像输入")
            encoder_hidden_states = self.image_encoder(control_data)  # (B, seq_len, cross_attention_dim)
        elif control_type == 'image_features' or control_type is None:
            # 特征模式：直接使用图控特征（已经是序列格式 (B, 257, 1024)）
            encoder_hidden_states = control_data  # (B, 257, 1024)
        else:
            raise ValueError(f"未知的控制类型: {control_type}")
        
        # 应用控制强度
        if encoder_hidden_states.dim() > 3:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
        encoder_hidden_states = encoder_hidden_states * self.control_strength
        
        # 确保timesteps是正确的格式
        if timesteps.dim() > 1:
            timesteps = timesteps.squeeze()
        
        # UNet前向传播（带cross-attention）
        return self.unet(
            x, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict
        )


class CrossAttentionImageControlledDDIMModel:
    """基于Cross-Attention的图控DDIM扩散模型"""
    
    def __init__(self, 
                 in_channels=256,
                 out_channels=256,
                 image_size=256,
                 feature_size=(16, 48),
                 cross_attention_dim=1024,
                 control_strength=1.0,
                 training_mode='auto',  # 'raw_image', 'image_features', 'auto'
                 use_clip=True,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.training_mode = training_mode
        self.cross_attention_dim = cross_attention_dim
        
        # 根据训练模式决定是否需要图像编码器
        use_image_encoder = training_mode in ['raw_image', 'auto']
        
        # 初始化模型
        self.model = CrossAttentionImageControlledUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            cross_attention_dim=cross_attention_dim,
            control_strength=control_strength,
            use_image_encoder=use_image_encoder,
            use_clip=use_clip
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
    
    def train_step(self, clean_features, control_data, control_type, optimizer):
        """
        训练步骤
        Args:
            clean_features: 干净特征 (batch_size, in_channels, h, w)
            control_data: 控制数据（图像或特征）
            control_type: 控制类型 'raw_image' 或 'image_features'
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
        
        # 预测噪声
        noise_pred = self.model(noisy_features, timesteps, control_data, control_type).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def generate(self, control_data, control_type=None, num_inference_steps=50, guidance_scale=1.0):
        """
        图控生成
        Args:
            control_data: 控制数据（图像或特征）
            control_type: 控制类型 'raw_image' 或 'image_features'（可选，模型会自动推断）
            num_inference_steps: 推理步数
            guidance_scale: 引导强度（暂时不用）
        """
        if isinstance(control_data, torch.Tensor):
            batch_size = control_data.shape[0]
            device = control_data.device
        else:
            raise ValueError("control_data必须是torch.Tensor")
        
        # 自动推断控制类型（如果未指定）
        if control_type is None:
            if self.training_mode == 'raw_image':
                control_type = 'raw_image'
            elif self.training_mode == 'image_features':
                control_type = 'image_features'
            else:
                # 根据数据形状推断
                if control_data.dim() == 4 and control_data.shape[1] == 3:
                    control_type = 'raw_image'
                else:
                    control_type = 'image_features'
        
        # 初始化随机噪声
        features = torch.randn(batch_size, self.in_channels, *self.feature_size, device=device)
        
        # 设置推理时间步
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps
        
        # 去噪过程
        for t in timesteps:
            # 确保timestep是正确的格式
            t_batch = t.unsqueeze(0).expand(batch_size)
            
            # 预测噪声
            with torch.no_grad():
                noise_pred = self.model(features, t_batch, control_data, control_type).sample
            
            # 更新特征
            features = self.noise_scheduler.step(noise_pred, t, features).prev_sample
        
        return features
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'feature_size': self.feature_size,
            'cross_attention_dim': self.cross_attention_dim,
            'training_mode': self.training_mode,
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载图控模型 V2: {path}")
        print(f"训练模式: {checkpoint.get('training_mode', 'unknown')}")
        print(f"Cross-attention维度: {checkpoint.get('cross_attention_dim', 'unknown')}")


if __name__ == "__main__":
    # 测试基于Cross-Attention的图控UNet模型
    print("测试基于Cross-Attention的图控UNet模型...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 测试原始图像模式
    print("\n=== 测试原始图像模式 ===")
    model_raw = CrossAttentionImageControlledDDIMModel(
        in_channels=256,
        out_channels=256,
        image_size=256,
        feature_size=(16, 48),
        cross_attention_dim=768,
        training_mode='raw_image',
        use_clip=False,  # 先不用CLIP测试
        device=device
    )
    
    # 创建测试数据
    batch_size = 2
    features = torch.randn(batch_size, 256, 16, 48, device=device)
    control_images = torch.randn(batch_size, 3, 256, 256, device=device)  # 原始图像
    timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
    
    print(f"输入特征形状: {features.shape}")
    print(f"控制图像形状: {control_images.shape}")
    
    try:
        # 测试前向传播
        with torch.no_grad():
            output = model_raw.model(features, timesteps, control_images, 'raw_image')
            print(f"✅ 原始图像模式前向传播成功！输出形状: {output.sample.shape}")
        
        # 测试生成
        with torch.no_grad():
            generated = model_raw.generate(control_images, 'raw_image', num_inference_steps=10)
            print(f"✅ 原始图像模式生成成功！生成特征形状: {generated.shape}")
        
    except Exception as e:
        print(f"❌ 原始图像模式测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试图像特征模式
    print("\n=== 测试图像特征模式 ===")
    model_features = CrossAttentionImageControlledDDIMModel(
        in_channels=256,
        out_channels=256,
        cross_attention_dim=768,
        training_mode='image_features',
        device=device
    )
    
    # 创建测试数据
    control_features = torch.randn(batch_size, 256, 16, 48, device=device)  # 预提取特征
    
    print(f"控制特征形状: {control_features.shape}")
    
    try:
        # 测试前向传播
        with torch.no_grad():
            output = model_features.model(features, timesteps, control_features, 'image_features')
            print(f"✅ 图像特征模式前向传播成功！输出形状: {output.sample.shape}")
        
        # 测试生成
        with torch.no_grad():
            generated = model_features.generate(control_features, 'image_features', num_inference_steps=10)
            print(f"✅ 图像特征模式生成成功！生成特征形状: {generated.shape}")
        
    except Exception as e:
        print(f"❌ 图像特征模式测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 基于Cross-Attention的图控UNet模型测试完成！") 