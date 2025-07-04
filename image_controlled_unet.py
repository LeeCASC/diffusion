#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图控UNet模型 - 支持2D图像控制的扩散模型
结合图像编码器和UNet进行图控生成
支持两种训练模式：原始图像模式 和 预提取特征模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class ImageEncoder(nn.Module):
    """
    图像编码器
    将控制图像编码为特征图
    """
    
    def __init__(self, 
                 input_channels=3,      # RGB图像
                 output_channels=256,   # 输出特征通道数
                 image_size=256,        # 输入图像尺寸
                 feature_size=(16, 48)): # 输出特征空间尺寸
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.image_size = image_size
        self.feature_size = feature_size
        
        # 图像编码器网络 - 使用ResNet-like结构
        self.encoder = nn.Sequential(
            # 第一阶段：256 -> 128
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 256 -> 64
            
            # 第二阶段：128 -> 64
            self._make_layer(64, 128, 2, stride=2),   # 64 -> 32
            
            # 第三阶段：64 -> 32
            self._make_layer(128, 256, 2, stride=2),  # 32 -> 16
            
            # 第四阶段：32 -> 16
            self._make_layer(256, 512, 2, stride=2),  # 16 -> 8
        )
        
        # 自适应池化到目标尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d(feature_size)
        
        # 特征映射到目标通道数
        self.feature_projection = nn.Sequential(
            nn.Conv2d(512, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # 位置编码（可选）
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoding = self._create_position_encoding(output_channels, feature_size)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建ResNet风格的层"""
        layers = []
        
        # 第一个block可能有stride
        layers.append(self._basic_block(in_channels, out_channels, stride))
        
        # 其余blocks
        for _ in range(1, blocks):
            layers.append(self._basic_block(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _basic_block(self, in_channels, out_channels, stride):
        """基础ResNet块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _create_position_encoding(self, channels, size):
        """创建位置编码"""
        h, w = size
        pos_h = torch.arange(h).float().unsqueeze(1).repeat(1, w)
        pos_w = torch.arange(w).float().unsqueeze(0).repeat(h, 1)
        
        # 归一化到[-1, 1]
        pos_h = (pos_h / (h - 1)) * 2 - 1
        pos_w = (pos_w / (w - 1)) * 2 - 1
        
        # 创建正弦位置编码
        pe = torch.zeros(channels, h, w)
        for i in range(0, channels, 4):
            if i < channels:
                pe[i] = torch.sin(pos_h * (10000 ** (i / channels)))
            if i + 1 < channels:
                pe[i + 1] = torch.cos(pos_h * (10000 ** (i / channels)))
            if i + 2 < channels:
                pe[i + 2] = torch.sin(pos_w * (10000 ** (i / channels)))
            if i + 3 < channels:
                pe[i + 3] = torch.cos(pos_w * (10000 ** (i / channels)))
        
        return nn.Parameter(pe, requires_grad=False)
    
    def forward(self, images):
        """
        前向传播
        Args:
            images: 控制图像 (batch_size, 3, image_size, image_size)
        Returns:
            image_features: 图像特征 (batch_size, output_channels, feature_h, feature_w)
        """
        batch_size = images.shape[0]
        
        # 图像编码
        features = self.encoder(images)  # (B, 512, H', W')
        
        # 自适应池化到目标尺寸
        features = self.adaptive_pool(features)  # (B, 512, feature_h, feature_w)
        
        # 特征投影
        features = self.feature_projection(features)  # (B, output_channels, feature_h, feature_w)
        
        # 添加位置编码
        if self.use_pos_encoding:
            features = features + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return features


class DualModeImageControlledUNet(nn.Module):
    """
    双模式图控UNet模型
    支持两种训练模式：
    1. raw_image模式：输入原始图像，通过内置编码器编码
    2. image_features模式：输入预提取的图像特征，直接使用
    """
    
    def __init__(self,
                 in_channels=256,       # 输入特征通道数
                 out_channels=256,      # 输出特征通道数
                 image_channels=3,      # 控制图像通道数
                 image_size=256,        # 控制图像尺寸
                 feature_size=(16, 48), # 特征空间尺寸
                 control_strength=1.0,  # 控制强度
                 use_image_encoder=True): # 是否使用图像编码器
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.control_strength = control_strength
        self.use_image_encoder = use_image_encoder
        
        # 图像编码器（仅在需要时创建）
        if use_image_encoder:
            self.image_encoder = ImageEncoder(
                input_channels=image_channels,
                output_channels=in_channels,
                image_size=image_size,
                feature_size=feature_size
            )
        else:
            self.image_encoder = None
        
        # 控制特征融合网络
        self.control_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),  # 1x1卷积
        )
        
        # UNet模型
        self.unet = UNet2DModel(
            sample_size=feature_size,
            in_channels=in_channels * 2,  # 输入 + 控制特征
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D", 
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2D",
            norm_num_groups=32,
            add_attention=True,
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
            # 原始图像模式：需要编码
            if self.image_encoder is None:
                raise ValueError("模型未配置图像编码器，但收到原始图像输入")
            control_features = self.image_encoder(control_data)
        elif control_type == 'image_features' or control_type is None:
            # 特征模式：直接使用
            control_features = control_data
        else:
            raise ValueError(f"未知的控制类型: {control_type}")
        
        # 融合控制特征
        control_features = self.control_fusion(control_features)  # (B, in_channels, h, w)
        
        # 应用控制强度
        control_features = control_features * self.control_strength
        
        # 拼接输入特征和控制特征
        x_with_control = torch.cat([x, control_features], dim=1)  # (B, 2*in_channels, h, w)
        
        # 确保timesteps是正确的格式
        if timesteps.dim() > 1:
            timesteps = timesteps.squeeze()
        
        # UNet前向传播
        return self.unet(x_with_control, timesteps, return_dict=return_dict)


class DualModeImageControlledDDIMModel:
    """双模式图控DDIM扩散模型"""
    
    def __init__(self, 
                 in_channels=256,
                 out_channels=256,
                 image_size=256,
                 feature_size=(16, 48),
                 control_strength=1.0,
                 training_mode='auto',  # 'raw_image', 'image_features', 'auto'
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.training_mode = training_mode
        
        # 根据训练模式决定是否需要图像编码器
        use_image_encoder = training_mode in ['raw_image', 'auto']
        
        # 初始化模型
        self.model = DualModeImageControlledUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            control_strength=control_strength,
            use_image_encoder=use_image_encoder
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
                # 原始方式：硬拼接
                x_with_control = torch.cat([features, control_features], dim=1)
                # 新方式：Cross-Attention
                encoder_hidden_states = self.model.image_encoder(control_data)
                output = self.model.unet(features, t_batch, encoder_hidden_states=encoder_hidden_states)
            
            # 更新特征
            features = self.noise_scheduler.step(output, t, features).prev_sample
        
        return features
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'feature_size': self.feature_size,
            'training_mode': self.training_mode,
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载图控模型: {path}")
        print(f"训练模式: {checkpoint.get('training_mode', 'unknown')}")


# 为了向后兼容，保留原始类名
ImageControlledUNet = DualModeImageControlledUNet
ImageControlledDDIMModel = DualModeImageControlledDDIMModel


if __name__ == "__main__":
    # 测试双模式图控UNet模型
    print("测试双模式图控UNet模型...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 测试原始图像模式
    print("\n=== 测试原始图像模式 ===")
    model_raw = DualModeImageControlledDDIMModel(
        in_channels=256,
        out_channels=256,
        image_size=256,
        feature_size=(16, 48),
        training_mode='raw_image',
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
    model_features = DualModeImageControlledDDIMModel(
        in_channels=256,
        out_channels=256,
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
    
    print("\n✅ 双模式图控UNet模型测试完成！") 