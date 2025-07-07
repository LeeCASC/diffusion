#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本控制UNet模型 - 基于Cross-Attention的文本控制扩散模型
使用UNet2DConditionalModel和cross-attention机制实现文本控制
支持CLIP文本编码器或简单的文本嵌入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class TextEncoder(nn.Module):
    """
    文本编码器 - 将文本prompt编码为序列特征用于cross-attention
    """
    
    def __init__(self, 
                 output_dim=1024,       # 输出特征维度（匹配cross-attention）
                 max_length=77,         # 最大文本长度
                 use_clip=True):        # 是否使用CLIP文本编码器
        super().__init__()
        
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_clip = use_clip
        
        if use_clip:
            # 使用预训练的CLIP文本编码器
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
            
            # 投影层：CLIP输出维度 -> 目标维度
            clip_dim = self.text_encoder.config.hidden_size
            self.projection = nn.Linear(clip_dim, output_dim)
            
            # 冻结CLIP参数（可选）
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            # 简单的文本嵌入编码器
            vocab_size = 50000  # 假设词汇表大小
            embed_dim = 512
            
            self.tokenizer = None  # 需要外部提供分词器
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
            
            # 投影到目标维度
            self.projection = nn.Linear(embed_dim, output_dim)
    
    def forward(self, text_prompts):
        """
        前向传播
        Args:
            text_prompts: 文本prompt列表 (batch_size,) 或 已编码的token (batch_size, seq_len)
        Returns:
            features: 文本特征序列 (batch_size, seq_len, output_dim)
        """
        if self.use_clip:
            # 使用CLIP编码器
            if isinstance(text_prompts, (list, tuple)) and isinstance(text_prompts[0], str):
                # 文本字符串列表
                inputs = self.tokenizer(
                    text_prompts, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(next(self.text_encoder.parameters()).device) for k, v in inputs.items()}
            else:
                # 已编码的tokens
                inputs = {"input_ids": text_prompts}
            
            with torch.no_grad():
                clip_features = self.text_encoder(**inputs).last_hidden_state  # (B, seq_len, clip_dim)
            
            # 投影到目标维度
            features = self.projection(clip_features)  # (B, seq_len, output_dim)
            
        else:
            # 简单编码器
            if isinstance(text_prompts, (list, tuple)) and isinstance(text_prompts[0], str):
                raise NotImplementedError("简单编码器需要外部提供分词功能")
            
            # 假设已经是token ids
            embeddings = self.embedding(text_prompts)  # (B, seq_len, embed_dim)
            
            # Transformer编码
            transformer_out = self.transformer(embeddings)  # (B, seq_len, embed_dim)
            
            # 投影到目标维度
            features = self.projection(transformer_out)  # (B, seq_len, output_dim)
        
        return features


class TextControlledUNet(nn.Module):
    """
    基于Cross-Attention的文本控制UNet模型
    使用UNet2DConditionalModel和cross-attention机制
    """
    
    def __init__(self,
                 in_channels=256,           # 输入特征通道数
                 out_channels=256,          # 输出特征通道数
                 feature_size=(16, 48),     # 特征空间尺寸
                 cross_attention_dim=1024,  # Cross-attention维度
                 control_strength=1.0,      # 控制强度
                 use_clip=True,             # 是否使用CLIP编码器
                 max_text_length=77):       # 最大文本长度
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim
        self.control_strength = control_strength
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            output_dim=cross_attention_dim,
            max_length=max_text_length,
            use_clip=use_clip
        )
        
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
    
    def forward(self, x, timesteps, text_prompts, return_dict=True):
        """
        前向传播
        Args:
            x: 输入特征 (batch_size, in_channels, h, w)
            timesteps: 时间步 (batch_size,)
            text_prompts: 文本prompt列表或已编码的tokens
            return_dict: 是否返回字典格式
        """
        batch_size = x.shape[0]
        
        # 编码文本prompt
        encoder_hidden_states = self.text_encoder(text_prompts)  # (B, seq_len, cross_attention_dim)
        
        # 应用控制强度
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


class TextControlledDDIMModel:
    """基于Cross-Attention的文本控制DDIM扩散模型"""
    
    def __init__(self, 
                 in_channels=256,
                 out_channels=256,
                 feature_size=(16, 48),
                 cross_attention_dim=1024,
                 control_strength=1.0,
                 use_clip=True,
                 max_text_length=77,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.cross_attention_dim = cross_attention_dim
        
        # 初始化模型
        self.model = TextControlledUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            cross_attention_dim=cross_attention_dim,
            control_strength=control_strength,
            use_clip=use_clip,
            max_text_length=max_text_length
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
    
    def train_step(self, clean_features, text_prompts, optimizer):
        """
        训练步骤
        Args:
            clean_features: 干净特征 (batch_size, in_channels, h, w)
            text_prompts: 文本prompt列表
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
        noise_pred = self.model(noisy_features, timesteps, text_prompts).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def generate(self, text_prompts, num_inference_steps=50, guidance_scale=7.5):
        """
        文本控制生成
        Args:
            text_prompts: 文本prompt列表
            num_inference_steps: 推理步数
            guidance_scale: 引导强度（用于classifier-free guidance）
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        batch_size = len(text_prompts)
        
        # 初始化随机噪声
        features = torch.randn(batch_size, self.in_channels, *self.feature_size, device=self.device)
        
        # 设置推理时间步
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        
        # 去噪过程
        for t in timesteps:
            # 确保timestep是正确的格式
            t_batch = t.unsqueeze(0).expand(batch_size)
            
            # 预测噪声
            with torch.no_grad():
                noise_pred = self.model(features, t_batch, text_prompts).sample
            
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
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载文本控制模型: {path}")
        print(f"Cross-attention维度: {checkpoint.get('cross_attention_dim', 'unknown')}")


if __name__ == "__main__":
    # 测试基于Cross-Attention的文本控制UNet模型
    print("测试基于Cross-Attention的文本控制UNet模型...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 测试文本控制模型
        print("\n=== 测试文本控制模型 ===")
        model = TextControlledDDIMModel(
            in_channels=256,
            out_channels=256,
            feature_size=(16, 48),
            cross_attention_dim=1024,
            control_strength=1.0,
            use_clip=False,  # 先不用CLIP测试
            device=device
        )
        
        # 创建测试数据
        batch_size = 2
        features = torch.randn(batch_size, 256, 16, 48, device=device)
        text_prompts = ["a wooden chair", "a modern table"]
        timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
        
        print(f"输入特征形状: {features.shape}")
        print(f"文本prompt: {text_prompts}")
        
        # 测试前向传播（注意：不使用CLIP时需要预处理文本）
        # 为了测试，我们直接使用token ids
        dummy_tokens = torch.randint(1, 1000, (batch_size, 77), device=device)
        
        with torch.no_grad():
            output = model.model(features, timesteps, dummy_tokens)
            print(f"✅ 文本控制模式前向传播成功！输出形状: {output.sample.shape}")
        
        # 测试生成（使用CLIP版本）
        print("\n=== 测试CLIP文本编码器版本 ===")
        model_clip = TextControlledDDIMModel(
            in_channels=256,
            out_channels=256,
            feature_size=(16, 48),
            cross_attention_dim=1024,
            use_clip=True,
            device=device
        )
        
        with torch.no_grad():
            generated = model_clip.generate(text_prompts, num_inference_steps=5)
            print(f"✅ 文本控制生成成功！生成特征形状: {generated.shape}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 基于Cross-Attention的文本控制UNet模型测试完成！") 