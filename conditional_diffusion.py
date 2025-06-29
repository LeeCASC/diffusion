import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
import numpy as np
from typing import Optional, Union, Dict, Any
from PIL import Image
import os


class ImageEncoder(nn.Module):
    """图像编码器，将 (3, 256, 256) 图像编码为条件特征"""
    
    def __init__(self, output_dim=256, hidden_dim=512):
        super().__init__()
        
        # 使用CLIP视觉编码器作为backbone
        self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # 冻结CLIP参数
        for param in self.clip_vision.parameters():
            param.requires_grad = False
        
        # 投影层，将CLIP特征映射到目标维度
        self.projection = nn.Sequential(
            nn.Linear(self.clip_vision.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 空间投影，将特征映射到目标空间尺寸 (16, 48)
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((16, 48))
        )
    
    def forward(self, images):
        """
        Args:
            images: (batch_size, 3, 256, 256)
        Returns:
            condition_features: (batch_size, 256, 16, 48)
        """
        # CLIP编码
        clip_output = self.clip_vision(images)
        pooled_features = clip_output.pooler_output  # (batch_size, hidden_size)
        
        # 投影到目标维度
        projected_features = self.projection(pooled_features)  # (batch_size, 256)
        
        # 重塑并空间投影
        batch_size = projected_features.shape[0]
        spatial_features = projected_features.view(batch_size, 256, 1, 1)
        spatial_features = spatial_features.expand(-1, -1, 32, 32)  # 扩展到32x32
        condition_features = self.spatial_projection(spatial_features)  # (batch_size, 256, 16, 48)
        
        return condition_features


class TextEncoder(nn.Module):
    """文本编码器，将文本编码为条件特征"""
    
    def __init__(self, output_dim=256, hidden_dim=512):
        super().__init__()
        
        # 使用CLIP文本编码器
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # 冻结CLIP参数
        for param in self.clip_text.parameters():
            param.requires_grad = False
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(self.clip_text.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 空间投影
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((16, 48))
        )
    
    def forward(self, text_inputs):
        """
        Args:
            text_inputs: 文本输入（可以是字符串列表或tokenized输入）
        Returns:
            condition_features: (batch_size, 256, 16, 48)
        """
        # 如果是字符串，先tokenize
        if isinstance(text_inputs[0], str):
            tokenized = self.tokenizer(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            tokenized = {k: v.to(self.clip_text.device) for k, v in tokenized.items()}
        else:
            tokenized = text_inputs
        
        # CLIP编码
        clip_output = self.clip_text(**tokenized)
        pooled_features = clip_output.pooler_output  # (batch_size, hidden_size)
        
        # 投影到目标维度
        projected_features = self.projection(pooled_features)  # (batch_size, 256)
        
        # 重塑并空间投影
        batch_size = projected_features.shape[0]
        spatial_features = projected_features.view(batch_size, 256, 1, 1)
        spatial_features = spatial_features.expand(-1, -1, 32, 32)  # 扩展到32x32
        condition_features = self.spatial_projection(spatial_features)  # (batch_size, 256, 16, 48)
        
        return condition_features


class ConditionalUNet(nn.Module):
    """条件UNet模型，支持图像和文本条件"""
    
    def __init__(self, in_channels=256, out_channels=256, condition_dim=256):
        super().__init__()
        
        # 基础UNet
        self.unet = UNet2DModel(
            sample_size=(16, 48),
            in_channels=in_channels + condition_dim,  # 输入 + 条件
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 1024),
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
            norm_num_groups=32,
            add_attention=True,
        )
        
        # 条件编码器
        self.image_encoder = ImageEncoder(output_dim=condition_dim)
        self.text_encoder = TextEncoder(output_dim=condition_dim)
        
        # 条件融合层
        self.condition_fusion = nn.Sequential(
            nn.Conv2d(condition_dim, condition_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(condition_dim, condition_dim, kernel_size=1)
        )
    
    def forward(self, x, timesteps, image_condition=None, text_condition=None, return_dict=True):
        """
        Args:
            x: 输入特征 (batch_size, 256, 16, 48)
            timesteps: 时间步
            image_condition: 图像条件 (batch_size, 3, 256, 256) 或 None
            text_condition: 文本条件 (字符串列表) 或 None
            return_dict: 是否返回字典
        """
        batch_size = x.shape[0]
        condition_features = torch.zeros(batch_size, 256, 16, 48, device=x.device)
        
        # 处理图像条件
        if image_condition is not None:
            image_features = self.image_encoder(image_condition)
            condition_features = condition_features + image_features
        
        # 处理文本条件
        if text_condition is not None:
            text_features = self.text_encoder(text_condition)
            condition_features = condition_features + text_features
        
        # 融合条件特征
        condition_features = self.condition_fusion(condition_features)
        
        # 拼接输入和条件
        combined_input = torch.cat([x, condition_features], dim=1)  # (batch_size, 512, 16, 48)
        
        # UNet前向传播
        return self.unet(combined_input, timesteps, return_dict=return_dict)


class ConditionalDDIMDiffusionModel:
    """条件DDIM扩散模型"""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.model = ConditionalUNet().to(device)
        
        # 调度器
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
    
    def train_step(self, clean_features, optimizer, image_condition=None, text_condition=None):
        """条件训练步骤"""
        batch_size = clean_features.shape[0]
        
        # 添加噪声
        noise = torch.randn_like(clean_features)
        timesteps = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        
        noisy_features = self.train_scheduler.add_noise(clean_features, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.model(noisy_features, timesteps, image_condition, text_condition).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, batch_size=1, num_inference_steps=50, eta=0.0, 
               image_condition=None, text_condition=None, cfg_scale=7.5):
        """
        条件采样
        Args:
            cfg_scale: Classifier-Free Guidance scale，0表示无条件生成
        """
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # 初始化噪声
        x = torch.randn(
            (batch_size, 256, 16, 48),
            device=self.device,
            dtype=torch.float32
        )
        
        # CFG采样
        if cfg_scale > 0 and (image_condition is not None or text_condition is not None):
            # 条件生成
            for t in self.noise_scheduler.timesteps:
                noise_pred_cond = self.model(x, t, image_condition, text_condition).sample
                noise_pred_uncond = self.model(x, t, None, None).sample
                
                # CFG插值
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                
                scheduler_output = self.noise_scheduler.step(noise_pred, int(t), x, eta=eta)
                x = scheduler_output.prev_sample
        else:
            # 普通采样
            for t in self.noise_scheduler.timesteps:
                noise_pred = self.model(x, t, image_condition, text_condition).sample
                scheduler_output = self.noise_scheduler.step(noise_pred, int(t), x, eta=eta)
                x = scheduler_output.prev_sample
        
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


class ConditionalLoRAModel:
    """条件LoRA模型"""
    
    def __init__(self, base_model_path, rank=16, alpha=32, device="cuda"):
        self.device = device
        
        # 加载基础条件模型
        self.base_model = ConditionalDDIMDiffusionModel(device=device)
        self.base_model.load_model(base_model_path)
        
        # 应用LoRA
        self._apply_lora_to_model()
    
    def _apply_lora_to_model(self):
        """将LoRA应用到条件模型的线性层"""
        from lora_adapter import LoRALinear
        
        self.lora_layers = {}
        
        for name, module in self.base_model.model.named_modules():
            if isinstance(module, nn.Linear):
                if "out" in name or "final" in name:
                    continue
                
                lora_wrapper = LoRALinear(module, rank=16, alpha=32)
                
                # 替换层
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(self.base_model.model.named_modules())[parent_name]
                    setattr(parent_module, child_name, lora_wrapper)
                else:
                    setattr(self.base_model.model, child_name, lora_wrapper)
                
                self.lora_layers[name] = lora_wrapper
        
        print(f"已应用LoRA到条件模型的 {len(self.lora_layers)} 个线性层")
    
    def get_lora_parameters(self):
        """获取LoRA参数"""
        lora_params = []
        for layer in self.lora_layers.values():
            lora_params.extend(layer.lora_layer.parameters())
        return lora_params
    
    def train_step(self, clean_features, optimizer, image_condition=None, text_condition=None):
        """LoRA训练步骤"""
        return self.base_model.train_step(clean_features, optimizer, image_condition, text_condition)
    
    @torch.no_grad()
    def sample(self, batch_size=1, num_inference_steps=50, eta=0.0, 
               image_condition=None, text_condition=None, cfg_scale=7.5):
        """LoRA采样"""
        return self.base_model.sample(batch_size, num_inference_steps, eta, 
                                     image_condition, text_condition, cfg_scale)
    
    def save_lora_weights(self, path):
        """保存LoRA权重"""
        lora_state_dict = {}
        for name, layer in self.lora_layers.items():
            lora_state_dict[f"{name}.lora_A.weight"] = layer.lora_layer.lora_A.weight
            lora_state_dict[f"{name}.lora_B.weight"] = layer.lora_layer.lora_B.weight
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'noise_scheduler': self.base_model.noise_scheduler,
            'train_scheduler': self.base_model.train_scheduler,
        }, path)


def create_conditional_dataset(image_dir=None, text_file=None, num_samples=10000):
    """创建条件数据集"""
    from torch.utils.data import Dataset
    
    class ConditionalFeatureDataset(Dataset):
        def __init__(self, image_dir, text_file, num_samples=10000):
            self.image_dir = image_dir
            self.text_file = text_file
            self.num_samples = num_samples
            self.feature_shape = (256, 16, 48)
            
            # 加载文本数据
            if text_file and os.path.exists(text_file):
                with open(text_file, 'r', encoding='utf-8') as f:
                    self.texts = [line.strip() for line in f.readlines()]
            else:
                self.texts = [f"Sample text {i}" for i in range(num_samples)]
            
            # 加载图像数据
            if image_dir and os.path.exists(image_dir):
                self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            else:
                self.image_files = None
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 生成特征
            features = torch.randn(self.feature_shape)
            
            # 生成图像条件（随机图像或从文件加载）
            if self.image_files:
                # 从文件加载图像
                try:
                    import torchvision.transforms as transforms
                    from PIL import Image
                    
                    img_file = self.image_files[idx % len(self.image_files)]
                    img_path = os.path.join(self.image_dir, img_file)
                    image = Image.open(img_path).convert('RGB')
                    
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                    ])
                    image_condition = transform(image)
                except ImportError:
                    # 如果没有torchvision，使用随机图像
                    image_condition = torch.randn(3, 256, 256)
            else:
                # 生成随机图像
                image_condition = torch.randn(3, 256, 256)
            
            # 文本条件
            text_condition = self.texts[idx % len(self.texts)]
            
            return {
                'features': features,
                'image_condition': image_condition,
                'text_condition': text_condition
            }
    
    return ConditionalFeatureDataset(image_dir, text_file, num_samples)


if __name__ == "__main__":
    # 测试条件模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建条件模型
    model = ConditionalDDIMDiffusionModel(device=device)
    
    # 测试前向传播
    batch_size = 2
    x = torch.randn(batch_size, 256, 16, 48).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    image_condition = torch.randn(batch_size, 3, 256, 256).to(device)
    text_condition = ["A beautiful landscape", "A modern city"]
    
    output = model.model(x, timesteps, image_condition, text_condition)
    print(f"输出形状: {output.sample.shape}")
    
    # 测试采样
    samples = model.sample(
        batch_size=2,
        image_condition=image_condition,
        text_condition=text_condition,
        cfg_scale=7.5
    )
    print(f"生成样本形状: {samples.shape}") 