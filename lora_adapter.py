import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class LoRALayer(nn.Module):
    """LoRA层实现"""
    
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA矩阵 A 和 B
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LoRALinear(nn.Module):
    """LoRA线性层包装器"""
    
    def __init__(self, original_layer: nn.Linear, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora_layer = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # 冻结原始参数
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_layer(x)
        return original_output + lora_output


class AdapterLayer(nn.Module):
    """Adapter层实现"""
    
    def __init__(self, hidden_size, adapter_size=64, dropout=0.1):
        super().__init__()
        self.adapter_size = adapter_size
        
        # Adapter网络
        self.down_projection = nn.Linear(hidden_size, adapter_size)
        self.up_projection = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # 初始化
        nn.init.kaiming_uniform_(self.down_projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
    
    def forward(self, x):
        # 残差连接
        adapter_output = self.up_projection(
            self.activation(
                self.dropout(
                    self.down_projection(x)
                )
            )
        )
        return x + adapter_output


class LoRADiffusionModel:
    """使用LoRA的扩散模型"""
    
    def __init__(self, base_model_path, rank=16, alpha=32, dropout=0.1, device="cuda"):
        self.device = device
        self.rank = rank
        self.alpha = alpha
        
        # 加载基础模型
        from ddim_model import DDIMDiffusionModel
        self.base_model = DDIMDiffusionModel(device=device)
        self.base_model.load_model(base_model_path)
        
        # 应用LoRA到线性层
        self._apply_lora_to_model()
        
        # 保存原始调度器
        self.noise_scheduler = self.base_model.noise_scheduler
        self.train_scheduler = self.base_model.train_scheduler
    
    def _apply_lora_to_model(self):
        """将LoRA应用到模型的线性层"""
        self.lora_layers = {}
        
        for name, module in self.base_model.model.named_modules():
            if isinstance(module, nn.Linear):
                # 跳过某些层（如输出层）
                if "out" in name or "final" in name:
                    continue
                
                # 创建LoRA包装器
                lora_wrapper = LoRALinear(
                    module, 
                    rank=self.rank, 
                    alpha=self.alpha
                )
                
                # 替换原始层
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(self.base_model.model.named_modules())[parent_name]
                    setattr(parent_module, child_name, lora_wrapper)
                else:
                    setattr(self.base_model.model, child_name, lora_wrapper)
                
                self.lora_layers[name] = lora_wrapper
        
        print(f"已应用LoRA到 {len(self.lora_layers)} 个线性层")
    
    def get_lora_parameters(self):
        """获取LoRA参数"""
        lora_params = []
        for layer in self.lora_layers.values():
            lora_params.extend(layer.lora_layer.parameters())
        return lora_params
    
    def train_step(self, clean_features, optimizer):
        """LoRA训练步骤"""
        batch_size = clean_features.shape[0]
        
        # 添加噪声
        noise = torch.randn_like(clean_features)
        timesteps = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,), device=self.device)
        timesteps = timesteps.long()
        
        noisy_features = self.train_scheduler.add_noise(clean_features, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.base_model.model(noisy_features, timesteps).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, batch_size=1, num_inference_steps=50, eta=0.0):
        """使用LoRA模型生成样本"""
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        x = torch.randn(
            (batch_size, 256, 16, 48),
            device=self.device,
            dtype=torch.float32
        )
        
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.base_model.model(x, t).sample
            x = self.noise_scheduler.step(noise_pred, t, x, eta=eta).prev_sample
        
        return x
    
    def save_lora_weights(self, path):
        """保存LoRA权重"""
        lora_state_dict = {}
        for name, layer in self.lora_layers.items():
            lora_state_dict[f"{name}.lora_A.weight"] = layer.lora_layer.lora_A.weight
            lora_state_dict[f"{name}.lora_B.weight"] = layer.lora_layer.lora_B.weight
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'rank': self.rank,
            'alpha': self.alpha,
            'noise_scheduler': self.noise_scheduler,
            'train_scheduler': self.train_scheduler,
        }, path)
    
    def load_lora_weights(self, path):
        """加载LoRA权重"""
        checkpoint = torch.load(path, map_location=self.device)
        lora_state_dict = checkpoint['lora_state_dict']
        
        for name, layer in self.lora_layers.items():
            if f"{name}.lora_A.weight" in lora_state_dict:
                layer.lora_layer.lora_A.weight.data = lora_state_dict[f"{name}.lora_A.weight"]
            if f"{name}.lora_B.weight" in lora_state_dict:
                layer.lora_layer.lora_B.weight.data = lora_state_dict[f"{name}.lora_B.weight"]


class AdapterDiffusionModel:
    """使用Adapter的扩散模型"""
    
    def __init__(self, base_model_path, adapter_size=64, dropout=0.1, device="cuda"):
        self.device = device
        self.adapter_size = adapter_size
        
        # 加载基础模型
        from ddim_model import DDIMDiffusionModel
        self.base_model = DDIMDiffusionModel(device=device)
        self.base_model.load_model(base_model_path)
        
        # 应用Adapter到特定层
        self._apply_adapters_to_model()
        
        # 保存原始调度器
        self.noise_scheduler = self.base_model.noise_scheduler
        self.train_scheduler = self.base_model.train_scheduler
    
    def _apply_adapters_to_model(self):
        """将Adapter应用到模型的特定层"""
        self.adapter_layers = {}
        
        # 在UNet的中间块和上采样块中添加Adapter
        for name, module in self.base_model.model.named_modules():
            # 在注意力层后添加Adapter
            if "attn" in name and "to_out" in name:
                adapter = AdapterLayer(
                    module.out_features,
                    adapter_size=self.adapter_size,
                    dropout=dropout
                )
                
                # 替换原始层
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(self.base_model.model.named_modules())[parent_name]
                    setattr(parent_module, child_name, adapter)
                else:
                    setattr(self.base_model.model, child_name, adapter)
                
                self.adapter_layers[name] = adapter
        
        print(f"已应用Adapter到 {len(self.adapter_layers)} 个层")
    
    def get_adapter_parameters(self):
        """获取Adapter参数"""
        adapter_params = []
        for layer in self.adapter_layers.values():
            adapter_params.extend(layer.parameters())
        return adapter_params
    
    def train_step(self, clean_features, optimizer):
        """Adapter训练步骤"""
        batch_size = clean_features.shape[0]
        
        # 添加噪声
        noise = torch.randn_like(clean_features)
        timesteps = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,), device=self.device)
        timesteps = timesteps.long()
        
        noisy_features = self.train_scheduler.add_noise(clean_features, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.base_model.model(noisy_features, timesteps).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, batch_size=1, num_inference_steps=50, eta=0.0):
        """使用Adapter模型生成样本"""
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        x = torch.randn(
            (batch_size, 256, 16, 48),
            device=self.device,
            dtype=torch.float32
        )
        
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.base_model.model(x, t).sample
            x = self.noise_scheduler.step(noise_pred, t, x, eta=eta).prev_sample
        
        return x
    
    def save_adapter_weights(self, path):
        """保存Adapter权重"""
        adapter_state_dict = {}
        for name, layer in self.adapter_layers.items():
            adapter_state_dict[f"{name}.down_projection.weight"] = layer.down_projection.weight
            adapter_state_dict[f"{name}.down_projection.bias"] = layer.down_projection.bias
            adapter_state_dict[f"{name}.up_projection.weight"] = layer.up_projection.weight
            adapter_state_dict[f"{name}.up_projection.bias"] = layer.up_projection.bias
        
        torch.save({
            'adapter_state_dict': adapter_state_dict,
            'adapter_size': self.adapter_size,
            'noise_scheduler': self.noise_scheduler,
            'train_scheduler': self.train_scheduler,
        }, path)
    
    def load_adapter_weights(self, path):
        """加载Adapter权重"""
        checkpoint = torch.load(path, map_location=self.device)
        adapter_state_dict = checkpoint['adapter_state_dict']
        
        for name, layer in self.adapter_layers.items():
            if f"{name}.down_projection.weight" in adapter_state_dict:
                layer.down_projection.weight.data = adapter_state_dict[f"{name}.down_projection.weight"]
                layer.down_projection.bias.data = adapter_state_dict[f"{name}.down_projection.bias"]
            if f"{name}.up_projection.weight" in adapter_state_dict:
                layer.up_projection.weight.data = adapter_state_dict[f"{name}.up_projection.weight"]
                layer.up_projection.bias.data = adapter_state_dict[f"{name}.up_projection.bias"]


def compare_parameter_efficiency():
    """比较不同方法的参数效率"""
    print("=== 参数效率对比 ===\n")
    
    # 假设模型参数数量
    total_params = 100_000_000  # 1亿参数
    
    # 全参数微调
    full_finetune_params = total_params
    full_finetune_memory = total_params * 4  # 4字节/参数
    
    # LoRA (rank=16)
    lora_params = total_params * 0.01  # 约1%的参数
    lora_memory = lora_params * 4
    
    # Adapter (adapter_size=64)
    adapter_params = total_params * 0.005  # 约0.5%的参数
    adapter_memory = adapter_params * 4
    
    print(f"模型总参数: {total_params:,}")
    print(f"全参数微调: {full_finetune_params:,} 参数 ({full_finetune_memory/1024**3:.2f} GB)")
    print(f"LoRA微调: {lora_params:,} 参数 ({lora_memory/1024**3:.2f} GB)")
    print(f"Adapter微调: {adapter_params:,} 参数 ({adapter_memory/1024**3:.2f} GB)")
    
    print(f"\n内存节省:")
    print(f"LoRA vs 全参数: {(1 - lora_memory/full_finetune_memory)*100:.1f}%")
    print(f"Adapter vs 全参数: {(1 - adapter_memory/full_finetune_memory)*100:.1f}%")
    print(f"LoRA vs Adapter: {(1 - lora_memory/adapter_memory)*100:.1f}%")


if __name__ == "__main__":
    compare_parameter_efficiency() 