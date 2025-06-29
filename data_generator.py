import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class SyntheticFeatureDataset(Dataset):
    """
    合成特征数据集，用于训练扩散模型
    生成具有特定模式的特征数据
    """
    
    def __init__(self, num_samples=10000, feature_shape=(256, 16, 48)):
        self.num_samples = num_samples
        self.feature_shape = feature_shape
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成具有特定模式的特征
        features = self._generate_synthetic_features()
        return features
    
    def _generate_synthetic_features(self):
        """
        生成合成特征，包含多种模式：
        1. 随机噪声
        2. 周期性模式
        3. 局部相关性
        """
        # 基础随机特征
        features = torch.randn(self.feature_shape)
        
        # 添加周期性模式
        for c in range(0, self.feature_shape[0], 32):
            # 在通道维度上添加周期性模式
            pattern = torch.sin(torch.arange(self.feature_shape[1]) * 0.5 + c * 0.1).unsqueeze(1)
            pattern = pattern.expand(-1, self.feature_shape[2])
            features[c:c+32] += pattern.unsqueeze(0) * 0.3
        
        # 添加空间相关性
        for h in range(self.feature_shape[1]):
            for w in range(self.feature_shape[2]):
                if h > 0 and w > 0:
                    # 添加局部平滑性
                    features[:, h, w] += 0.1 * (features[:, h-1, w] + features[:, h, w-1])
        
        # 归一化
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features


class RealisticFeatureDataset(Dataset):
    """
    更现实的特征数据集，模拟真实场景中的特征分布
    """
    
    def __init__(self, num_samples=10000, feature_shape=(256, 16, 48)):
        self.num_samples = num_samples
        self.feature_shape = feature_shape
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        features = self._generate_realistic_features()
        return features
    
    def _generate_realistic_features(self):
        """
        生成更现实的特征，模拟CNN特征图的特性
        """
        # 创建基础特征图
        features = torch.zeros(self.feature_shape)
        
        # 模拟不同层级的特征
        # 低级特征（边缘、纹理）
        low_level_features = torch.randn(64, self.feature_shape[1], self.feature_shape[2]) * 0.5
        features[:64] = low_level_features
        
        # 中级特征（形状、模式）
        mid_level_features = torch.randn(128, self.feature_shape[1], self.feature_shape[2]) * 0.8
        # 添加空间相关性
        for c in range(128):
            kernel = torch.randn(3, 3)
            mid_level_features[c] = torch.nn.functional.conv2d(
                mid_level_features[c].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze()
        features[64:192] = mid_level_features
        
        # 高级特征（语义信息）
        high_level_features = torch.randn(64, self.feature_shape[1], self.feature_shape[2]) * 1.2
        # 添加全局相关性
        global_context = torch.randn(64, 1, 1).expand(-1, self.feature_shape[1], self.feature_shape[2])
        high_level_features += global_context * 0.3
        features[192:] = high_level_features
        
        # 归一化
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features


def create_dataloader(dataset_type="synthetic", batch_size=8, num_samples=10000):
    """
    创建数据加载器
    Args:
        dataset_type: "synthetic" 或 "realistic"
        batch_size: 批量大小
        num_samples: 样本数量
    """
    if dataset_type == "synthetic":
        dataset = SyntheticFeatureDataset(num_samples=num_samples)
    elif dataset_type == "realistic":
        dataset = RealisticFeatureDataset(num_samples=num_samples)
    else:
        raise ValueError("dataset_type must be 'synthetic' or 'realistic'")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


def visualize_features(features, save_path=None):
    """
    可视化特征图
    Args:
        features: 特征张量 (batch_size, 256, 16, 48)
        save_path: 保存路径
    """
    batch_size = features.shape[0]
    
    # 选择第一个样本进行可视化
    sample_features = features[0]  # (256, 16, 48)
    
    # 创建子图
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    # 选择16个通道进行可视化
    channels_to_show = np.linspace(0, 255, 16, dtype=int)
    
    for i, channel in enumerate(channels_to_show):
        if i < 16:
            im = axes[i].imshow(sample_features[channel].cpu().numpy(), cmap='viridis')
            axes[i].set_title(f'Channel {channel}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # 测试数据生成器
    dataloader = create_dataloader("realistic", batch_size=4, num_samples=100)
    
    for batch_idx, features in enumerate(dataloader):
        print(f"Batch {batch_idx}: {features.shape}")
        print(f"Features stats - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        
        # 可视化第一个批次
        if batch_idx == 0:
            visualize_features(features, "sample_features.png")
        
        if batch_idx >= 2:  # 只测试前几个批次
            break 