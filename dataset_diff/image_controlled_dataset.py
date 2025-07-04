#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图控数据集 - 支持读取2D图片和对应的特征数据
用于图像控制的扩散模型训练
支持两种图像格式：原始2D图像 和 预提取的图像特征
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import json

class ImageControlledDataset(Dataset):
    """
    图控数据集
    同时加载2D图片和对应的特征数据
    支持两种图像格式：
    1. raw_image: 原始2D图像，需要在训练时通过image encoder编码
    2. image_features: 预提取的图像特征，直接使用
    """
    
    def __init__(self, 
                 train_path,           # 训练列表文件路径
                 tri_dir,             # 特征数据目录
                 image_dir,           # 图片数据目录
                 image_format='jpg',  # 图片格式
                 feature_ext='.pt',   # 特征文件扩展名
                 image_mode='raw_image',  # 图像模式: 'raw_image' 或 'image_features'
                 image_size=256,      # 原始图像尺寸 (仅raw_image模式需要)
                 image_feature_ext='.pt'):  # 图像特征文件扩展名 (仅image_features模式需要)
        
        self.train_path = train_path
        self.tri_dir = tri_dir
        self.image_dir = image_dir
        self.image_format = image_format
        self.feature_ext = feature_ext
        self.image_mode = image_mode
        self.image_size = image_size
        self.image_feature_ext = image_feature_ext
        
        # 验证image_mode参数
        if image_mode not in ['raw_image', 'image_features']:
            raise ValueError(f"image_mode必须是'raw_image'或'image_features'，当前值: {image_mode}")
        
        # 读取训练列表
        with open(train_path, 'r') as f:
            self.data_list = f.read().splitlines()
        
        # 根据模式设置不同的预处理
        if image_mode == 'raw_image':
            # 原始图像模式：需要图片预处理transforms
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet标准化
            ])
        else:
            # 图像特征模式：不需要transforms
            self.image_transform = None
        
        # 验证数据完整性
        self._validate_data()
        
        print(f"图控数据集初始化完成:")
        print(f"  训练样本数: {len(self.data_list)}")
        print(f"  特征目录: {tri_dir}")
        print(f"  图片目录: {image_dir}")
        print(f"  图像模式: {image_mode}")
        if image_mode == 'raw_image':
            print(f"  图片尺寸: {image_size}x{image_size}")
        else:
            print(f"  图像特征扩展名: {image_feature_ext}")
    
    def _validate_data(self):
        """验证数据完整性"""
        valid_samples = []
        missing_features = 0
        missing_images = 0
        
        print("正在验证数据完整性...")
        sample_count = min(100, len(self.data_list))  # 只验证前100个
        
        for i, sample_name in enumerate(self.data_list[:sample_count]):
            # 检查特征文件
            feature_path = os.path.join(self.tri_dir, sample_name + self.feature_ext)
            
            # 根据模式检查图像文件
            if self.image_mode == 'raw_image':
                image_path = self._get_raw_image_path(sample_name)
            else:  # image_features
                image_path = self._get_image_feature_path(sample_name)
            
            feature_exists = os.path.exists(feature_path)
            image_exists = os.path.exists(image_path)
            
            if feature_exists and image_exists:
                valid_samples.append(sample_name)
            else:
                if not feature_exists:
                    missing_features += 1
                if not image_exists:
                    missing_images += 1
        
        print(f"数据验证结果 (前{sample_count}个样本):")
        print(f"  有效样本: {len(valid_samples)}")
        print(f"  缺失特征: {missing_features}")
        print(f"  缺失图像: {missing_images}")
    
    def _get_raw_image_path(self, sample_name):
        """获取原始图片路径"""
        # 支持多种图片路径格式
        sample_path = '/models/mv_model_normalized.obj_area_group_1/00000'
        possible_paths = [
            os.path.join(self.image_dir, sample_name + sample_path + f'.{self.image_format}'),
            os.path.join(self.image_dir, sample_name + sample_path + '.jpg'),
            os.path.join(self.image_dir, sample_name + sample_path + '.png'),
            os.path.join(self.image_dir, sample_name + sample_path + '.jpeg'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果都不存在，返回第一个路径（会在加载时报错）
        return possible_paths[0]
    
    def _get_image_feature_path(self, sample_name):
        """获取图像特征路径"""
        return os.path.join(self.image_dir, sample_name + self.image_feature_ext)
    
    def _load_raw_image(self, sample_name):
        """加载原始图像"""
        image_path = self._get_raw_image_path(sample_name)
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transform is not None:
                control_image = self.image_transform(image)
            else:
                raise RuntimeError("image_transform未初始化，但尝试加载原始图像")
            return control_image, image_path
        except Exception as e:
            raise RuntimeError(f"加载原始图片失败 {image_path}: {e}")
    
    def _load_image_features(self, sample_name):
        """加载预提取的图像特征"""
        feature_path = self._get_image_feature_path(sample_name)
        try:
            image_features = torch.load(feature_path, map_location='cpu')
            # 确保特征格式正确
            if image_features.dim() == 4:  # (1, C, H, W)
                image_features = image_features.squeeze(0)
            elif image_features.dim() != 3:  # 期望 (C, H, W)
                raise ValueError(f"图像特征维度错误，期望3D张量(C,H,W)，实际: {image_features.shape}")
            return image_features, feature_path
        except Exception as e:
            raise RuntimeError(f"加载图像特征失败 {feature_path}: {e}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample_name = self.data_list[idx]
        
        # 加载目标特征数据
        feature_path = os.path.join(self.tri_dir, sample_name + self.feature_ext)
        try:
            feature_tensor = torch.load(feature_path, map_location='cpu')
            if feature_tensor.dim() == 4:  # (1, C, H, W)
                features = feature_tensor.squeeze(0)
            else:  # (C, H, W)
                features = feature_tensor
        except Exception as e:
            raise RuntimeError(f"加载特征失败 {feature_path}: {e}")
        
        # 根据模式加载控制图像或图像特征
        if self.image_mode == 'raw_image':
            control_data, control_path = self._load_raw_image(sample_name)
            return {
                "features": features,              # 目标特征 [C, H, W]
                "control_image": control_data,     # 控制图片 [3, H, W]
                "control_type": "raw_image",       # 控制类型标识
                "sample_name": sample_name,        # 样本名称
                "feature_path": feature_path,      # 特征路径
                "control_path": control_path       # 控制数据路径
            }
        else:  # image_features
            control_data, control_path = self._load_image_features(sample_name)
            return {
                "features": features,              # 目标特征 [C, H, W]
                "control_features": control_data,  # 控制特征 [C, H, W]
                "control_type": "image_features",  # 控制类型标识
                "sample_name": sample_name,        # 样本名称
                "feature_path": feature_path,      # 特征路径
                "control_path": control_path       # 控制数据路径
            }
    
    def get_sample_info(self, idx):
        """获取样本详细信息"""
        sample_name = self.data_list[idx]
        feature_path = os.path.join(self.tri_dir, sample_name + self.feature_ext)
        
        if self.image_mode == 'raw_image':
            control_path = self._get_raw_image_path(sample_name)
        else:
            control_path = self._get_image_feature_path(sample_name)
        
        return {
            "sample_name": sample_name,
            "feature_path": feature_path,
            "control_path": control_path,
            "feature_exists": os.path.exists(feature_path),
            "control_exists": os.path.exists(control_path),
            "image_mode": self.image_mode
        }


class ImageControlledDatasetWithCategories(ImageControlledDataset):
    """
    支持类别的图控数据集
    同时支持图像控制和类别条件
    继承自ImageControlledDataset，支持两种图像模式
    """
    
    def __init__(self, 
                 train_path,
                 tri_dir,
                 image_dir,
                 category_mapping=None,  # 类别映射字典
                 image_format='jpg',
                 feature_ext='.pt',
                 image_mode='raw_image',  # 图像模式: 'raw_image' 或 'image_features'
                 image_size=256,
                 image_feature_ext='.pt'):
        
        super().__init__(
            train_path=train_path,
            tri_dir=tri_dir,
            image_dir=image_dir,
            image_format=image_format,
            feature_ext=feature_ext,
            image_mode=image_mode,
            image_size=image_size,
            image_feature_ext=image_feature_ext
        )
        
        # 设置类别映射
        if category_mapping is None:
            # 自动从数据中检测类别
            self.category_mapping = self._create_category_mapping()
        else:
            self.category_mapping = category_mapping
        
        self.num_classes = len(self.category_mapping)
        
        # 统计每个类别的样本数
        self._build_category_stats()
        
        print(f"类别信息:")
        print(f"  类别数量: {self.num_classes}")
        for cat_id, label in self.category_mapping.items():
            count = self.category_stats.get(label, 0)
            print(f"  {cat_id} -> 类别{label}: {count} 样本")
    
    def _create_category_mapping(self):
        """从数据中自动创建类别映射"""
        category_ids = set()
        for sample_name in self.data_list:
            # 假设类别ID是文件名的第一部分
            category_id = sample_name.split('/')[0] if '/' in sample_name else sample_name.split('_')[0]
            category_ids.add(category_id)
        
        # 按字母顺序排序并分配标签
        sorted_categories = sorted(list(category_ids))
        category_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted_categories)}
        
        return category_mapping
    
    def _build_category_stats(self):
        """构建类别统计信息"""
        self.category_stats = {}
        for sample_name in self.data_list:
            category_id = sample_name.split('/')[0] if '/' in sample_name else sample_name.split('_')[0]
            if category_id in self.category_mapping:
                label = self.category_mapping[category_id]
                self.category_stats[label] = self.category_stats.get(label, 0) + 1
    
    def __getitem__(self, idx):
        # 获取基础数据
        base_data = super().__getitem__(idx)
        
        # 添加类别信息
        sample_name = base_data["sample_name"]
        category_id = sample_name.split('/')[0] if '/' in sample_name else sample_name.split('_')[0]
        
        if category_id in self.category_mapping:
            category_label = self.category_mapping[category_id]
        else:
            category_label = 0  # 默认类别
        
        base_data["category_label"] = torch.tensor(category_label, dtype=torch.long)
        base_data["category_id"] = category_id
        
        return base_data


def create_dual_mode_dataset(train_path, tri_dir, image_dir, 
                           raw_image_dir=None, image_feature_dir=None,
                           prefer_mode='auto', **kwargs):
    """
    创建双模式数据集，自动选择最佳的图像模式
    
    Args:
        train_path: 训练列表文件
        tri_dir: 特征目录
        image_dir: 图像目录（如果只有一个目录）
        raw_image_dir: 原始图像目录（如果分别指定）
        image_feature_dir: 图像特征目录（如果分别指定）
        prefer_mode: 优先模式 'raw_image', 'image_features', 'auto'
    """
    
    # 确定图像目录
    if raw_image_dir and image_feature_dir:
        # 分别指定了两个目录
        raw_dir = raw_image_dir
        feature_dir = image_feature_dir
    else:
        # 使用同一个目录
        raw_dir = image_dir
        feature_dir = image_dir
    
    # 根据prefer_mode决定使用哪种模式
    if prefer_mode == 'auto':
        # 自动检测：优先使用预提取特征，如果没有则使用原始图像
        # 检查第一个样本来决定
        with open(train_path, 'r') as f:
            sample_list = f.read().splitlines()
        
        if sample_list:
            first_sample = sample_list[0]
            feature_path = os.path.join(feature_dir, first_sample + '.pt')
            sample_path = '/models/mv_model_normalized.obj_area_group_1/00000'
            raw_path_candidates = [
                os.path.join(raw_dir, first_sample + sample_path + '.jpg'),
                os.path.join(raw_dir, first_sample + sample_path + '.png'),
                os.path.join(raw_dir, first_sample + sample_path + '.jpeg'),
            ]
            
            has_features = os.path.exists(feature_path)
            has_raw = any(os.path.exists(p) for p in raw_path_candidates)
            
            if has_features:
                selected_mode = 'image_features'
                selected_dir = feature_dir
            elif has_raw:
                selected_mode = 'raw_image'
                selected_dir = raw_dir
            else:
                raise FileNotFoundError(f"未找到图像数据，检查的路径:\n特征: {feature_path}\n原始图像: {raw_path_candidates}")
        else:
            raise ValueError("训练列表为空")
    elif prefer_mode == 'raw_image':
        selected_mode = 'raw_image'
        selected_dir = raw_dir
    elif prefer_mode == 'image_features':
        selected_mode = 'image_features'
        selected_dir = feature_dir
    else:
        raise ValueError(f"未知的prefer_mode: {prefer_mode}")
    
    print(f"自动选择图像模式: {selected_mode}")
    print(f"使用图像目录: {selected_dir}")
    
    # 根据是否需要类别支持选择数据集类型
    use_categories = kwargs.pop('use_categories', False)  # 先移除use_categories参数
    
    # 创建数据集参数（不包含use_categories）
    dataset_kwargs = {
        'train_path': train_path,
        'tri_dir': tri_dir,
        'image_dir': selected_dir,
        'image_mode': selected_mode,
        **kwargs  # 现在kwargs中已经不包含use_categories了
    }
    
    if use_categories:
        return ImageControlledDatasetWithCategories(**dataset_kwargs)
    else:
        return ImageControlledDataset(**dataset_kwargs)


# if __name__ == "__main__":
#     # 测试图控数据集
#     print("测试图控数据集...")
    
#     # 示例路径（需要根据实际情况调整）
#     train_path = "/home/lizihao/MMD3D_JS/data_split/shapenet/train.txt"
#     tri_dir = "/home/lizihao/MMD3D_pr15_vae/exp-shapenet-all/05_15-18_55_55_finetune/triplane_256/model_geometry_opt-1st/train"
#     image_dir = "/mnt/nvme0n1/datasets/ShapeNet/BaseColor/ShapeNet_MV"  # 需要指定实际的图片目录
#     image_dir2 = "/mnt/nvme0n1/lizihao/model/baseline_mv/triplane_256/image_prompt"
    
#     try:
#         # 测试原始图像模式
#         print("\n=== 测试原始图像模式 ===")
#         dataset_raw = ImageControlledDataset(
#             train_path=train_path,
#             tri_dir=tri_dir,
#             image_dir=image_dir,
#             image_mode='raw_image',
#             image_size=256
#         )
        
#         print(f"原始图像模式数据集大小: {len(dataset_raw)}")
        
#         # 测试图像特征模式
#         print("\n=== 测试图像特征模式 ===")
#         dataset_features = ImageControlledDataset(
#             train_path=train_path,
#             tri_dir=tri_dir,
#             image_dir=image_dir2,
#             image_mode='image_features',
#             image_feature_ext='.pt'
#         )
        
#         print(f"图像特征模式数据集大小: {len(dataset_features)}")
        
#         # 测试自动模式选择
#         print("\n=== 测试自动模式选择 ===")
#         dataset_auto = create_dual_mode_dataset(
#             train_path=train_path,
#             tri_dir=tri_dir,
#             image_dir=image_dir,
#             prefer_mode='auto'
#         )
        
#         print(f"自动选择模式数据集大小: {len(dataset_auto)}")
        
#         # 测试数据加载
#         if len(dataset_auto) > 0:
#             sample = dataset_auto[0]
#             print(f"\n样本测试:")
#             print(f"特征形状: {sample['features'].shape}")
#             print(f"控制类型: {sample['control_type']}")
#             print(f"样本名称: {sample['sample_name']}")
            
#             if sample['control_type'] == 'raw_image':
#                 print(f"控制图片形状: {sample['control_image'].shape}")
#             else:
#                 print(f"控制特征形状: {sample['control_features'].shape}")
        
#         print("✅ 图控数据集测试成功！")
        
#     except Exception as e:
#         print(f"❌ 测试失败: {e}")
#         print("请检查数据路径是否正确")
#         import traceback
#         traceback.print_exc() 