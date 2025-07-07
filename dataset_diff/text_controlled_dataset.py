#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本控制数据集 - 支持读取2D图片和对应的文本描述
用于文本控制的扩散模型训练
通过data_split中的id作为key读取json中的文本prompt
"""

import torch
from torch.utils.data import Dataset
import json
import os
import random

class TextControlledDataset(Dataset):
    """
    文本控制数据集
    同时加载特征数据和对应的文本描述
    """
    
    def __init__(self, 
                 train_path,           # 训练列表文件路径
                 tri_dir,             # 特征数据目录
                 text_json_path,      # 文本描述json文件路径
                 feature_ext='.pt',   # 特征文件扩展名
                 random_text=True):   # 是否随机选择文本描述
        
        self.train_path = train_path
        self.tri_dir = tri_dir
        self.text_json_path = text_json_path
        self.feature_ext = feature_ext
        self.random_text = random_text
        
        # 读取训练列表
        with open(train_path, 'r') as f:
            self.data_list = f.read().splitlines()
        
        # 读取文本描述json文件
        try:
            with open(text_json_path, 'r', encoding='utf-8') as f:
                self.text_prompts = json.load(f)
            print(f"成功加载文本描述文件: {text_json_path}")
            print(f"文本描述数量: {len(self.text_prompts)}")
        except Exception as e:
            raise RuntimeError(f"加载文本描述文件失败 {text_json_path}: {e}")
        
        # 验证数据完整性
        self._validate_data()
        
        print(f"文本控制数据集初始化完成:")
        print(f"  训练样本数: {len(self.data_list)}")
        print(f"  特征目录: {tri_dir}")
        print(f"  文本描述文件: {text_json_path}")
        print(f"  随机选择文本: {self.random_text}")
    
    def _validate_data(self):
        """验证数据完整性"""
        valid_samples = []
        missing_features = 0
        missing_texts = 0
        
        print("正在验证数据完整性...")
        sample_count = min(100, len(self.data_list))  # 只验证前100个
        
        for i, sample_name in enumerate(self.data_list[:sample_count]):
            # 检查特征文件
            feature_path = os.path.join(self.tri_dir, sample_name + self.feature_ext)
            feature_exists = os.path.exists(feature_path)
            
            # 检查文本描述
            text_exists = sample_name in self.text_prompts
            
            if feature_exists and text_exists:
                valid_samples.append(sample_name)
            else:
                if not feature_exists:
                    missing_features += 1
                if not text_exists:
                    missing_texts += 1
        
        print(f"数据验证结果 (前{sample_count}个样本):")
        print(f"  有效样本: {len(valid_samples)}")
        print(f"  缺失特征: {missing_features}")
        print(f"  缺失文本: {missing_texts}")
        
        if len(valid_samples) == 0:
            raise RuntimeError("没有找到有效的样本！请检查数据路径和文件格式。")
    
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
        
        # 获取文本描述
        if sample_name in self.text_prompts:
            text_descriptions = self.text_prompts[sample_name]
            
            # 如果文本描述是列表，根据设置选择
            if isinstance(text_descriptions, list):
                if self.random_text:
                    # 随机选择一个文本描述
                    text_prompt = random.choice(text_descriptions)
                else:
                    # 固定选择第一个文本描述
                    text_prompt = text_descriptions[0]
            else:
                # 如果是单个字符串，直接使用
                text_prompt = text_descriptions
        else:
            # 如果没有对应的文本，使用默认文本
            text_prompt = "a 3d object"
            print(f"警告: 样本 {sample_name} 没有对应的文本描述，使用默认文本")
        
        return {
            "features": features,              # 目标特征 [C, H, W]
            "text_prompt": text_prompt,        # 文本描述
            "sample_name": sample_name,        # 样本名称
            "feature_path": feature_path,      # 特征路径
        }
    
    def get_sample_info(self, idx):
        """获取样本详细信息"""
        sample_name = self.data_list[idx]
        feature_path = os.path.join(self.tri_dir, sample_name + self.feature_ext)
        
        return {
            "sample_name": sample_name,
            "feature_path": feature_path,
            "feature_exists": os.path.exists(feature_path),
            "text_exists": sample_name in self.text_prompts,
            "text_prompt": self.text_prompts.get(sample_name, "N/A")
        }


class TextControlledDatasetWithCategories(TextControlledDataset):
    """
    支持类别的文本控制数据集
    同时支持文本控制和类别条件
    继承自TextControlledDataset
    """
    
    def __init__(self, 
                 train_path,
                 tri_dir,
                 text_json_path,
                 category_mapping=None,  # 类别映射字典
                 feature_ext='.pt',
                 random_text=True):      # 是否随机选择文本描述
        
        super().__init__(
            train_path=train_path,
            tri_dir=tri_dir,
            text_json_path=text_json_path,
            feature_ext=feature_ext,
            random_text=random_text
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


def create_text_controlled_dataset(train_path, tri_dir, text_json_path, use_categories=False, random_text=True, **kwargs):
    """
    创建文本控制数据集
    
    Args:
        train_path: 训练列表文件
        tri_dir: 特征目录
        text_json_path: 文本描述json文件路径
        use_categories: 是否使用支持类别的数据集
        random_text: 是否随机选择文本描述（推荐True）
        **kwargs: 其他参数
    """
    
    # 创建数据集参数
    dataset_kwargs = {
        'train_path': train_path,
        'tri_dir': tri_dir,
        'text_json_path': text_json_path,
        'random_text': random_text,
        **kwargs
    }
    
    if use_categories:
        return TextControlledDatasetWithCategories(**dataset_kwargs)
    else:
        return TextControlledDataset(**dataset_kwargs)


# if __name__ == "__main__":
#     # 测试文本控制数据集
#     print("测试文本控制数据集...")
    
#     # 示例路径（需要根据实际情况调整）
#     train_path = "/home/lizihao/MMD3D_JS/data_split/shapenet/train.txt"
#     tri_dir = "/home/lizihao/MMD3D_pr15_vae/exp-shapenet-all/05_15-18_55_55_finetune/triplane_256/model_geometry_opt-1st/train"
#     text_json_path = "/mnt/nvme0n1/datasets/ShapeNet/BaseColor/Cap3D_automated_ShapeNet.json"  # 需要指定实际的文本描述文件
    
#     try:
#         # 测试基础文本控制数据集
#         print("\n=== 测试基础文本控制数据集 ===")
        
#         # 创建示例文本描述文件（用于测试）
#         sample_texts = {
#             "02691156/1a04e3eab45ca15dd86060f189eb133": "a wooden chair with four legs",
#             "02691156/1a6f615e8b1b5ae4dbbc9440457e303e": "a modern office chair",
#             "02691156/1a74a83fa6d24b3cacd67ce2c72c02e": "a vintage armchair"
#         }
        
#         # 如果文本文件不存在，创建一个示例文件
#         if not os.path.exists(text_json_path):
#             test_json_path = "test_text_descriptions.json"
#             with open(test_json_path, 'w', encoding='utf-8') as f:
#                 json.dump(sample_texts, f, ensure_ascii=False, indent=2)
#             text_json_path = test_json_path
#             print(f"创建测试文本描述文件: {test_json_path}")
        
#         dataset = create_text_controlled_dataset(
#             train_path=train_path,
#             tri_dir=tri_dir,
#             text_json_path=text_json_path,
#             use_categories=False
#         )
        
#         print(f"文本控制数据集大小: {len(dataset)}")
        
#         # 测试数据加载
#         if len(dataset) > 0:
#             sample = dataset[0]
#             print(f"\n样本测试:")
#             print(f"特征形状: {sample['features'].shape}")
#             print(f"文本描述: {sample['text_prompt']}")
#             print(f"样本名称: {sample['sample_name']}")
        
#         # 测试类别支持的数据集
#         print("\n=== 测试支持类别的文本控制数据集 ===")
#         dataset_with_categories = create_text_controlled_dataset(
#             train_path=train_path,
#             tri_dir=tri_dir,
#             text_json_path=text_json_path,
#             use_categories=True
#         )
        
#         print(f"支持类别的文本控制数据集大小: {len(dataset_with_categories)}")
        
#         if len(dataset_with_categories) > 0:
#             sample_cat = dataset_with_categories[0]
#             print(f"\n类别样本测试:")
#             print(f"特征形状: {sample_cat['features'].shape}")
#             print(f"文本描述: {sample_cat['text_prompt']}")
#             print(f"类别标签: {sample_cat['category_label'].item()}")
#             print(f"类别ID: {sample_cat['category_id']}")
        
#         print("✅ 文本控制数据集测试成功！")
        
#         # 清理测试文件
#         if os.path.exists("test_text_descriptions.json"):
#             os.remove("test_text_descriptions.json")
#             print("已清理测试文件")
        
#     except Exception as e:
#         print(f"❌ 测试失败: {e}")
#         print("请检查数据路径是否正确")
#         import traceback
#         traceback.print_exc() 