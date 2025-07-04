import torch
from torch.utils.data import Dataset
import os

# ShapeNet类别映射 (示例，您需要根据实际类别调整)
CATEGORY_MAPPING = {
    "02958343": 0,  # car (假设)
    "03001627": 1,  # chair (假设)
    "04379243": 2,  # table (假设) 
    "02691156": 3,  # airplane (假设)
}

# 类别名称映射
CATEGORY_NAMES = {
    0: "car",
    1: "chair", 
    2: "table",
    3: "airplane"
}

class ConditionalFeatureDataset(Dataset):
    """
    支持多类别条件生成的数据集
    返回特征和对应的类别标签
    """
    def __init__(self, train_path, tri_dir, num_classes=4):
        self.train_path = train_path
        self.tri_dir = tri_dir
        self.num_classes = num_classes
        
        # 读取文件列表
        with open(train_path, 'r') as f:
            all_tri_list = f.read().splitlines()
        
        # 通过category_id筛选，只保留在CATEGORY_MAPPING中的样本
        self.tri_list = []
        filtered_count = 0
        
        for tri_name in all_tri_list:
            category_id = tri_name.split('/')[0] if '/' in tri_name else tri_name.split('_')[0]
            if category_id in CATEGORY_MAPPING:
                self.tri_list.append(tri_name)
            else:
                filtered_count += 1
        
        print(f"数据筛选完成:")
        print(f"  原始样本数: {len(all_tri_list)}")
        print(f"  过滤样本数: {filtered_count}")
        print(f"  保留样本数: {len(self.tri_list)}")
        
        # 构建类别统计
        self.category_stats = {}
        for tri_name in self.tri_list:
            category_id = tri_name.split('/')[0] if '/' in tri_name else tri_name.split('_')[0]
            label = CATEGORY_MAPPING[category_id]  # 这里不需要再检查，因为已经筛选过了
            if label not in self.category_stats:
                self.category_stats[label] = 0
            self.category_stats[label] += 1
        
        print(f"数据集统计:")
        for label, count in self.category_stats.items():
            category_name = CATEGORY_NAMES.get(label, f"unknown_{label}")
            print(f"  类别 {label} ({category_name}): {count} 样本")
        print(f"总样本数: {len(self.tri_list)}")

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, idx):
        tri_name = self.tri_list[idx]
        
        # 提取类别信息
        category_id = tri_name.split('/')[0] if '/' in tri_name else tri_name.split('_')[0]
        
        # 映射到类别标签（由于已经筛选过，所有category_id都在映射中）
        label = CATEGORY_MAPPING[category_id]
        
        # 加载特征数据
        data_path = os.path.join(self.tri_dir, tri_name + ".pt")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        try:
            data_tensor = torch.load(data_path, map_location='cpu')
            features = data_tensor.squeeze(0)
        except Exception as e:
            raise RuntimeError(f"加载数据失败 {data_path}: {e}")
        
        return {
            "features": features,  # 特征数据 [C, H, W]
            "label": torch.tensor(label, dtype=torch.long),  # 类别标签
            "category_name": CATEGORY_NAMES.get(label, f"unknown_{label}"),  # 类别名称
            "tri_name": tri_name  # 原始文件名（用于调试）
        }
    
    def get_class_samples(self, class_label, num_samples=10):
        """获取指定类别的样本索引"""
        indices = []
        for idx, tri_name in enumerate(self.tri_list):
            category_id = tri_name.split('/')[0] if '/' in tri_name else tri_name.split('_')[0]
            if category_id in CATEGORY_MAPPING:
                label = CATEGORY_MAPPING[category_id]
                if label == class_label:
                    indices.append(idx)
                    if len(indices) >= num_samples:
                        break
        return indices

def create_category_mapping_from_data(train_path):
    """
    从训练数据中自动创建类别映射
    如果您的类别ID与上面的不同，使用此函数
    """
    with open(train_path, 'r') as f:
        tri_list = f.read().splitlines()
    
    # 提取所有唯一的类别ID
    category_ids = set()
    for tri_name in tri_list:
        category_id = tri_name.split('/')[0] if '/' in tri_name else tri_name.split('_')[0]
        category_ids.add(category_id)
    
    # 按字母顺序排序并分配标签
    sorted_categories = sorted(list(category_ids))
    category_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted_categories)}
    
    print("从数据中检测到的类别映射:")
    for cat_id, label in category_mapping.items():
        print(f"  {cat_id} -> {label}")
    
    return category_mapping, sorted_categories

if __name__ == "__main__":
    # 测试数据集
    train_path = "/home/lizihao/MMD3D_JS/data_split/shapenet/train.txt"
    tri_dir = "/home/lizihao/MMD3D_pr15_vae/exp-shapenet-all/05_15-18_55_55_finetune/triplane_256/model_geometry_opt-1st/train"
    
    # 创建类别映射（如果需要）
    # auto_mapping, auto_categories = create_category_mapping_from_data(train_path)
    
    # 创建数据集
    dataset = ConditionalFeatureDataset(train_path, tri_dir)
    
    # 测试数据加载
    print(f"\n数据集测试:")
    sample = dataset[0]
    print(f"特征形状: {sample['features'].shape}")
    print(f"类别标签: {sample['label'].item()}")
    print(f"类别名称: {sample['category_name']}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in dataloader:
        print(f"\nBatch测试:")
        print(f"特征形状: {batch['features'].shape}")
        print(f"标签形状: {batch['label'].shape}")
        print(f"标签值: {batch['label']}")
        break 