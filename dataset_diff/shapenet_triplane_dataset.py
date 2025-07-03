from torch.utils.data import Dataset
import os
import torch

# class FeatureDataset(Dataset):
#     def __init__(self, train_path, tri_dir, mode="train"):
#         self.train_path = train_path
#         self.tri_dir = tri_dir
#         with open(train_path, 'r') as f:
#             self.tri_list = f.read().splitlines()

#     def __len__(self):
#         return len(self.tri_list)

#     def __getitem__(self, idx):
#         tri_name = self.tri_list[idx]
#         tri_class = tri_name.split('/')[0]
#         tri_label = shapenet_id_label_dict[tri_class]
#         data_path = os.path.join(self.tri_dir, tri_name+".pt")
#         data_tensor = torch.load(data_path)
#         return {"tri_latent": data_tensor.squeeze(0),
#                 "tri_class": torch.tensor(tri_label, dtype=torch.long)}


class FeatureDatasetSingleClass(Dataset):
    def __init__(self, train_path, tri_dir):
        self.train_path = train_path
        self.tri_dir = tri_dir
        with open(train_path, 'r') as f:
            self.tri_list = f.read().splitlines()

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, idx):
        tri_name = self.tri_list[idx]
        tri_class = tri_name.split('/')[0]
        data_path = os.path.join(self.tri_dir, tri_name+".pt")
        data_tensor = torch.load(data_path)
        return data_tensor.squeeze(0)

# if __name__ == '__main__':
#     train_path = "/home/lizihao/MMD3D_JS/data_split/shapenet_car/train.txt"
#     tri_dir = "/home/lizihao/MMD3D_pr15_vae/exp-shapenet-all/05_15-18_55_55_finetune/triplane_256/model_geometry_opt-1st/train/02958343"
#     dataset = FeatureDatasetSingleClass(train_path, tri_dir)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
#     for i, data in enumerate(dataloader):
#         print(i)
#         print(data.shape)
#         break