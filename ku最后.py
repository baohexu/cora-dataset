import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.datasets import Planetoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='D:\\QQ\\cora', name='Cora')
data = dataset[0]

class CoraDataset(Dataset):
    def __init__(self, pyg_data, mode='train'):
        self.data = pyg_data
        self.mode = mode
        
        self.features_df = pd.DataFrame(
           pyg_data.x.cpu().numpy(),
            columns=[f'feat_{i}' for i in range(pyg_data.x.shape[1])]
        )
        
        self.labels_series = pd.Series(
             pyg_data.y.cpu().numpy(),
            name='label'
        )
        
        if mode == 'train':
           mask = pyg_data.train_mask.cpu().numpy()
        elif mode == 'val':
            mask = pyg_data.val_mask.cpu().numpy()
        else:
            mask = pyg_data.test_mask.cpu().numpy()
        
        
        self.indices = pd.Series(range(len(mask)))[mask].values
        self.full_df = pd.concat([self.features_df, self.labels_series], axis=1)
        self.subset_df = self.full_df.loc[self.indices].reset_index(drop=True)
        
        self.features = torch.FloatTensor(self.subset_df.iloc[:, :-1].values).to(device)
        self.labels = torch.LongTensor(self.subset_df['label'].values).to(device)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.subset_df)

train_dataset = CoraDataset(data, mode='train')
val_dataset = CoraDataset(data, mode='val') 
test_dataset = CoraDataset(data, mode='test')

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"特征维度: {train_dataset.features.shape[1]}")
print(f"类别数量: {len(torch.unique(train_dataset.labels))}")

features, label = train_dataset[0]

print(f"样本特征形状: {features.shape}, 标签: {label.item()}")
