## 写py
在example/ranking建立run_ml_deepfm.py

```python
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.features import SparseFeature
from sklearn.preprocessing import LabelEncoder # === 必须引入这个工具 ===

# === 自定义数据集类 (保持不变) ===
class DeepFMDataset(Dataset):
    def __init__(self, feature_dict, label_array):
        # 此时 feature_dict 里的值已经是 int 类型了，torch.tensor 没问题
        self.features = {k: torch.tensor(v).long() for k, v in feature_dict.items()} # 强制转为long类型
        self.label = torch.tensor(label_array).float()
        
    def __getitem__(self, index):
        return {k: v[index] for k, v in self.features.items()}, self.label[index]
        
    def __len__(self):
        return len(self.label)

def main():
    print(">>> 正在运行最终完美版代码 <<<")
    
    # 1. 读取数据
    data_path = './data/ml-1m/ml-1m_sample.csv'
    print("正在读取数据...")
    data = pd.read_csv(data_path)
    
    # 2. 数据预处理
    print("正在进行 LabelEncoding (字符串转数字)...")
    data['label'] = (data['rating'] > 3).astype(int)
    
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'genres']
    
    for feature in sparse_features:
        lbe = LabelEncoder()
        # 把这一列的字符串都转成数字索引
        data[feature] = lbe.fit_transform(data[feature])
    
    # 定义特征配置
    feature_cols = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) 
                    for feature_name in sparse_features]
    
    # 3. 划分训练集和测试集
    train_size = int(len(data) * 0.9)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # 准备输入数据 (字典格式)
    train_model_input = {name: train_data[name].values for name in sparse_features}
    train_label = train_data['label'].values
    test_model_input = {name: test_data[name].values for name in sparse_features}
    test_label = test_data['label'].values
    
    # 4. 生成数据加载器
    print("正在生成 DataLoader...")
    train_dataset = DeepFMDataset(train_model_input, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    
    test_dataset = DeepFMDataset(test_model_input, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    # 5. 定义模型 (包含之前的修复)
    model = DeepFM(
        deep_features=feature_cols, 
        fm_features=feature_cols, 
        mlp_params={"dims": [256, 128]}
    )
    
    # 6. 开始训练
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"开始训练 DeepFM，使用设备: {device}")
    
     if not os.path.exists('./saved'):
        os.makedirs('./saved')
    
    trainer = CTRTrainer(
        model, 
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        n_epoch=5, 
        earlystop_patience=2, 
        device=device,
        model_path='./saved'
    )
    
    trainer.fit(train_dataloader, test_dataloader)
    
    # 7. 评估结果
    print("正在评估...")
    auc = trainer.evaluate(test_dataloader, abs=True)
    print(f"最终评估结果 AUC: {auc}")

if __name__ == '__main__':
    main()
```

## 运行
python example/ranking/run_ml_deepfm.py

## 结果
AUC 0.72