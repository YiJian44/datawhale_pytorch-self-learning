## 写代码
稍微修改[[DEEPFM精排模型]]的代码
在example/ranking建立run_ml——mmoe.py
```python
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# 引入 torch-rechub 的核心模块
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.basic.features import SparseFeature

# === 1. 自定义数据集类 (修复版) ===
# 修复点：标签必须打包成张量 (Tensor)，不能是字典
class MultiTaskDataset(Dataset):
    def __init__(self, feature_dict, label_dict):
        # 特征转为 LongTensor (整数索引)
        self.features = {k: torch.tensor(v).long() for k, v in feature_dict.items()}
        
        # 标签转为 FloatTensor 并堆叠
        # 例如: click=[0,1], like=[0,0] -> [[0,0], [1,0]]
        # 这样 Trainer 才能把它们 .to(device)
        self.labels = torch.stack([torch.tensor(v).float() for v in label_dict.values()], dim=1)
        
    def __getitem__(self, index):
        # 返回 (特征字典, 标签张量)
        return {k: v[index] for k, v in self.features.items()}, self.labels[index]
        
    def __len__(self):
        return len(self.labels)

def main():
    print(">>> 正在运行 Task04: MMOE 多任务模型 (最终修复版) <<<")
    
    # 2. 读取数据
    data_path = './data/ml-1m/ml-1m_sample.csv'
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到文件 {data_path}")
        return
        
    print("正在读取数据...")
    data = pd.read_csv(data_path)
    
    # 3. 数据预处理
    print("正在进行 LabelEncoding...")
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'genres']
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])

    # 构造两个任务的标签
    # 任务1 (Click): 评分 > 3
    data['label_click'] = (data['rating'] > 3).astype(int)
    # 任务2 (Like): 评分 == 5
    data['label_like'] = (data['rating'] == 5).astype(int)
    
    # 定义特征列
    feature_cols = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) 
                    for feature_name in sparse_features]

    # 4. 划分数据集
    train_size = int(len(data) * 0.9)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # 准备输入 (特征字典)
    train_model_input = {name: train_data[name].values for name in sparse_features}
    test_model_input = {name: test_data[name].values for name in sparse_features}
    
    # 准备标签 (注意字典顺序: click 在前, like 在后)
    train_label_dict = {'click': train_data['label_click'].values, 'like': train_data['label_like'].values}
    test_label_dict = {'click': test_data['label_click'].values, 'like': test_data['label_like'].values}
    
    # 5. 生成 DataLoader
    print("正在生成 DataLoader...")
    train_dataset = MultiTaskDataset(train_model_input, train_label_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    
    test_dataset = MultiTaskDataset(test_model_input, test_label_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    # 6. 定义 MMOE 模型 (修复参数名)
    # 修复点：使用 n_expert 而不是 num_experts
    model = MMOE(
        features=feature_cols, 
        task_types=['classification', 'classification'], 
        n_expert=3, 
        expert_params={"dims": [64, 32]}, 
        tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}] 
    )
    
    # 7. 开始训练
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"开始训练 MMOE，使用设备: {device}")
    
    # 自动创建保存文件夹
    if not os.path.exists('./saved'):
        os.makedirs('./saved')

    # 定义训练器 (修复参数名)
    # 修复点：使用 task_types 而不是 task_names
    trainer = MTLTrainer(
        model, 
        task_types=['classification', 'classification'], 
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        n_epoch=5, 
        earlystop_patience=2, 
        device=device,
        model_path='./saved'
    )
    
    trainer.fit(train_dataloader, test_dataloader)
    
    # 8. 评估
    print("正在评估...")
    # 修复点：必须传入 model 参数
    evaluate_res = trainer.evaluate(model, test_dataloader)
    
    print("-" * 30)
    print("✅ Task04 最终评估结果:")
    # 结果可能是 {0: auc_val, 1: auc_val}，0对应click，1对应like
    print(evaluate_res)
    print("-" * 30)

if __name__ == '__main__':
    main()
```

## 运行
python 在example/ranking/run_ml_mmoe.py

## 结果
1. **Click / 点击预估**
    - **AUC**：**0.7236**
2. ** Like / 喜爱预估**
    - **AUC**：**0.7316**
