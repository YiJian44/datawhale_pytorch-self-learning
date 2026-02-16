## 激活环境
win+R     cmd
E：
cd Ai_Project
.venv\Scrpits\activate.bat

## 克隆库
git clone https://github.com/datawhalechina/torch-rechub.git

## 库版本兼容
uv pip install -e .

## 下载数据
https://grouplens.org/datasets/movielens/1m/
下载1M的
放在data/ml-1m里

## 修改格式
由于下载出来的是.dat格式,而需要的是.csv格式
python
```python
import pandas as pd
import os

# 1. 定义文件路径 
root_path = "./data/ml-1m/"

# 2. 读取原始 .dat 文件 (MovieLens 格式是用 :: 分隔的)
# 读取用户数据
users = pd.read_csv(root_path + 'users.dat', sep='::', header=None, engine='python',  names=['user_id', 'gender', 'age', 'occupation', 'zip'])

# 读取电影数据 (注意编码，防止报错)
movies = pd.read_csv(root_path + 'movies.dat', sep='::', header=None, engine='python', names=['movie_id', 'title', 'genres'], encoding='latin-1')

# 读取评分数据
ratings = pd.read_csv(root_path + 'ratings.dat', sep='::', header=None, engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# 3. 合并数据
data = pd.merge(pd.merge(ratings, users), movies)

# 4. 只取前 1000 行作为 sample (避免数据量太大跑得慢，先跑通流程)
# 如果你想跑全量数据，把下面这就话删掉，或者改成 data = data
data_sample = data.sample(n=10000, random_state=2022) 

# 5. 保存为代码需要的 csv 文件
output_path = root_path + 'ml-1m_sample.csv'
data_sample.to_csv(output_path, index=False)

print(f"成功！文件已生成在: {output_path}")
```
exit（）

## 运行结果
python examples/matching/run_ml_dssm.py --model_name dssm --device cuda:0 --learning_rate 0.001 --epoch 5 --batch_size 4096 --weight_decay 0.0001 --save_dir saved/dssm_ml-100k
## 结果
Hit@10: 0.0032 (命中率)
NDCG@10: 0.0014 (归一化折损累计增益)
Recall@10: 0.0032 (召回率)

# 进阶
由于上述的结果太低，做出一些小的改变，使最终结果好一点。（增加数据量）
python
```python
import pandas as pd
import os

# 1. 路径设置
root_path = "./data/ml-1m/"

# 2. 读取原始数据
users = pd.read_csv(root_path + 'users.dat', sep='::', header=None, engine='python', names=['user_id', 'gender', 'age', 'occupation', 'zip'])
movies = pd.read_csv(root_path + 'movies.dat', sep='::', header=None, engine='python', names=['movie_id', 'title', 'genres'], encoding='latin-1')
ratings = pd.read_csv(root_path + 'ratings.dat', sep='::', header=None, engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# 3. 合并所有数据
data = pd.merge(pd.merge(ratings, users), movies)

# 4. 【关键修改】不再采样，直接使用全量数据！
print(f"全量数据行数: {len(data)}")  # 应该显示 100万左右

# 5. 为了偷懒不改代码，我们依然把它保存为 sample.csv，覆盖旧文件
output_path = root_path + 'ml-1m_sample.csv'
data.to_csv(output_path, index=False)

print(f"全量数据已保存至: {output_path}")
```

## 运行
python examples/matching/run_ml_dssm.py --model_name dssm --device cuda:0 --learning_rate 0.001 --epoch 10 --batch_size 2048 --weight_decay 0.0001 --save_dir saved/dssm_ml-1m_full

## 结果
|指标|之前 (1万条小数据)|**现在 (全量数据)**|**提升倍数**|
|---|---|---|---|
|**Hit@10 (命中率)**|0.0032 (0.32%)|**0.0219 (2.19%)**|**提升约 7 倍！**|
|**Recall@10 (召回率)**|0.0032|**0.0219**|**提升约 7 倍！**|
|**NDCG@10**|0.0014|**0.0097**|**提升约 7 倍！**|
