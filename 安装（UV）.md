## 安装UV：
win+R          powershell
输入：powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
## 安装git
win+R          cmd
winget install --id Git.Git -e --source winget

## 环境布置
win+R          cmd
### 文件夹
E:
mkdir AI_Project
cd AI_Project

### 创建虚拟环境
uv venv .venv --python 3.9

### 激活虚拟环境
.venv\Scripts\activate.bat
### 安装torch 
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### 安装其他库
uv pip install numpy pandas scipy scikit-learn

## 环境创建好了

## 补充：
后续召回需要annoy库：
uv pip install annoy