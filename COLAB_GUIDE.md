# Google Colab 使用指南

## 快速开始（数据已在Google Drive）

如果数据已经处理完成并上传到Google Drive，使用以下步骤：

```python
# 1. 克隆仓库
!git clone https://github.com/wentao2024/youtube-shorts-recommender.git
%cd youtube-shorts-recommender

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 4. 开始训练（会自动从Drive加载数据）
!python3 train_kuairec_colab.py
```

**数据文件位置**：脚本会自动从以下位置查找数据（按优先级）：
1. `/content/data/` - 如果已经复制到这里
2. `/content/drive/MyDrive/AI/youtobe-shorts-data/` - 用户指定的Drive路径
3. `/content/drive/MyDrive/youtube-shorts-recommender/data/` - 默认Drive路径
4. 项目目录下的 `data/` - 本地路径

如果数据在其他位置，可以手动复制：
```python
!cp /content/drive/MyDrive/AI/youtobe-shorts-data/ratings_kuairec.csv /content/data/
!cp /content/drive/MyDrive/AI/youtobe-shorts-data/videos_kuairec.csv /content/data/
```

## 完整流程（从零开始）

### 1. 克隆仓库

```python
# 在Colab中运行
!git clone https://github.com/wentao2024/youtube-shorts-recommender.git
%cd youtube-shorts-recommender
```

**注意**：如果路径嵌套了（出现多个 `youtube-shorts-recommender` 目录），脚本会自动处理。

### 2. 安装依赖

```python
!pip install -r requirements.txt
```

### 3. 下载并预处理KuaiRec数据

```python
# 下载KuaiRec数据集（需要手动下载，见 KUAIREC_SETUP.md）
# 或者如果已经下载到Google Drive，可以挂载Drive后复制

# 预处理数据
!python3 data_prep_kuairec.py --kuairec_dir "data/KuaiRec 2.0"

# 上传到Google Drive（可选，方便下次使用）
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/youtube-shorts-recommender/data
!cp data/ratings_kuairec.csv /content/drive/MyDrive/youtube-shorts-recommender/data/
!cp data/videos_kuairec.csv /content/drive/MyDrive/youtube-shorts-recommender/data/
```

### 4. 训练模型（使用Colab版本）

```python
!python3 train_kuairec_colab.py
```

## Colab版本特性

`train_kuairec_colab.py` 相比普通版本有以下优势：

1. **自动检测Colab环境**：自动识别是否在Colab中运行
2. **自动从Drive加载数据**：如果本地没有数据，自动从Google Drive复制
3. **自动挂载Google Drive**：训练完成后自动保存模型到Google Drive
4. **GPU自动检测**：自动使用GPU（如果可用），大幅提升训练速度
5. **路径自动修复**：自动处理重复克隆导致的路径嵌套问题
6. **优化的batch size**：根据GPU/CPU自动调整batch size

## 完整工作流程

```python
# 1. 克隆仓库
!git clone https://github.com/wentao2024/youtube-shorts-recommender.git
%cd youtube-shorts-recommender

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 挂载Google Drive（如果需要从Drive读取数据或保存模型）
from google.colab import drive
drive.mount('/content/drive')

# 4. 如果数据在Google Drive，复制到项目目录
# !cp -r "/content/drive/MyDrive/KuaiRec 2.0" data/

# 5. 预处理数据
!python3 data_prep_kuairec.py --kuairec_dir "data/KuaiRec 2.0"

# 6. 训练模型（会自动保存到Google Drive）
!python3 train_kuairec_colab.py
```

## 常见问题

### Q: 路径嵌套问题（多个youtube-shorts-recommender目录）

**A**: `train_kuairec_colab.py` 会自动检测并修复路径问题。如果仍有问题，可以手动：

```python
import os
os.chdir('/content/youtube-shorts-recommender')  # 调整到正确的路径
```

### Q: 找不到数据文件

**A**: 确保：
1. 已经运行 `data_prep_kuairec.py` 预处理数据
2. 数据文件在 `data/` 目录下：
   - `data/ratings_kuairec.csv`
   - `data/videos_kuairec.csv`

### Q: GPU内存不足

**A**: 可以减小batch size，编辑 `train_kuairec_colab.py`：

```python
batch_size = 256 if device == "cuda" else 128  # 减小batch size
```

或者减少epochs：

```python
epochs = 10  # 从15减少到10
```

### Q: 模型文件太大，无法下载

**A**: 模型会自动保存到Google Drive：
- 路径：`/content/drive/MyDrive/youtube-shorts-recommender/models/`
- 可以从Google Drive网页版下载

## 性能优化建议

1. **使用GPU**：Colab免费版提供T4 GPU，训练速度提升10-20倍
2. **使用small_matrix.csv**：如果数据太大，可以使用小数据集：
   ```python
   # 在 data_prep_kuairec.py 中使用 small_matrix.csv 而不是 big_matrix.csv
   ```
3. **减少epochs**：对于快速测试，可以设置 `epochs=5`
4. **使用混合精度训练**：可以进一步优化（需要修改train_advanced.py）

## 训练时间估算

- **CPU训练**：~50-100小时（1250万条数据，15 epochs）
- **GPU训练（T4）**：~5-10小时
- **GPU训练（V100/A100）**：~2-5小时

## 下一步

训练完成后：
1. 模型已保存到 `models/` 和 Google Drive
2. 运行评估：`!python3 evaluate_comparison.py`
3. 比较结果：查看生成的评估报告

