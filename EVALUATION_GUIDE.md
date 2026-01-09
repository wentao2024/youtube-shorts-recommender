# System Comparison Evaluation Guide

本指南说明如何对比评估传统系统和高级系统的性能。

## 📋 概述

本项目支持两种推荐系统：

1. **传统系统** (Traditional System)
   - 基于协同过滤 (SVD)
   - 多路召回：CF, Popularity, High Rating, User Similarity
   - 适用于 MovieLens-100K 数据集

2. **高级系统** (Advanced System)
   - 深度学习：Two-Tower 模型
   - 文本检索：BM25
   - 精排：Cross-Encoder
   - 适用于 MicroLens-100K 数据集

## 🚀 快速开始

### 步骤 1: 准备数据

#### MovieLens 数据（传统系统）
```bash
python data_prep.py
```

#### MicroLens 数据（高级系统）
```bash
# 首先下载 MicroLens 数据集
# 从 https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/ 下载
# 解压到 data/microlens-100k/ 目录

python data_prep_microlens.py
```

### 步骤 2: 训练模型

#### 训练传统系统
```bash
python train.py
# 这会训练 SVD 模型并保存到 models/svd_model.pkl
```

#### 训练高级系统
```bash
python train_advanced.py
# 这会训练：
# - Two-Tower 模型 (models/two_tower_model.pth)
# - BM25 索引 (models/bm25_model.pkl)
```

### 步骤 3: 运行对比评估

```bash
python evaluate_comparison.py
```

这个脚本会：
1. 在测试集上评估传统系统
2. 在测试集上评估高级系统
3. 计算评估指标
4. 生成对比报告
5. 保存结果到 CSV 文件

## 📊 评估指标说明

### NDCG@10 (Normalized Discounted Cumulative Gain)
- **含义**: 归一化折损累积增益，衡量推荐列表的质量
- **范围**: 0-1，越高越好
- **计算**: 考虑位置权重的相关性分数

### Recall@10
- **含义**: 在前10个推荐中找到的相关物品比例
- **范围**: 0-1，越高越好
- **计算**: 相关物品数 / 总相关物品数

### Precision@10
- **含义**: 前10个推荐中相关物品的比例
- **范围**: 0-1，越高越好
- **计算**: 相关物品数 / 10

### Coverage
- **含义**: 推荐系统覆盖的物品比例
- **范围**: 0-1，越高越好
- **计算**: 被推荐过的物品数 / 总物品数

### Diversity
- **含义**: 推荐列表的多样性
- **范围**: 0-1，越高越多样
- **计算**: 推荐物品之间的平均差异度

## 📈 结果解读

评估完成后，会生成对比报告：

```
Metric          Traditional  Advanced  Improvement  Improvement %
NDCG@10         0.xxxx      0.xxxx    +0.xxxx      +xx.xx%
Recall@10       0.xxxx      0.xxxx    +0.xxxx      +xx.xx%
Precision@10    0.xxxx      0.xxxx    +0.xxxx      +xx.xx%
Coverage        0.xxxx      0.xxxx    +0.xxxx      +xx.xx%
Diversity       0.xxxx      0.xxxx    +0.xxxx      +xx.xx%
```

### 如何解读

- **Improvement**: 高级系统相对于传统系统的绝对提升
- **Improvement %**: 相对提升百分比
- **正值**: 高级系统更好
- **负值**: 传统系统更好

## 🔧 自定义评估

### 修改评估参数

编辑 `evaluate_comparison.py`:

```python
# 修改测试集比例
test_ratio = 0.2  # 20% 作为测试集

# 修改评估的 top-k
k = 10  # 评估 top-10

# 修改测试用户数量
num_test_users = 100  # 评估100个用户
```

### 只评估单个系统

```python
# 只评估传统系统
metrics, recs = evaluate_system(
    "MovieLens-100K",
    ratings_path,
    videos_path,
    use_advanced=False,
    k=10
)

# 只评估高级系统
metrics, recs = evaluate_system(
    "MicroLens-100K",
    ratings_path,
    videos_path,
    use_advanced=True,
    k=10
)
```

## 📝 输出文件

评估完成后会生成：

- `data/comparison_results_MovieLens_100K.csv`: 对比结果表格
- 控制台输出：详细的评估过程和信息

## ⚠️ 注意事项

1. **数据要求**: 确保两个数据集都已正确预处理
2. **模型要求**: 高级系统需要先训练 Two-Tower 和 BM25 模型
3. **内存要求**: 高级系统需要更多内存（特别是 Two-Tower 模型）
4. **时间要求**: 高级系统评估时间更长（深度学习模型推理）

## 🐛 故障排除

### 问题: 找不到高级模型
```
Error: Two-Tower model not found
```
**解决**: 运行 `python train_advanced.py` 训练模型

### 问题: 内存不足
```
Error: CUDA out of memory
```
**解决**: 
- 减少 `num_test_users`
- 使用 CPU 模式（device="cpu"）
- 减少 batch_size

### 问题: 数据格式错误
```
Error: Column not found
```
**解决**: 检查数据预处理是否正确完成

## 📚 更多信息

- 查看 `src/evaluation/metrics.py` 了解指标计算细节
- 查看 `src/models/` 了解模型实现
- 查看 API 文档了解如何使用推荐服务




