# KuaiRec数据集设置指南

## 数据集介绍

KuaiRec是一个高稠密度的短视频推荐系统数据集，由快手与中国科学技术大学合作发布。

### 数据集特点

- **高稠密度**：99.6%的稠密度（big_matrix），几乎所有用户都观看了所有视频
- **全观测数据**：1,411名用户对3,327个视频的完整交互记录
- **真实数据**：来自快手的真实短视频推荐场景
- **丰富特征**：包含用户特征和视频特征

### 数据集规模

- **用户数**：1,411
- **视频数**：3,327
- **交互数**：约470万（big_matrix）
- **稠密度**：99.6%（big_matrix）或稀疏（small_matrix）

## 下载数据集

1. 访问KuaiRec官网：https://kuairec.com/
2. 注册并下载数据集
3. 解压到项目目录

## 数据预处理

### 方法1：使用预处理脚本

```bash
# 如果数据集在 data/kuairec/ 目录
python3 data_prep_kuairec.py --kuairec_dir data/kuairec

# 使用small_matrix（稀疏数据，更接近真实场景）
python3 data_prep_kuairec.py --kuairec_dir data/kuairec --use_small_matrix
```

### 方法2：手动处理

如果数据集格式不同，可以手动调整 `data_prep_kuairec.py` 中的加载逻辑。

## 预期文件结构

```
data/
├── kuairec/
│   ├── big_matrix.csv          # 全观测交互矩阵（99.6%稠密度）
│   ├── small_matrix.csv        # 稀疏交互矩阵
│   ├── user_features.csv       # 用户特征
│   ├── item_features.csv       # 视频特征
│   └── ...
├── ratings_kuairec.csv         # 处理后的交互数据
├── videos_kuairec.csv          # 处理后的视频特征
└── user_features_kuairec.csv  # 处理后的用户特征
```

## 使用KuaiRec数据

### 1. 运行诊断工具

```bash
# 修改diagnostic_tool.py使用ratings_kuairec.csv
python3 diagnostic_tool.py
```

### 2. 训练模型

修改 `train_advanced.py` 或创建新脚本：

```python
from data_prep_kuairec import prepare_kuairec

# 准备数据
ratings_path = prepare_kuairec(
    kuairec_dir=Path("data/kuairec"),
    output_dir=Path("data"),
    use_big_matrix=True
)

# 训练模型
# ... 使用ratings_kuairec.csv和videos_kuairec.csv
```

### 3. 评估系统

修改 `evaluate_comparison.py` 使用KuaiRec数据：

```python
# 在main()函数中添加
kuairec_ratings = data_dir / "ratings_kuairec.csv"
kuairec_videos = data_dir / "videos_kuairec.csv"
if kuairec_ratings.exists():
    datasets.append(("KuaiRec", kuairec_ratings, kuairec_videos))
```

## KuaiRec vs MovieLens-100K

| 特性 | MovieLens-100K | KuaiRec |
|------|----------------|---------|
| 用户数 | 943 | 1,411 |
| 物品数 | 1,682 | 3,327 |
| 交互数 | 100,000 | ~4,700,000 |
| 稠密度 | 6.3% | 99.6% (big) |
| 数据来源 | 电影评分 | 短视频交互 |
| 适用场景 | 稀疏推荐 | 全观测研究 |

## 优势

1. **高稠密度**：适合研究全观测推荐系统
2. **真实场景**：来自真实短视频平台
3. **丰富特征**：包含用户和视频的详细特征
4. **更大规模**：比MovieLens-100K更大，更适合深度学习

## 注意事项

1. **big_matrix vs small_matrix**：
   - `big_matrix`：全观测数据，99.6%稠密度，适合研究无偏推荐
   - `small_matrix`：稀疏数据，更接近真实推荐场景

2. **数据格式**：
   - KuaiRec的交互矩阵格式可能与脚本预期不同
   - 可能需要根据实际数据格式调整加载逻辑

3. **特征字段**：
   - 视频特征字段名称可能与预期不同
   - 需要检查并调整 `create_videos_csv_from_features` 函数

## 参考资源

- **官网**：https://kuairec.com/
- **论文**：https://arxiv.org/abs/2202.10842
- **GitHub**：可能在官网提供

## 下一步

1. 下载KuaiRec数据集
2. 运行 `data_prep_kuairec.py` 预处理数据
3. 运行 `diagnostic_tool.py` 分析数据特性
4. 训练和评估推荐系统



