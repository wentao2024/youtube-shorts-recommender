# KuaiRec快速开始指南

## 快速步骤

### 1. 下载KuaiRec数据集

访问 https://kuairec.com/ 下载数据集，解压到 `data/kuairec/` 目录。

### 2. 预处理数据

```bash
python3 data_prep_kuairec.py --kuairec_dir data/kuairec
```

这会生成：
- `data/ratings_kuairec.csv` - 交互数据
- `data/videos_kuairec.csv` - 视频特征（如果有）
- `data/user_features_kuairec.csv` - 用户特征（如果有）

### 3. 运行诊断工具

```bash
# 修改diagnostic_tool.py中的ratings_path为ratings_kuairec.csv
# 或直接运行（如果已修改）
python3 diagnostic_tool.py
```

### 4. 训练模型

```bash
# 训练所有模型（会使用KuaiRec数据）
python3 train_advanced.py
```

注意：需要修改 `train_advanced.py` 使用 `ratings_kuairec.csv` 和 `videos_kuairec.csv`。

### 5. 评估系统

```bash
# evaluate_comparison.py已自动支持KuaiRec
python3 evaluate_comparison.py
```

## 预期优势

KuaiRec相比MovieLens-100K的优势：

1. **更大规模**：
   - 用户：1,411 vs 943
   - 视频：3,327 vs 1,682
   - 交互：~4.7M vs 100K

2. **高稠密度**：
   - big_matrix：99.6%稠密度
   - 适合研究全观测推荐系统

3. **真实场景**：
   - 来自真实短视频平台
   - 更接近实际应用场景

4. **更适合深度学习**：
   - 数据量更大
   - 用户/物品规模更大
   - 每用户平均交互更多

## 注意事项

1. **big_matrix vs small_matrix**：
   - `big_matrix`：全观测数据，适合研究
   - `small_matrix`：稀疏数据，更接近真实场景
   - 使用 `--use_small_matrix` 参数选择

2. **数据格式**：
   - 如果KuaiRec的实际格式与脚本预期不同，需要调整 `data_prep_kuairec.py`

3. **特征字段**：
   - 检查视频特征字段名称
   - 可能需要调整 `create_videos_csv_from_features` 函数

## 故障排除

### 问题1：文件未找到

```
FileNotFoundError: File not found: data/kuairec/big_matrix.csv
```

**解决**：确保数据集已下载并解压到正确位置。

### 问题2：数据格式不匹配

如果KuaiRec的数据格式与预期不同，需要修改 `data_prep_kuairec.py` 中的加载逻辑。

### 问题3：特征字段缺失

如果视频特征中没有 `title` 或 `description` 字段，BM25和Cross-Encoder可能无法正常工作。

## 下一步

1. 运行诊断工具，了解KuaiRec数据特性
2. 对比KuaiRec和MovieLens-100K的诊断结果
3. 训练模型并评估性能
4. 分析为什么KuaiRec可能更适合深度学习



