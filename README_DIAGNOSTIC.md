# 推荐系统诊断工具

## 功能

`diagnostic_tool.py` 是一个全面的推荐系统诊断工具，用于快速定位性能问题和数据质量问题。

## 使用方法

```bash
python3 diagnostic_tool.py
```

## 诊断内容

### 1. 数据质量检查
- 用户数、物品数、评分数统计
- 数据稀疏性分析
- 评分分布检查
- 数据量是否足够训练深度学习

### 2. 稀疏性分析
- 用户活跃度分布（均值、中位数、最值）
- 物品流行度分布
- 低活跃用户比例

### 3. 冷启动分析
- 用户冷启动比例（<10个评分）
- 物品冷启动比例（<5个评分）
- 冷启动风险评估

### 4. 热门偏差检查
- 基尼系数计算（衡量分布不均程度）
- Top 20%物品占交互比例
- 长尾分布分析

### 5. 深度学习适用性检查
- 数据量是否足够（推荐>100K）
- 用户/物品规模是否足够（推荐>1000）
- 每用户平均评分数（推荐>20）
- 是否适合使用深度学习模型

## 输出

### 文本诊断报告
- 详细的数据统计
- 发现的问题列表
- 快速修复建议

### 可视化报告（可选）
如果安装了matplotlib，会生成：
- `diagnostics_report.png`：包含4个可视化图表
  1. 用户活跃度分布
  2. 物品流行度分布
  3. 评分分布
  4. 长尾分布（物品流行度）

## 示例输出

```
======================================================================
RECOMMENDER SYSTEM DIAGNOSTICS
======================================================================

1. Data Quality Check
----------------------------------------------------------------------
Users: 943
Items: 1,682
Ratings: 100,000
Sparsity: 0.9370
  ⚠ Low data volume (100,000 ratings). Deep learning may underperform.

Rating distribution:
  1: 6,110 (6.1%)
  2: 11,393 (11.4%)
  3: 27,145 (27.1%)
  4: 34,174 (34.2%)
  5: 21,178 (21.2%)

...

======================================================================
DIAGNOSTIC SUMMARY
======================================================================
⚠ Found 3 issues:
  1. ⚠ Low data volume (100,000 ratings). Deep learning may underperform.
  2. ⚠ High user cold-start: 45.2% users
  3. Deep learning may underperform due to limited data

======================================================================
QUICK FIX SUGGESTIONS
======================================================================

1. Use data augmentation:
   - Implicit feedback (views, clicks)
   - User demographics if available
   - Item metadata

2. Enhance cold-start handling:
   - Increase BM25 weight for new users
   - Use popularity-based fallback
   - Add content-based features

4. Optimize deep learning approach:
   - Start with ensemble (70% CF + 30% DL)
   - Pre-train on larger dataset if available
   - Use simpler model architecture
   - Focus on feature engineering
```

## 依赖

### 必需
- pandas
- numpy

### 可选（用于可视化）
- matplotlib
- seaborn

如果没有安装matplotlib，工具仍然可以运行，只是不会生成可视化图表。

## 常见问题诊断

### 为什么深度学习表现不如传统方法？

诊断工具会检查：
1. **数据量不足**：MovieLens-100K只有100K评分，可能不足以训练复杂的深度学习模型
2. **用户活跃度低**：平均每用户评分数少，难以学习有效的用户表示
3. **冷启动问题**：大量新用户/新物品，深度学习难以处理

**建议**：
- 使用传统CF（SVD）作为基础
- 深度学习作为增强（ensemble）
- 使用BM25和内容特征处理冷启动

### 如何解读诊断结果？

- **✓** 表示正常，没有问题
- **⚠** 表示警告，可能影响性能
- **✗** 表示严重问题，需要修复

### 修复建议

根据诊断结果，工具会自动提供针对性的修复建议：
- 数据量不足 → 数据增强建议
- 冷启动问题 → 内容特征建议
- 热门偏差 → 多样性排序建议
- 深度学习适用性 → 模型选择建议

## 集成到工作流

建议在以下情况运行诊断：

1. **项目开始前**：了解数据集特性
2. **性能问题排查**：快速定位问题
3. **模型选择**：判断是否适合深度学习
4. **系统优化**：找到改进方向



