# 修复CSV解析错误

## 问题

读取 `kuairec_caption_category.csv` 时出现解析错误：
```
pandas.errors.ParserError: Error tokenizing data. C error: Buffer overflow caught
```

## 原因

可能的原因：
1. CSV文件中某些字段包含特殊字符（换行符、引号等）
2. 字段分隔符不一致
3. 文件编码问题
4. 某些行格式不正确

## 已实施的修复

### 1. 多层错误处理

脚本现在会尝试多种读取方式：
1. 标准UTF-8读取
2. GBK编码读取
3. Python引擎 + 错误跳过
4. 最宽松的选项

### 2. 兼容性处理

- 支持pandas < 1.3.0（使用`error_bad_lines`）
- 支持pandas >= 1.3.0（使用`on_bad_lines`）

### 3. 优雅降级

如果视频特征文件无法读取：
- ✅ 仍然生成 `ratings_kuairec.csv`
- ⚠️ 跳过 `videos_kuairec.csv` 的创建
- ⚠️ BM25和Cross-Encoder可能无法使用

## 解决方案

### 方案1：手动修复CSV文件

如果文件确实有问题，可以尝试：

```python
# 使用更宽松的方式读取并重新保存
import pandas as pd

# 读取
df = pd.read_csv(
    'data/KuaiRec 2.0/data/kuairec_caption_category.csv',
    engine='python',
    error_bad_lines=False,
    warn_bad_lines=False
)

# 清理数据
df = df.dropna(subset=['video_id'])  # 删除video_id为空的行

# 重新保存
df.to_csv('data/KuaiRec 2.0/data/kuairec_caption_category_fixed.csv', index=False)
```

### 方案2：跳过视频特征（如果不需要BM25/Cross-Encoder）

如果只需要交互数据，可以：
1. 让脚本继续运行（会自动跳过视频特征）
2. 只使用 `ratings_kuairec.csv` 进行训练
3. 不使用BM25和Cross-Encoder

### 方案3：使用其他视频特征文件

如果 `kuairec_caption_category.csv` 有问题，可以尝试：
- `item_categories.csv`
- 或其他可用的特征文件

## 当前状态

脚本已经更新，会：
1. ✅ 自动处理CSV读取错误
2. ✅ 如果视频特征读取失败，继续处理其他数据
3. ✅ 至少生成 `ratings_kuairec.csv`

## 下一步

重新运行预处理：

```bash
python3 data_prep_kuairec.py --kuairec_dir "data/KuaiRec 2.0"
```

即使视频特征读取失败，也会生成：
- ✅ `ratings_kuairec.csv` - 交互数据（最重要）
- ⚠️ `videos_kuairec.csv` - 可能缺失（如果读取失败）

有了 `ratings_kuairec.csv`，就可以：
- 训练传统CF模型
- 训练Two-Tower模型
- 评估推荐系统

只是BM25和Cross-Encoder需要视频特征才能工作。




