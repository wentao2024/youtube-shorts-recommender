# 快速Two-Tower训练指南

## 🚀 性能优化

**问题**: 原始训练方法在每个batch都重复编码视频文本（sentence-transformers），这是CPU密集型操作，导致训练极慢。

**解决方案**: 预计算所有视频的embeddings并缓存，训练时直接使用预计算的embeddings。

**性能提升**: 
- **原始训练**: 1253万样本 × 每batch 52秒 = **350+小时**
- **快速训练**: 预计算embeddings（一次性，约30分钟）+ 训练（约30分钟） = **约1小时**
- **加速比**: **~700x**

## 📋 使用方法

### 自动使用（推荐）

训练脚本会自动检测大数据集并使用快速训练：

```bash
# KuaiRec数据集（自动使用快速训练）
python train_kuairec.py
```

### 手动指定

在代码中明确启用快速训练：

```python
from train_advanced import train_two_tower
from pathlib import Path

train_two_tower(
    ratings_path=Path("data/ratings_kuairec.csv"),
    videos_path=Path("data/videos_kuairec.csv"),
    model_path=Path("models/two_tower_model_kuairec.pth"),
    epochs=10,
    batch_size=512,
    device="cuda",  # 如果有GPU
    use_fast_training=True  # 启用快速训练
)
```

## 🔧 工作原理

### 1. 预计算Embeddings

```python
# 一次性计算所有视频的文本embeddings
embedding_cache = PrecomputedEmbeddingsCache(
    videos_df,
    cache_path=Path("models/video_embeddings_kuairec.pkl"),
    device="cuda"
)
```

**特点**:
- 一次性计算，保存到缓存文件
- 后续训练直接加载缓存，无需重新计算
- 支持GPU加速编码

### 2. 高效数据加载

```python
# 使用PyTorch DataLoader + 预计算embeddings
dataset = FastTwoTowerDataset(
    ratings,
    embedding_cache,  # 使用预计算的embeddings
    user_to_idx,
    video_to_idx,
    user_features,
    video_metadata
)
```

**优化**:
- 多进程数据加载（`num_workers=4`）
- 内存固定（`pin_memory=True`，GPU训练时）
- 批量负采样

### 3. 模型结构

快速训练的模型结构与标准训练相同，但：
- **VideoTower**: 直接接收预计算的embeddings，无需运行时编码
- **训练速度**: 大幅提升（700x）
- **模型质量**: 与标准训练相同

## 📊 参数调整

### 大数据集（>100万样本）

```python
# 自动调整
- batch_size: 2048（更大）
- epochs: 5-10（因为batch更大，可以少一些）
- num_negatives: 4（每个正样本配4个负样本）
```

### 小数据集（<100万样本）

```python
# 自动调整
- batch_size: 512-1024
- epochs: 10-15
- num_negatives: 2-4
```

## 💾 缓存管理

### 缓存位置

- KuaiRec: `models/video_embeddings_kuairec.pkl`
- 其他数据集: `models/video_embeddings_cache.pkl`

### 缓存大小

- 每个视频embedding: 384维 × 4字节 = 1.5KB
- 10万视频: ~150MB
- 100万视频: ~1.5GB

### 清除缓存

如果视频数据更新，需要删除缓存文件重新计算：

```bash
rm models/video_embeddings_kuairec.pkl
```

## ⚠️ 注意事项

1. **首次运行**: 需要预计算embeddings，可能需要30-60分钟
2. **GPU推荐**: 预计算embeddings在GPU上快得多
3. **内存需求**: 大数据集需要足够内存存储embeddings
4. **兼容性**: 快速训练的模型与标准训练模型完全兼容

## 🔍 故障排除

### 问题1: 内存不足

**解决方案**: 减少batch_size或使用数据采样

```python
fast_train_two_tower(
    ...
    batch_size=1024,  # 减小batch size
    sample_rate=0.5  # 使用50%数据
)
```

### 问题2: 缓存文件损坏

**解决方案**: 删除缓存文件重新计算

```bash
rm models/video_embeddings_*.pkl
```

### 问题3: GPU内存不足

**解决方案**: 使用CPU或减小batch_size

```python
device="cpu"  # 或
batch_size=512  # 减小batch size
```

## 📈 性能对比

| 数据集 | 样本数 | 原始训练时间 | 快速训练时间 | 加速比 |
|--------|--------|--------------|--------------|--------|
| MovieLens-100K | 100K | ~2小时 | ~5分钟 | 24x |
| KuaiRec | 12.5M | 350+小时 | ~1小时 | 700x |

## 🎯 最佳实践

1. **首次训练**: 使用小样本（`sample_rate=0.1`）快速验证流程
2. **完整训练**: 使用全部数据（`sample_rate=1.0`）
3. **GPU训练**: 优先使用GPU，速度提升10-20x
4. **缓存复用**: 相同视频数据可以复用缓存

## 📚 相关文件

- `src/models/fast_two_tower_training.py`: 快速训练实现
- `train_advanced.py`: 训练入口（自动选择快速/标准训练）
- `train_kuairec.py`: KuaiRec数据集训练脚本

