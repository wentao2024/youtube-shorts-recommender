# 从GitHub下载KuaiRec

## GitHub仓库

**仓库地址**：https://github.com/chongminggao/KuaiRec

## 快速开始

### 方法1：使用脚本自动下载

```bash
python3 data_prep_kuairec.py --download_github
```

脚本会自动：
1. 使用git clone下载仓库
2. 检查是否包含数据文件
3. 如果找到数据文件，直接使用
4. 如果没有，提供其他下载选项

### 方法2：手动克隆

```bash
cd data
git clone https://github.com/chongminggao/KuaiRec.git
```

然后运行预处理：
```bash
python3 data_prep_kuairec.py --kuairec_dir data/KuaiRec
```

## 重要提示

### ⚠️ GitHub可能不包含完整数据集

根据GitHub仓库的常见情况：

1. **可能只包含代码**：
   - 数据文件可能太大，无法直接存储在GitHub
   - 可能只包含示例数据或部分数据

2. **完整数据集位置**：
   - 完整数据集通常在官网：https://kuairec.com/
   - 或通过其他方式提供（Google Drive、百度网盘等）

3. **检查文件**：
   克隆后检查是否有以下文件：
   ```bash
   ls data/KuaiRec/*.csv
   # 应该看到：
   # - big_matrix.csv (必需)
   # - small_matrix.csv (可选)
   # - 其他特征文件
   ```

## 如果GitHub没有完整数据

如果GitHub仓库不包含完整数据集（big_matrix.csv等），需要：

1. **访问官网下载**：
   - https://kuairec.com/
   - 注册并下载完整数据集

2. **检查README**：
   - GitHub仓库的README可能包含数据下载链接
   - 查看是否有Google Drive或其他下载方式

3. **使用GitHub的代码**：
   - 即使没有数据，GitHub仓库也包含有用的代码
   - 可以参考 `loaddata.py` 了解数据格式

## 验证下载

运行预处理脚本验证：

```bash
python3 data_prep_kuairec.py --kuairec_dir data/KuaiRec
```

如果成功，会看到：
```
Loading big_matrix.csv (fully observed, 99.6% density)...
Matrix shape: (1411, 3327)
...
✓ Saved ratings to data/ratings_kuairec.csv
```

如果失败，会显示详细的错误信息和下载指导。

## 推荐流程

1. **先尝试GitHub**（快速）：
   ```bash
   python3 data_prep_kuairec.py --download_github
   ```

2. **如果GitHub没有完整数据**，从官网下载：
   - 访问 https://kuairec.com/
   - 下载完整数据集
   - 解压到 `data/kuairec/`

3. **运行预处理**：
   ```bash
   python3 data_prep_kuairec.py --kuairec_dir data/kuairec
   ```




