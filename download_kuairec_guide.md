# KuaiRec数据集下载指南

## 快速下载步骤

### 方法1：从GitHub下载（快速）

GitHub仓库：https://github.com/chongminggao/KuaiRec

**使用脚本自动下载**：
```bash
python3 data_prep_kuairec.py --download_github
```

**或手动克隆**：
```bash
cd data
git clone https://github.com/chongminggao/KuaiRec.git
```

**注意**：GitHub仓库可能只包含代码和部分数据文件，完整数据集可能需要从官网下载。

### 方法2：从官网下载（完整数据集，推荐）

1. **访问官网**
   ```
   https://kuairec.com/
   ```

2. **注册账号**
   - 点击注册/登录
   - 填写必要信息

3. **下载数据集**
   - 在下载页面选择数据集版本
   - 下载压缩包

4. **解压到项目目录**
   ```bash
   # 假设下载的文件是 kuairec.zip
   unzip kuairec.zip -d data/
   # 或者
   cd data && unzip ../kuairec.zip
   ```

5. **验证文件结构**
   ```bash
   ls data/kuairec/
   # 应该看到：
   # - big_matrix.csv
   # - small_matrix.csv (可选)
   # - item_features.csv (可选)
   # - user_features.csv (可选)
   ```

## 文件说明

### 必需文件

- **big_matrix.csv**: 全观测交互矩阵
  - 99.6%稠密度
  - 包含所有用户-视频交互
  - 格式：CSV，行为用户，列为视频

### 可选文件

- **small_matrix.csv**: 稀疏交互矩阵
  - 更接近真实推荐场景
  - 使用 `--use_small_matrix` 参数

- **item_features.csv**: 视频特征
  - 用于BM25和Cross-Encoder
  - 包含视频标题、描述等

- **user_features.csv**: 用户特征
  - 用户统计信息
  - 可用于特征工程

## 验证下载

运行预处理脚本验证：

```bash
python3 data_prep_kuairec.py --kuairec_dir data/kuairec
```

如果成功，会看到：
```
Loading big_matrix.csv (fully observed, 99.6% density)...
Matrix shape: (1411, 3327)
Users: 1,411, Videos: 3,327
...
✓ Saved ratings to data/ratings_kuairec.csv
```

## 常见问题

### Q1: 找不到下载链接

**A**: 访问 https://kuairec.com/ 查看最新下载说明。可能需要：
- 注册账号
- 填写使用目的
- 等待审核

### Q2: 文件格式不对

**A**: 确保：
- 文件是CSV格式
- 文件名完全匹配（大小写敏感）
- 文件编码为UTF-8

### Q3: 解压后文件结构不对

**A**: 可能需要：
- 检查是否有嵌套目录
- 将文件移动到正确位置
- 确保 `data/kuairec/big_matrix.csv` 存在

### Q4: 下载速度慢

**A**: 可以尝试：
- 使用镜像站点（如果有）
- 使用下载工具（wget, curl）
- 分时段下载

## 替代方案

如果无法下载KuaiRec，可以考虑：

1. **使用MovieLens-1M**
   ```bash
   # 更大的MovieLens数据集
   python3 data_prep.py  # 修改为1M版本
   ```

2. **使用其他短视频数据集**
   - TikTok数据集（如果有公开）
   - YouTube-8M（视频分类，可转换）

3. **继续使用MovieLens-100K**
   - 虽然规模小，但足够演示系统
   - 可以分析为什么深度学习表现不佳

## 下一步

下载完成后：

1. **运行预处理**
   ```bash
   python3 data_prep_kuairec.py --kuairec_dir data/kuairec
   ```

2. **运行诊断**
   ```bash
   # 修改diagnostic_tool.py使用ratings_kuairec.csv
   python3 diagnostic_tool.py
   ```

3. **训练模型**
   ```bash
   python3 train_advanced.py
   ```

4. **评估系统**
   ```bash
   python3 evaluate_comparison.py
   ```

