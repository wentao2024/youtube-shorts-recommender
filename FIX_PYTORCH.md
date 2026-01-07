# 修复 PyTorch 架构不兼容问题

## 问题说明

错误信息显示：
```
mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64')
```

这是因为：
- 您的 Mac 是 Apple Silicon (M1/M2/M3) 芯片，需要 **arm64** 架构
- 当前安装的 PyTorch 是 **x86_64** 架构（Intel 版本）

## 解决方案

### 方法 1: 重新安装 PyTorch（推荐）

1. **卸载当前的 PyTorch**：
```bash
pip3 uninstall torch torchvision torchaudio
```

2. **安装 Apple Silicon 版本的 PyTorch**：
```bash
# 使用 pip 安装（推荐）
pip3 install torch torchvision torchaudio
```

或者使用 conda（如果您使用 conda）：
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

3. **验证安装**：
```bash
python3 -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

应该看到：
- PyTorch 版本号
- `True`（表示可以使用 Apple 的 Metal Performance Shaders）

### 方法 2: 使用虚拟环境（更安全）

```bash
# 创建新的虚拟环境
python3 -m venv venv_arm64

# 激活虚拟环境
source venv_arm64/bin/activate

# 安装依赖（会自动安装正确架构的 PyTorch）
pip install -r requirements.txt
```

### 方法 3: 暂时禁用高级模型（如果不需要）

如果暂时不需要使用 Two-Tower 模型，代码已经做了优雅降级处理：

```python
# 在 evaluate_comparison.py 中，可以设置
use_advanced = False  # 只使用传统系统
```

## 验证修复

运行以下命令验证：

```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
```

如果成功，应该看到版本号，而不是错误。

## 如果仍然有问题

1. **检查 Python 架构**：
```bash
python3 -c "import platform; print(platform.machine())"
```
应该显示 `arm64`

2. **检查 pip 安装的包架构**：
```bash
pip3 show torch | grep Location
```

3. **完全清理并重新安装**：
```bash
# 卸载所有相关包
pip3 uninstall torch torchvision torchaudio transformers sentence-transformers

# 清理缓存
pip3 cache purge

# 重新安装
pip3 install torch torchvision torchaudio
pip3 install transformers sentence-transformers
```

## 注意事项

- Apple Silicon Mac 上，PyTorch 会自动使用 MPS (Metal Performance Shaders) 加速
- 如果遇到其他依赖问题，可能需要重新安装整个 requirements.txt：
  ```bash
  pip3 install --upgrade --force-reinstall -r requirements.txt
  ```



