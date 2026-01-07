"""
Colab快速启动脚本
如果数据已经在Google Drive中，直接运行此脚本即可开始训练
"""
# 在Colab中运行此代码块

# 1. 克隆仓库
!git clone https://github.com/wentao2024/youtube-shorts-recommender.git
%cd youtube-shorts-recommender

# 2. 拉取最新代码
!git pull

# 3. 安装依赖
!pip install -r requirements.txt

# 4. 挂载Google Drive（会自动提示授权）
from google.colab import drive
drive.mount('/content/drive')

# 5. 检查数据文件（支持多个位置）
import os
from pathlib import Path

print("Checking for data files in multiple locations...")

# 可能的数据位置
possible_locations = [
    Path("/content/data"),  # 如果已经复制到这里
    Path("/content/drive/MyDrive/AI/youtobe-shorts-data"),  # 用户指定的Drive路径
    Path("/content/drive/MyDrive/youtube-shorts-recommender/data"),  # 默认Drive路径
]

ratings_found = False
videos_found = False

for location in possible_locations:
    ratings_file = location / "ratings_kuairec.csv"
    videos_file = location / "videos_kuairec.csv"
    
    if ratings_file.exists():
        print(f"✓ Found ratings_kuairec.csv in {location}")
        ratings_found = True
    
    if videos_file.exists():
        print(f"✓ Found videos_kuairec.csv in {location}")
        videos_found = True

if not ratings_found:
    print("⚠ ratings_kuairec.csv not found")
    print("  Script will search automatically, or you can copy manually:")
    print("  !cp /content/drive/MyDrive/AI/youtobe-shorts-data/*.csv /content/data/")

if not videos_found:
    print("⚠ videos_kuairec.csv not found (optional, for BM25)")

# 6. 开始训练（会自动从Drive加载数据）
print("\n" + "="*70)
print("Starting training...")
print("="*70)
!python3 train_kuairec_colab.py

