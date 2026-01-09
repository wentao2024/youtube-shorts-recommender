"""
MicroLens-100K Dataset Preprocessing
专门为短视频推荐设计的数据集
"""
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import ssl
import zipfile
import os
from typing import Optional

def download_microlens_100k(data_dir: Path = Path("data")):
    """下载 MicroLens-100K 数据集"""
    data_dir.mkdir(exist_ok=True)
    microlens_dir = data_dir / "microlens-100k"
    microlens_dir.mkdir(exist_ok=True)
    
    url = "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/"
    
    print("=" * 60)
    print("MicroLens-100K Dataset Download")
    print("=" * 60)
    print("Please download MicroLens-100K from:")
    print("https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/")
    print(f"Then extract to: {microlens_dir}")
    print("=" * 60)
    
    return microlens_dir

def preprocess_microlens(
    microlens_dir: Path,
    output_path: Path = Path("data/ratings.csv"),
    videos_output_path: Path = Path("data/videos.csv")
):
    """
    预处理 MicroLens 数据
    
    MicroLens 数据格式（根据官方文档）:
    - interactions.csv: user_id, video_id, rating, timestamp
    - videos.csv: video_id, title, description, etc.
    - users.csv: user_id, features, etc.
    """
    interactions_path = microlens_dir / "interactions.csv"
    
    if not interactions_path.exists():
        print(f"Error: {interactions_path} not found!")
        print("Please ensure you have downloaded and extracted the MicroLens dataset.")
        return None
    
    print("Loading interactions data...")
    interactions = pd.read_csv(interactions_path)
    
    # 检查列名（MicroLens可能有不同的列名）
    print(f"Columns in interactions: {interactions.columns.tolist()}")
    
    # 尝试识别列名
    user_col = None
    video_col = None
    rating_col = None
    
    for col in interactions.columns:
        col_lower = col.lower()
        if 'user' in col_lower:
            user_col = col
        elif 'video' in col_lower or 'item' in col_lower:
            video_col = col
        elif 'rating' in col_lower or 'score' in col_lower:
            rating_col = col
    
    # 如果没有找到，使用前几列
    if user_col is None:
        user_col = interactions.columns[0]
    if video_col is None:
        video_col = interactions.columns[1]
    
    # 重命名列以匹配现有代码
    ratings = interactions.rename(columns={
        user_col: 'user_id',
        video_col: 'video_id',
    })
    
    # 处理评分
    if rating_col:
        ratings['rating'] = interactions[rating_col]
    else:
        # 如果没有显式评分，使用观看时长或点赞作为隐式评分
        if 'watch_time' in interactions.columns:
            # 将观看时长转换为评分（归一化到1-5）
            watch_times = interactions['watch_time']
            ratings['rating'] = pd.cut(
                watch_times,
                bins=5,
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
        elif 'like' in interactions.columns:
            ratings['rating'] = interactions['like'].map({0: 1, 1: 5})
        else:
            # 默认隐式反馈为3分
            ratings['rating'] = 3
    
    # 确保评分在1-5范围内
    ratings['rating'] = ratings['rating'].clip(1, 5)
    
    # 保存处理后的数据
    ratings[['user_id', 'video_id', 'rating']].to_csv(
        output_path, index=False
    )
    
    # 处理视频元数据
    videos_path = microlens_dir / "videos.csv"
    if videos_path.exists():
        print("Loading videos metadata...")
        videos = pd.read_csv(videos_path)
        
        # 尝试识别video_id列
        video_id_col = None
        for col in videos.columns:
            if 'video' in col.lower() or 'item' in col.lower() or 'id' in col.lower():
                video_id_col = col
                break
        
        if video_id_col:
            videos = videos.rename(columns={video_id_col: 'video_id'})
        
        videos.to_csv(videos_output_path, index=False)
        print(f"Saved videos metadata to {videos_output_path}")
    else:
        print(f"Warning: {videos_path} not found. Creating minimal videos.csv...")
        # 创建最小的视频元数据
        unique_videos = ratings['video_id'].unique()
        videos = pd.DataFrame({
            'video_id': unique_videos,
            'title': [f"Video {vid}" for vid in unique_videos],
            'description': [""] * len(unique_videos)
        })
        videos.to_csv(videos_output_path, index=False)
    
    print(f"\n{'='*60}")
    print("Data Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(ratings)} interactions")
    print(f"Users: {ratings['user_id'].nunique()}")
    print(f"Videos: {ratings['video_id'].nunique()}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    print(f"Rating distribution:")
    print(ratings['rating'].value_counts().sort_index())
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}")
    
    return ratings

def main():
    """主函数"""
    data_dir = Path("data")
    microlens_dir = download_microlens_100k(data_dir)
    
    # 检查数据是否存在
    interactions_path = microlens_dir / "interactions.csv"
    if not interactions_path.exists():
        print("\n⚠️  Please download and extract MicroLens-100K dataset first!")
        print("   Download from: https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/")
        print(f"   Extract to: {microlens_dir}")
        return
    
    # 预处理数据
    ratings = preprocess_microlens(
        microlens_dir,
        output_path=data_dir / "ratings.csv",
        videos_output_path=data_dir / "videos.csv"
    )
    
    if ratings is not None:
        print("\n✅ Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()





