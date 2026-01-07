"""
Advanced Model Training Script
训练高级模型：Two-Tower, BM25
"""
import pandas as pd
from pathlib import Path
import sys
from typing import Optional

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.two_tower_recall import TwoTowerRecall
from src.models.bm25_ranking import BM25Ranker


def train_two_tower(
    ratings_path: Path,
    videos_path: Optional[Path],
    model_path: Path,
    epochs: int = 10,
    batch_size: int = 256,
    device: str = "cpu"
):
    """训练双塔模型"""
    print("=" * 60)
    print("Training Two-Tower Model")
    print("=" * 60)
    
    two_tower = TwoTowerRecall(ratings_path, videos_path, device=device)
    
    print(f"Training on {len(two_tower.ratings)} interactions")
    print(f"Users: {len(two_tower.user_to_idx)}")
    print(f"Videos: {len(two_tower.video_to_idx)}")
    
    two_tower.train(
        epochs=epochs,
        batch_size=batch_size,
        save_path=model_path
    )
    
    print(f"\n✅ Two-Tower model training completed!")
    print(f"Model saved to: {model_path}")


def train_bm25(
    ratings_path: Path,
    videos_path: Optional[Path],
    model_path: Path
):
    """训练BM25模型（实际上是构建索引）"""
    print("=" * 60)
    print("Building BM25 Index")
    print("=" * 60)
    
    bm25_ranker = BM25Ranker(ratings_path, videos_path, model_path)
    
    print(f"\n✅ BM25 index built successfully!")
    print(f"Model saved to: {model_path}")


def main():
    """主函数"""
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 检查数据文件
    ratings_path = data_dir / "ratings.csv"
    videos_path = data_dir / "videos.csv"
    
    if not ratings_path.exists():
        print(f"Error: {ratings_path} not found!")
        print("Please run data preprocessing first.")
        return
    
    if not videos_path.exists():
        print(f"Warning: {videos_path} not found. Using minimal video metadata.")
        videos_path = None
    
    # 训练Two-Tower模型（可选，如果遇到问题可以跳过）
    print("\n" + "=" * 60)
    print("Step 1: Training Two-Tower Model (Optional)")
    print("=" * 60)
    print("Note: Two-Tower training can be unstable. If it fails,")
    print("      you can skip it and only use BM25 for advanced recall.")
    print("=" * 60)
    
    skip_two_tower = input("\nSkip Two-Tower training? (y/n, default=n): ").strip().lower() == 'y'
    
    if not skip_two_tower:
        two_tower_path = models_dir / "two_tower_model.pth"
        
        try:
            train_two_tower(
                ratings_path,
                videos_path,
                two_tower_path,
                epochs=3,  # 减少epochs，避免训练时间过长
                batch_size=64,  # 减小batch size，提高稳定性
                device="cpu"  # 如果有GPU可以改为"cuda"
            )
        except Exception as e:
            print(f"Error training Two-Tower model: {e}")
            print("Continuing with BM25...")
    else:
        print("Skipping Two-Tower training. Only BM25 will be trained.")
    
    # 训练BM25模型
    print("\n" + "=" * 60)
    print("Step 2: Building BM25 Index")
    print("=" * 60)
    bm25_path = models_dir / "bm25_model.pkl"
    
    try:
        train_bm25(ratings_path, videos_path, bm25_path)
    except Exception as e:
        print(f"Error building BM25 index: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Advanced Model Training Complete!")
    print("=" * 60)
    print("\nYou can now use these models in the recall system.")
    print("Set use_advanced_models=True when initializing MultiRecallSystem.")


if __name__ == "__main__":
    main()

