"""
训练所有高级模型
包括 BM25 和 Two-Tower
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.bm25_ranking import BM25Ranker
from train_two_tower_fixed import train_two_tower_stable


def main():
    """训练所有模型"""
    print("=" * 60)
    print("Training All Advanced Models")
    print("=" * 60)
    
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    ratings_path = data_dir / "ratings.csv"
    videos_path = data_dir / "videos.csv"
    
    if not ratings_path.exists():
        print(f"Error: {ratings_path} not found!")
        return
    
    # 创建videos.csv（如果不存在）
    if not videos_path.exists():
        print("\nCreating videos.csv...")
        try:
            from create_videos_csv import create_videos_csv
            create_videos_csv()
        except Exception as e:
            print(f"Warning: Could not create videos.csv: {e}")
            videos_path = None
    
    # 1. 训练 BM25
    print("\n" + "=" * 60)
    print("Step 1: Training BM25 Model")
    print("=" * 60)
    bm25_path = models_dir / "bm25_model.pkl"
    
    try:
        bm25_ranker = BM25Ranker(ratings_path, videos_path, bm25_path)
        print(f"✅ BM25 model saved to: {bm25_path}")
    except Exception as e:
        print(f"❌ Error training BM25: {e}")
        return
    
    # 2. 训练 Two-Tower
    print("\n" + "=" * 60)
    print("Step 2: Training Two-Tower Model")
    print("=" * 60)
    two_tower_path = models_dir / "two_tower_model.pth"
    
    # 检查PyTorch是否可用
    try:
        import torch
        print("✓ PyTorch is available")
    except (ImportError, OSError) as e:
        print(f"❌ PyTorch not available: {e}")
        print("\nSkipping Two-Tower training.")
        print("BM25 model is ready to use.")
        return
    
    success = train_two_tower_stable(
        ratings_path,
        videos_path if videos_path and videos_path.exists() else None,
        two_tower_path,
        epochs=2,
        batch_size=32,
        learning_rate=0.00005,
        device="cpu"
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All Models Trained Successfully!")
        print("=" * 60)
        print("\nAvailable models:")
        print(f"  ✓ BM25: {bm25_path}")
        print(f"  ✓ Two-Tower: {two_tower_path}")
        print("\nYou can now run:")
        print("  python3 evaluate_advanced.py")
    else:
        print("\n" + "=" * 60)
        print("⚠️  Partial Success")
        print("=" * 60)
        print("\nAvailable models:")
        print(f"  ✓ BM25: {bm25_path}")
        print(f"  ✗ Two-Tower: Training failed")
        print("\nYou can still use BM25 in the advanced system.")


if __name__ == "__main__":
    main()




