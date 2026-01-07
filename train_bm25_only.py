"""
只训练 BM25 模型（不需要 PyTorch）
这是一个更简单、更稳定的选择
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.bm25_ranking import BM25Ranker


def main():
    """只训练 BM25 模型"""
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    ratings_path = data_dir / "ratings.csv"
    videos_path = data_dir / "videos.csv"
    bm25_path = models_dir / "bm25_model.pkl"
    
    print("=" * 60)
    print("Training BM25 Model Only")
    print("=" * 60)
    print("\nThis is simpler and more stable than Two-Tower model.")
    print("BM25 doesn't require PyTorch and trains much faster.\n")
    
    try:
        bm25_ranker = BM25Ranker(ratings_path, videos_path, bm25_path)
        print(f"\n✅ BM25 model trained and saved to: {bm25_path}")
        print("\nYou can now use BM25 in the recall system:")
        print("  recall_system = MultiRecallSystem(use_advanced_models=True)")
        print("  # BM25 will be used automatically if available")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



