"""
训练KuaiRec数据集的模型
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from train_advanced import train_two_tower, train_bm25


def main():
    """训练KuaiRec数据集的模型"""
    print("=" * 70)
    print("Training Models for KuaiRec Dataset")
    print("=" * 70)
    
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # KuaiRec数据路径
    ratings_path = data_dir / "ratings_kuairec.csv"
    videos_path = data_dir / "videos_kuairec.csv"
    
    if not ratings_path.exists():
        print(f"Error: {ratings_path} not found!")
        print("Please run: python3 data_prep_kuairec.py --kuairec_dir 'data/KuaiRec 2.0'")
        return
    
    print(f"\nUsing data:")
    print(f"  Ratings: {ratings_path}")
    print(f"  Videos: {videos_path if videos_path.exists() else 'Not available'}")
    
    # 检查PyTorch
    try:
        import torch
        torch_available = True
        print(f"\n✓ PyTorch available: {torch.__version__}")
    except (ImportError, OSError) as e:
        torch_available = False
        print(f"\n⚠ PyTorch not available: {e}")
        print("  Two-Tower training will be skipped")
        print("  See FIX_PYTORCH.md for installation instructions")
    
    # 训练BM25
    print("\n" + "=" * 70)
    print("1. Training BM25 Model")
    print("=" * 70)
    
    bm25_model_path = models_dir / "bm25_model_kuairec.pkl"
    if videos_path.exists():
        try:
            train_bm25(ratings_path, videos_path, bm25_model_path)
            print(f"✅ BM25 model saved to: {bm25_model_path}")
        except Exception as e:
            print(f"❌ BM25 training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠ Videos file not found, skipping BM25 training")
        print("  BM25 requires video titles/descriptions")
    
    # 训练Two-Tower
    if torch_available:
        print("\n" + "=" * 70)
        print("2. Training Two-Tower Model")
        print("=" * 70)
        
        two_tower_model_path = models_dir / "two_tower_model_kuairec.pth"
        
        try:
            # 检测GPU
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
                else:
                    print("  ⚠ GPU not available, using CPU (will be slower)")
            except:
                device = "cpu"
            
            # KuaiRec数据量大，使用快速训练（预计算embeddings）
            # 快速训练会自动调整batch size和epochs
            train_two_tower(
                ratings_path=ratings_path,
                videos_path=videos_path if videos_path.exists() else None,
                model_path=two_tower_model_path,
                epochs=10,  # 快速训练使用更大的batch，所以epochs可以少一些
                batch_size=512,  # 快速训练会自动增大batch size
                device=device,
                use_fast_training=True  # 启用快速训练（700x加速！）
            )
            print(f"✅ Two-Tower model saved to: {two_tower_model_path}")
        except Exception as e:
            print(f"❌ Two-Tower training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ Two-Tower training skipped (PyTorch not available)")
    
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    if bm25_model_path.exists():
        print(f"✓ BM25 model: {bm25_model_path}")
    if torch_available and (models_dir / "two_tower_model_kuairec.pth").exists():
        print(f"✓ Two-Tower model: {two_tower_model_path}")
    
    print("\nNext steps:")
    print("1. Run evaluation: python3 evaluate_comparison.py")
    print("   (Will automatically detect and evaluate KuaiRec dataset)")
    print("2. Compare with MovieLens-100K results")


if __name__ == "__main__":
    main()




