"""
训练KuaiRec数据集的模型 - Colab版本
自动检测Colab环境，使用GPU，保存到Google Drive
"""
import pandas as pd
from pathlib import Path
import sys
import os

# 检测Colab环境
try:
    import google.colab
    IN_COLAB = True
    print("✓ Detected Google Colab environment")
except ImportError:
    IN_COLAB = False

# 处理路径问题（如果重复克隆导致路径嵌套）
current_path = Path.cwd()
if 'youtube-shorts-recommender' in str(current_path):
    # 找到正确的项目根目录
    parts = current_path.parts
    for i, part in enumerate(parts):
        if part == 'youtube-shorts-recommender':
            project_root = Path(*parts[:i+1])
            os.chdir(project_root)
            print(f"✓ Changed to project root: {project_root}")
            break

sys.path.insert(0, str(Path.cwd()))

from train_advanced import train_two_tower, train_bm25


def setup_colab():
    """设置Colab环境"""
    if not IN_COLAB:
        return None, None
    
    # 挂载Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
        
        # 创建保存目录
        drive_models_dir = Path("/content/drive/MyDrive/youtube-shorts-recommender/models")
        drive_models_dir.mkdir(parents=True, exist_ok=True)
        
        return drive_models_dir, "/content/drive/MyDrive/youtube-shorts-recommender"
    except Exception as e:
        print(f"⚠ Could not mount Google Drive: {e}")
        return None, None


def main():
    """训练KuaiRec数据集的模型"""
    print("=" * 70)
    print("Training Models for KuaiRec Dataset (Colab Version)")
    print("=" * 70)
    
    # 设置Colab环境
    drive_models_dir = None
    drive_base_dir = None
    if IN_COLAB:
        drive_models_dir, drive_base_dir = setup_colab()
    
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
    
    # 检查PyTorch和GPU
    device = "cpu"
    torch_available = False
    try:
        import torch
        torch_available = True
        print(f"\n✓ PyTorch available: {torch.__version__}")
        
        # 检测GPU
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ GPU not available, using CPU")
    except (ImportError, OSError) as e:
        print(f"\n⚠ PyTorch not available: {e}")
        print("  Two-Tower training will be skipped")
    
    # 训练BM25
    print("\n" + "=" * 70)
    print("1. Training BM25 Model")
    print("=" * 70)
    
    bm25_model_path = models_dir / "bm25_model_kuairec.pkl"
    if videos_path.exists():
        try:
            train_bm25(ratings_path, videos_path, bm25_model_path)
            print(f"✅ BM25 model saved to: {bm25_model_path}")
            
            # 复制到Google Drive
            if drive_models_dir:
                import shutil
                drive_bm25_path = drive_models_dir / "bm25_model_kuairec.pkl"
                shutil.copy(bm25_model_path, drive_bm25_path)
                print(f"✅ BM25 model also saved to Google Drive: {drive_bm25_path}")
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
            # Colab上使用GPU，调整batch size
            batch_size = 512 if device == "cuda" else 256
            epochs = 15  # 可以根据需要调整
            
            print(f"Training configuration:")
            print(f"  Device: {device}")
            print(f"  Batch size: {batch_size}")
            print(f"  Epochs: {epochs}")
            
            train_two_tower(
                ratings_path=ratings_path,
                videos_path=videos_path if videos_path.exists() else None,
                model_path=two_tower_model_path,
                epochs=epochs,
                batch_size=batch_size,
                device=device
            )
            print(f"✅ Two-Tower model saved to: {two_tower_model_path}")
            
            # 复制到Google Drive
            if drive_models_dir:
                import shutil
                drive_tt_path = drive_models_dir / "two_tower_model_kuairec.pth"
                shutil.copy(two_tower_model_path, drive_tt_path)
                print(f"✅ Two-Tower model also saved to Google Drive: {drive_tt_path}")
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
    
    if drive_models_dir:
        print(f"\n✓ Models also saved to Google Drive: {drive_models_dir}")
    
    print("\nNext steps:")
    print("1. Run evaluation: python3 evaluate_comparison.py")
    print("   (Will automatically detect and evaluate KuaiRec dataset)")
    print("2. Compare with MovieLens-100K results")


if __name__ == "__main__":
    main()

