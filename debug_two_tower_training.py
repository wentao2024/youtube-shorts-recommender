"""
双塔模型训练调试工具
帮助诊断loss不下降的问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models.two_tower_recall import TwoTowerRecall, TwoTowerModel


def check_training_data(ratings_path: Path, videos_path: Optional[Path] = None):
    """检查训练数据的质量"""
    print("=" * 70)
    print("1. Checking Training Data Quality")
    print("=" * 70)
    
    ratings = pd.read_csv(ratings_path)
    print(f"\nRatings shape: {ratings.shape}")
    print(f"Rating columns: {ratings.columns.tolist()}")
    print(f"\nRating distribution:")
    print(ratings['rating'].describe())
    
    # 检查用户和视频数量
    num_users = ratings['user_id'].nunique()
    num_videos = ratings['video_id'].nunique()
    sparsity = 1 - (len(ratings) / (num_users * num_videos))
    
    print(f"\nUsers: {num_users}")
    print(f"Videos: {num_videos}")
    print(f"Interactions: {len(ratings)}")
    print(f"Sparsity: {sparsity:.4f} ({100*sparsity:.2f}%)")
    
    # 检查视频数据
    if videos_path and videos_path.exists():
        videos = pd.read_csv(videos_path)
        print(f"\nVideos shape: {videos.shape}")
        print(f"Video columns: {videos.columns.tolist()}")
        
        # 检查文本字段
        if 'title' in videos.columns:
            empty_titles = videos['title'].isna().sum()
            print(f"Empty titles: {empty_titles}/{len(videos)}")
        
        if 'description' in videos.columns:
            empty_desc = videos['description'].isna().sum()
            print(f"Empty descriptions: {empty_desc}/{len(videos)}")
    
    print("\n✓ Data quality check completed")


def check_embeddings(two_tower_model: TwoTowerRecall):
    """检查文本embeddings的质量"""
    print("\n" + "=" * 70)
    print("2. Checking Text Embeddings Quality")
    print("=" * 70)
    
    if not two_tower_model.text_embeddings:
        print("⚠ Warning: No text embeddings precomputed!")
        print("  Text embeddings will be computed during training")
        return
    
    embeddings = list(two_tower_model.text_embeddings.values())
    embeddings_array = np.array(embeddings)
    
    print(f"\nPrecomputed embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    print(f"Embedding mean: {embeddings_array.mean():.6f}")
    print(f"Embedding std: {embeddings_array.std():.6f}")
    print(f"Embedding min: {embeddings_array.min():.6f}")
    print(f"Embedding max: {embeddings_array.max():.6f}")
    
    # 检查相似性
    if len(embeddings) >= 2:
        from sklearn.metrics.pairwise import cosine_similarity
        sample_size = min(100, len(embeddings))
        sample_embeds = embeddings_array[:sample_size]
        similarities = cosine_similarity(sample_embeds)
        
        # 移除对角线（自相似性）
        np.fill_diagonal(similarities, 0)
        
        print(f"\nEmbedding similarity statistics (sample of {sample_size}):")
        print(f"  Mean similarity: {similarities.mean():.6f}")
        print(f"  Max similarity: {similarities.max():.6f}")
        print(f"  Min similarity: {similarities.min():.6f}")
    
    print("\n✓ Embeddings quality check completed")


def test_forward_pass(two_tower_model: TwoTowerRecall, batch_size: int = 32):
    """测试模型前向传播"""
    print("\n" + "=" * 70)
    print("3. Testing Forward Pass")
    print("=" * 70)
    
    # 创建临时模型
    num_users = len(two_tower_model.user_to_idx)
    device = two_tower_model.device
    model = TwoTowerModel(num_users).to(device)
    model.eval()
    
    # 随机采样
    all_users = list(two_tower_model.user_to_idx.keys())
    all_videos = list(two_tower_model.video_to_idx.keys())
    
    sample_users = np.random.choice(all_users, min(batch_size, len(all_users)), replace=False)
    sample_videos = np.random.choice(all_videos, min(batch_size, len(all_videos)), replace=False)
    
    # 获取索引
    user_idxs = torch.tensor([two_tower_model.user_to_idx[u] for u in sample_users], 
                             device=device, dtype=torch.long)
    
    # 获取embeddings和metadata
    text_embeds = []
    for vid in sample_videos:
        if vid in two_tower_model.text_embeddings:
            text_embeds.append(two_tower_model.text_embeddings[vid])
        else:
            text_embeds.append(np.zeros(384))
    text_embeds = torch.tensor(np.array(text_embeds), device=device, dtype=torch.float32)
    
    video_metadata = two_tower_model._get_video_metadata(sample_videos)
    user_features = two_tower_model._get_user_features(sample_users)
    
    # 前向传播
    with torch.no_grad():
        try:
            scores, user_embeds, video_embeds = model(
                user_idxs, text_embeds, video_metadata, user_features, device=str(device)
            )
            
            print(f"\nForward pass successful!")
            print(f"  User embeddings shape: {user_embeds.shape}")
            print(f"  Video embeddings shape: {video_embeds.shape}")
            print(f"  Scores shape: {scores.shape}")
            
            # 检查数值
            print(f"\nNumerical checks:")
            print(f"  User embeddings mean: {user_embeds.mean():.6f}")
            print(f"  User embeddings std: {user_embeds.std():.6f}")
            print(f"  Video embeddings mean: {video_embeds.mean():.6f}")
            print(f"  Video embeddings std: {video_embeds.std():.6f}")
            print(f"  Scores mean: {scores.mean():.6f}")
            print(f"  Scores std: {scores.std():.6f}")
            
            # 检查NaN/Inf
            print(f"\nSanity checks:")
            print(f"  User embeddings has NaN: {torch.isnan(user_embeds).any()}")
            print(f"  User embeddings has Inf: {torch.isinf(user_embeds).any()}")
            print(f"  Video embeddings has NaN: {torch.isnan(video_embeds).any()}")
            print(f"  Video embeddings has Inf: {torch.isinf(video_embeds).any()}")
            print(f"  Scores has NaN: {torch.isnan(scores).any()}")
            print(f"  Scores has Inf: {torch.isinf(scores).any()}")
            
            print("\n✓ Forward pass test completed")
            return True
            
        except Exception as e:
            print(f"\n✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_loss_computation(two_tower_model: TwoTowerRecall, batch_size: int = 32):
    """测试loss计算"""
    print("\n" + "=" * 70)
    print("4. Testing Loss Computation")
    print("=" * 70)
    
    # 创建临时模型
    num_users = len(two_tower_model.user_to_idx)
    device = two_tower_model.device
    model = TwoTowerModel(num_users).to(device)
    model.eval()
    
    # 构建mini batch
    all_users = list(two_tower_model.user_to_idx.keys())
    all_videos = list(two_tower_model.video_to_idx.keys())
    
    sample_users = np.random.choice(all_users, min(batch_size, len(all_users)), replace=False)
    sample_pos_videos = np.random.choice(all_videos, len(sample_users), replace=False)
    sample_neg_videos = np.random.choice(all_videos, len(sample_users), replace=False)
    
    # 获取索引
    user_idxs = torch.tensor([two_tower_model.user_to_idx[u] for u in sample_users], 
                             device=device, dtype=torch.long)
    
    # 获取正样本embeddings和metadata
    pos_text_embeds = []
    for vid in sample_pos_videos:
        if vid in two_tower_model.text_embeddings:
            pos_text_embeds.append(two_tower_model.text_embeddings[vid])
        else:
            pos_text_embeds.append(np.zeros(384))
    pos_text_embeds = torch.tensor(np.array(pos_text_embeds), device=device, dtype=torch.float32)
    pos_metadata = two_tower_model._get_video_metadata(sample_pos_videos)
    
    # 获取负样本embeddings和metadata
    neg_text_embeds = []
    for vid in sample_neg_videos:
        if vid in two_tower_model.text_embeddings:
            neg_text_embeds.append(two_tower_model.text_embeddings[vid])
        else:
            neg_text_embeds.append(np.zeros(384))
    neg_text_embeds = torch.tensor(np.array(neg_text_embeds), device=device, dtype=torch.float32)
    neg_metadata = two_tower_model._get_video_metadata(sample_neg_videos)
    
    user_features = two_tower_model._get_user_features(sample_users)
    
    # 计算scores
    with torch.no_grad():
        try:
            pos_scores, user_embeds, _ = model(
                user_idxs, pos_text_embeds, pos_metadata, user_features, device=str(device)
            )
            pos_scores = torch.diag(pos_scores)
            
            neg_scores, _, _ = model(
                user_idxs, neg_text_embeds, neg_metadata, user_features, device=str(device)
            )
            neg_scores = torch.diag(neg_scores)
            
            # 测试BPR loss计算
            score_diff = pos_scores - neg_scores
            
            print(f"\nScore statistics:")
            print(f"  Positive scores: mean={pos_scores.mean():.6f}, std={pos_scores.std():.6f}")
            print(f"  Negative scores: mean={neg_scores.mean():.6f}, std={neg_scores.std():.6f}")
            print(f"  Score diff: mean={score_diff.mean():.6f}, std={score_diff.std():.6f}")
            
            # 计算loss
            log_sigmoid_loss = -F.softplus(-score_diff)
            loss = -log_sigmoid_loss.mean()
            
            print(f"\nLoss computation:")
            print(f"  BPR loss: {loss.item():.6f}")
            print(f"  Loss is valid: {not (torch.isnan(loss) or torch.isinf(loss))}")
            
            print("\n✓ Loss computation test completed")
            return True
            
        except Exception as e:
            print(f"\n✗ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def mini_training_test(ratings_path: Path, videos_path: Optional[Path], 
                      epochs: int = 2, batch_size: int = 64):
    """进行mini训练测试"""
    print("\n" + "=" * 70)
    print("5. Running Mini Training Test")
    print("=" * 70)
    
    try:
        two_tower = TwoTowerRecall(ratings_path, videos_path, device="cpu")
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: 0.001")
        print(f"  Negative samples: 4")
        
        two_tower.train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            negative_samples=4,
            save_path=None
        )
        
        print("\n✓ Mini training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Mini training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("Two-Tower Model Training Debugger")
    print("=" * 70)
    
    data_dir = Path("data")
    
    # 选择数据集
    print("\nAvailable datasets:")
    print("  1. ratings.csv (MovieLens)")
    print("  2. ratings_kuairec.csv (KuaiRec)")
    
    choice = input("Select dataset (1 or 2, default=1): ").strip() or "1"
    
    if choice == "2":
        ratings_path = data_dir / "ratings_kuairec.csv"
        videos_path = data_dir / "videos_kuairec.csv"
    else:
        ratings_path = data_dir / "ratings.csv"
        videos_path = data_dir / "videos.csv"
    
    if not ratings_path.exists():
        print(f"Error: {ratings_path} not found!")
        return
    
    # 运行诊断
    check_training_data(ratings_path, videos_path if videos_path.exists() else None)
    
    # 初始化模型进行进一步检查
    print("\nInitializing Two-Tower model...")
    two_tower = TwoTowerRecall(
        ratings_path, 
        videos_path if videos_path.exists() else None,
        device="cpu"
    )
    
    check_embeddings(two_tower)
    test_forward_pass(two_tower, batch_size=32)
    test_loss_computation(two_tower, batch_size=32)
    
    # 运行mini训练测试
    run_training = input("\nRun mini training test? (y/n, default=n): ").strip().lower() == 'y'
    if run_training:
        mini_training_test(ratings_path, videos_path if videos_path.exists() else None, 
                          epochs=2, batch_size=64)
    
    print("\n" + "=" * 70)
    print("✓ Debugging completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
