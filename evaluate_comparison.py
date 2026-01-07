"""
System Comparison Evaluation Script
对比评估传统系统和高级系统的性能
支持MovieLens和MicroLens数据集
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Set, Optional
from tqdm import tqdm

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.recall_system import MultiRecallSystem
from src.evaluation.metrics import (
    evaluate_recommendations,
    compare_systems,
    ndcg_at_k,
    recall_at_k
)

# 尝试导入 Cross-Encoder（可选）
try:
    from src.models.cross_encoder_ranking import CrossEncoderRanker
    CROSS_ENCODER_AVAILABLE = True
except (ImportError, OSError) as e:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoderRanker = None
    print(f"Warning: Cross-Encoder not available ({type(e).__name__})")


def split_train_test(ratings: pd.DataFrame, test_ratio: float = 0.2):
    """划分训练集和测试集"""
    # 按时间戳排序（如果有）或随机打乱
    if 'timestamp' in ratings.columns:
        ratings = ratings.sort_values('timestamp')
    else:
        ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 按用户划分，确保每个用户在测试集中都有数据
    test_data = []
    train_data = []
    
    for user_id, group in ratings.groupby('user_id'):
        n_test = max(1, int(len(group) * test_ratio))
        user_test = group.tail(n_test)
        user_train = group.head(len(group) - n_test)
        
        test_data.append(user_test)
        train_data.append(user_train)
    
    test_df = pd.concat(test_data, ignore_index=True)
    train_df = pd.concat(train_data, ignore_index=True)
    
    return train_df, test_df


def generate_recommendations(
    recall_system: MultiRecallSystem,
    cross_encoder: CrossEncoderRanker,
    user_ids: List[int],
    use_advanced: bool,
    top_k: int = 10,
    use_reranking: bool = False,
    videos_path: Optional[Path] = None
) -> Dict[int, List[int]]:
    """
    为多个用户生成推荐
    
    Args:
        recall_system: 召回系统
        cross_encoder: Cross-Encoder精排模型（可选）
        user_ids: 用户ID列表
        use_advanced: 是否使用高级模型
        top_k: 最终推荐数量
        use_reranking: 是否使用Cross-Encoder精排
    
    Returns:
        {user_id: [video_ids]}
    """
    recommendations = {}
    
    for user_id in tqdm(user_ids, desc="Generating recommendations"):
        try:
            # 根据是否使用高级模型设置召回数量
            if use_advanced:
                # 先进系统：使用多路召回（包括高级模型）
                models_dir = Path("models")
                videos_path_check = videos_path if videos_path else Path("data/videos.csv")
                
                # 基础召回：保持传统方法数量
                recall_nums = {
                    'cf': 200,
                    'popular': 100,
                    'high_rating': 100,
                    'similarity': 100
                }
                
                # 添加高级模型召回
                if (models_dir / "bm25_model.pkl").exists():
                    if videos_path_check.exists():
                        recall_nums['bm25'] = 200
                    else:
                        recall_nums['bm25'] = 50
                
                # 检查Two-Tower是否可用
                try:
                    import torch
                    if (models_dir / "two_tower_model.pth").exists():
                        if hasattr(recall_system, 'two_tower') and recall_system.two_tower is not None:
                            recall_nums['two_tower'] = 150
                except (ImportError, OSError):
                    pass
            else:
                recall_nums = {
                    'cf': 200,
                    'popular': 100,
                    'high_rating': 100,
                    'similarity': 100
                }
            
            # 获取召回候选和分数（用于Cross-Encoder加权融合）
            recall_details = recall_system.get_recall_details(user_id, recall_nums=recall_nums)
            
            # 合并召回候选（保留分数信息）
            candidate_scores = {}
            for method, results in recall_details.items():
                for vid, score in results:
                    # 对于每个候选，保留最高分数（来自最重要的召回方法）
                    if vid not in candidate_scores or score > candidate_scores[vid]:
                        candidate_scores[vid] = score
            
            # 排除用户已看过的视频
            user_rated = recall_system.user_history.get(user_id, set())
            candidates = [vid for vid in candidate_scores.keys() if vid not in user_rated]
            
            if len(candidates) == 0:
                recommendations[user_id] = []
                continue
            
            # 精排（如果启用）
            if use_reranking and cross_encoder is not None and hasattr(cross_encoder, 'model') and cross_encoder.model is not None:
                # 使用Cross-Encoder对候选进行精排（结合召回分数）
                try:
                    # 传递召回分数给Cross-Encoder进行加权融合
                    recall_scores_dict = {vid: candidate_scores[vid] for vid in candidates}
                    reranked = cross_encoder.rank(
                        user_id, 
                        candidates, 
                        top_k=top_k,
                        recall_scores=recall_scores_dict
                    )
                    final_recs = [vid for vid, _ in reranked]
                except Exception as e:
                    # 精排失败，使用原始候选（按召回分数排序）
                    candidates_sorted = sorted(candidates, key=lambda x: candidate_scores.get(x, 0), reverse=True)
                    final_recs = candidates_sorted[:top_k]
            else:
                # 简单截取top_k（按召回分数排序）
                candidates_sorted = sorted(candidates, key=lambda x: candidate_scores.get(x, 0), reverse=True)
                final_recs = candidates_sorted[:top_k]
            
            recommendations[user_id] = final_recs
            
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            recommendations[user_id] = []
    
    return recommendations


def evaluate_system(
    dataset_name: str,
    ratings_path: Path,
    videos_path: Optional[Path],
    use_advanced: bool,
    use_reranking: bool = False,
    test_ratio: float = 0.2,
    k: int = 10,
    num_test_users: int = 100
):
    """
    评估单个系统
    
    Args:
        dataset_name: 数据集名称（用于显示）
        ratings_path: 评分数据路径
        videos_path: 视频元数据路径
        use_advanced: 是否使用高级模型
        use_reranking: 是否使用Cross-Encoder精排
        test_ratio: 测试集比例
        k: 评估的top-k
        num_test_users: 测试用户数量
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {'Advanced' if use_advanced else 'Traditional'} System")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据
    print("Loading data...")
    ratings = pd.read_csv(ratings_path)
    
    # 划分训练测试集
    print("Splitting train/test data...")
    train_data, test_data = split_train_test(ratings, test_ratio)
    
    # 保存临时训练数据（用于训练召回系统）
    temp_train_path = Path("data/temp_train_ratings.csv")
    train_data.to_csv(temp_train_path, index=False)
    
    # 初始化召回系统
    print("Initializing recall system...")
    recall_system = MultiRecallSystem(
        ratings_path=temp_train_path,
        videos_path=videos_path,
        use_advanced_models=use_advanced
    )
    
    # 初始化Cross-Encoder（如果使用）
    cross_encoder = None
    if use_reranking and CROSS_ENCODER_AVAILABLE and CrossEncoderRanker is not None:
        print("Initializing Cross-Encoder...")
        try:
            cross_encoder = CrossEncoderRanker(temp_train_path, videos_path)
        except Exception as e:
            print(f"Warning: Failed to initialize Cross-Encoder: {e}")
            cross_encoder = None
    elif use_reranking:
        print("Warning: Cross-Encoder not available, skipping reranking")
    
    # 选择测试用户
    test_users = test_data['user_id'].unique()[:num_test_users]
    print(f"Evaluating on {len(test_users)} test users...")
    
    # 生成推荐
    recommendations = generate_recommendations(
        recall_system,
        cross_encoder,
        test_users.tolist(),
        use_advanced,
        top_k=k,
        use_reranking=use_reranking,
        videos_path=videos_path
    )
    
    # 评估
    all_items = set(ratings['video_id'].unique())
    
    metrics = evaluate_recommendations(
        recommendations,
        test_data,
        k=k,
        all_items=all_items
    )
    
    # 清理临时文件
    if temp_train_path.exists():
        temp_train_path.unlink()
    
    return metrics, recommendations


def main():
    """主函数：对比评估两个系统"""
    print("=" * 60)
    print("System Comparison Evaluation")
    print("=" * 60)
    
    data_dir = Path("data")
    models_dir = Path("models")
    
    # 检查模型是否存在
    two_tower_exists = (models_dir / "two_tower_model.pth").exists()
    bm25_exists = (models_dir / "bm25_model.pkl").exists()
    
    # 检查 PyTorch 是否可用（Two-Tower需要，BM25不需要）
    try:
        import torch
        torch_available = True
    except (ImportError, OSError):
        torch_available = False
    
    # 只要有BM25就可以使用高级系统（Two-Tower是可选的）
    if bm25_exists:
        use_advanced = True
        if two_tower_exists and torch_available:
            print("\n✓ Advanced models available (Two-Tower + BM25)")
            print("  Will evaluate both traditional and advanced systems")
        elif bm25_exists:
            print("\n✓ BM25 model available (Two-Tower not available)")
            print("  Will evaluate both systems (advanced system will use BM25 only)")
    else:
        use_advanced = False
        if not torch_available:
            print("\n⚠️  PyTorch not available and BM25 not found")
            print("   Advanced models will be disabled.")
            print("   See FIX_PYTORCH.md for installation instructions.")
        else:
            print("\n⚠️  Advanced models not found!")
            print("   Please run at least: python3 train_bm25_only.py")
        print("\nProceeding with traditional system only...")
    
    # 评估配置
    datasets = []
    
    # MovieLens数据集
    movielens_ratings = data_dir / "ratings.csv"
    movielens_videos = data_dir / "videos.csv"
    if movielens_ratings.exists():
        datasets.append(("MovieLens-100K", movielens_ratings, movielens_videos))
    
    # KuaiRec数据集（如果存在）
    kuairec_ratings = data_dir / "ratings_kuairec.csv"
    kuairec_videos = data_dir / "videos_kuairec.csv"
    if kuairec_ratings.exists():
        datasets.append(("KuaiRec", kuairec_ratings, kuairec_videos if kuairec_videos.exists() else None))
    
    # MicroLens数据集（如果存在）
    microlens_ratings = data_dir / "microlens-100k" / "interactions.csv"
    if microlens_ratings.exists():
        # 需要先预处理
        print("\nMicroLens dataset found. Please ensure it's preprocessed.")
        # datasets.append(("MicroLens-100K", microlens_ratings, None))
    
    if len(datasets) == 0:
        print("Error: No dataset found!")
        print("Please ensure ratings.csv exists in data/ directory.")
        return
    
    # 对每个数据集进行评估
    for dataset_name, ratings_path, videos_path in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")
        
        # 评估传统系统
        print("\n1. Evaluating Traditional System...")
        try:
            trad_metrics, trad_recs = evaluate_system(
                dataset_name,
                ratings_path,
                videos_path,
                use_advanced=False,
                use_reranking=False,
                k=10,
                num_test_users=50  # 减少用户数以加快评估
            )
            
            print("\nTraditional System Metrics:")
            for metric, value in trad_metrics.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Error evaluating traditional system: {e}")
            trad_metrics = None
            trad_recs = None
        
        # 评估高级系统
        if use_advanced:
            print("\n2. Evaluating Advanced System...")
            # 检查Cross-Encoder是否可用
            use_reranking = CROSS_ENCODER_AVAILABLE and CrossEncoderRanker is not None
            if use_reranking:
                print("  ✓ Cross-Encoder available, will use fine ranking")
            else:
                print("  ⚠️  Cross-Encoder not available, will skip fine ranking")
            
            try:
                adv_metrics, adv_recs = evaluate_system(
                    dataset_name,
                    ratings_path,
                    videos_path,
                    use_advanced=True,
                    use_reranking=use_reranking,  # 启用Cross-Encoder精排（如果可用）
                    k=10,
                    num_test_users=50
                )
                
                print("\nAdvanced System Metrics:")
                for metric, value in adv_metrics.items():
                    print(f"  {metric}: {value:.4f}")
            except Exception as e:
                print(f"Error evaluating advanced system: {e}")
                import traceback
                traceback.print_exc()
                adv_metrics = None
                adv_recs = None
            
            # 对比结果
            if trad_metrics and adv_metrics:
                print("\n" + "="*60)
                print("Comparison Results")
                print("="*60)
                
                comparison = pd.DataFrame({
                    'Metric': ['NDCG@10', 'Recall@10', 'Precision@10', 'Coverage', 'Diversity'],
                    'Traditional': [
                        trad_metrics['ndcg@k'],
                        trad_metrics['recall@k'],
                        trad_metrics['precision@k'],
                        trad_metrics['coverage'],
                        trad_metrics['diversity']
                    ],
                    'Advanced': [
                        adv_metrics['ndcg@k'],
                        adv_metrics['recall@k'],
                        adv_metrics['precision@k'],
                        adv_metrics['coverage'],
                        adv_metrics['diversity']
                    ]
                })
                
                comparison['Improvement'] = comparison['Advanced'] - comparison['Traditional']
                comparison['Improvement %'] = (
                    (comparison['Advanced'] - comparison['Traditional']) /
                    (comparison['Traditional'] + 1e-8) * 100
                )
                
                print(comparison.to_string(index=False))
                
                # 保存结果
                output_path = data_dir / f"comparison_results_{dataset_name.replace('-', '_')}.csv"
                comparison.to_csv(output_path, index=False)
                print(f"\nResults saved to: {output_path}")
        else:
            # 只评估了传统系统，显示结果摘要
            print("\n" + "="*60)
            print("Evaluation Summary (Traditional System Only)")
            print("="*60)
            print("\nTraditional System Metrics:")
            for metric, value in trad_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print("\n" + "="*60)
            print("Note: Advanced system evaluation skipped")
            print("To evaluate advanced system:")
            print("  1. Fix PyTorch architecture: ./fix_pytorch.sh")
            print("  2. Train advanced models: python3 train_advanced.py")
            print("  3. Run evaluation again: python3 evaluate_comparison.py")
            print("="*60)
            
            # 保存传统系统结果
            if trad_metrics:
                output_path = data_dir / f"traditional_results_{dataset_name.replace('-', '_')}.csv"
                trad_df = pd.DataFrame([trad_metrics])
                trad_df.to_csv(output_path, index=False)
                print(f"\nTraditional system results saved to: {output_path}")


if __name__ == "__main__":
    main()

