"""
Evaluation Metrics for Recommendation Systems
评估指标：NDCG@K, Recall@K, Coverage, Diversity
"""
import numpy as np
from typing import List, Set, Dict, Tuple
import pandas as pd
from collections import Counter


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """
    计算NDCG@K
    
    Args:
        recommended: 推荐列表（按相关性排序）
        relevant: 相关物品集合（ground truth）
        k: 计算前k个
    
    Returns:
        NDCG@K score (0-1)
    """
    if k == 0:
        return 0.0
    
    # 只考虑前k个
    recommended = recommended[:k]
    
    if len(recommended) == 0:
        return 0.0
    
    # 计算DCG
    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            # 相关度分数为1（如果相关）
            relevance = 1.0
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts from 0
    
    # 计算IDCG（理想DCG）
    num_relevant = min(len(relevant), k)
    if num_relevant == 0:
        return 0.0
    
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
    
    # NDCG = DCG / IDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def recall_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """
    计算Recall@K
    
    Args:
        recommended: 推荐列表
        relevant: 相关物品集合（ground truth）
        k: 计算前k个
    
    Returns:
        Recall@K score (0-1)
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended = recommended[:k]
    hits = len(set(recommended) & relevant)
    
    return hits / len(relevant)


def precision_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """
    计算Precision@K
    
    Args:
        recommended: 推荐列表
        relevant: 相关物品集合
        k: 计算前k个
    
    Returns:
        Precision@K score (0-1)
    """
    if k == 0:
        return 0.0
    
    recommended = recommended[:k]
    if len(recommended) == 0:
        return 0.0
    
    hits = len(set(recommended) & relevant)
    return hits / len(recommended)


def coverage(recommendations: Dict[int, List[int]], all_items: Set[int]) -> float:
    """
    计算覆盖率（Coverage）
    推荐系统能够覆盖的物品比例
    
    Args:
        recommendations: 每个用户的推荐列表 {user_id: [item_ids]}
        all_items: 所有物品集合
    
    Returns:
        Coverage score (0-1)
    """
    if len(all_items) == 0:
        return 0.0
    
    recommended_items = set()
    for user_recs in recommendations.values():
        recommended_items.update(user_recs)
    
    return len(recommended_items) / len(all_items)


def diversity(recommendations: Dict[int, List[int]], item_features: Dict[int, List[str]] = None) -> float:
    """
    计算多样性（Diversity）
    推荐列表的平均物品间差异度
    
    Args:
        recommendations: 每个用户的推荐列表
        item_features: 物品特征字典（可选，用于计算相似度）
    
    Returns:
        Diversity score (0-1, 越高越多样)
    """
    if len(recommendations) == 0:
        return 0.0
    
    total_diversity = 0.0
    count = 0
    
    for user_id, recs in recommendations.items():
        if len(recs) < 2:
            continue
        
        # 如果没有特征，使用简单的集合差异度
        if item_features is None:
            # 简单的多样性：不同物品的比例
            diversity_score = len(set(recs)) / len(recs) if len(recs) > 0 else 0.0
        else:
            # 基于特征的多样性
            rec_features = [item_features.get(item, []) for item in recs]
            # 计算平均Jaccard距离
            similarities = []
            for i in range(len(rec_features)):
                for j in range(i + 1, len(rec_features)):
                    set_i = set(rec_features[i])
                    set_j = set(rec_features[j])
                    if len(set_i | set_j) == 0:
                        similarity = 0.0
                    else:
                        similarity = len(set_i & set_j) / len(set_i | set_j)
                    similarities.append(1.0 - similarity)  # 距离 = 1 - 相似度
            
            diversity_score = np.mean(similarities) if similarities else 0.0
        
        total_diversity += diversity_score
        count += 1
    
    return total_diversity / count if count > 0 else 0.0


def intra_list_diversity(recommended: List[int], item_features: Dict[int, List[str]] = None) -> float:
    """
    计算单个推荐列表的多样性
    
    Args:
        recommended: 推荐列表
        item_features: 物品特征字典
    
    Returns:
        Diversity score (0-1)
    """
    if len(recommended) < 2:
        return 0.0
    
    if item_features is None:
        return len(set(recommended)) / len(recommended)
    
    rec_features = [item_features.get(item, []) for item in recommended]
    similarities = []
    for i in range(len(rec_features)):
        for j in range(i + 1, len(rec_features)):
            set_i = set(rec_features[i])
            set_j = set(rec_features[j])
            if len(set_i | set_j) == 0:
                similarity = 0.0
            else:
                similarity = len(set_i & set_j) / len(set_i | set_j)
            similarities.append(1.0 - similarity)
    
    return np.mean(similarities) if similarities else 0.0


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    test_data: pd.DataFrame,
    k: int = 10,
    item_features: Dict[int, List[str]] = None,
    all_items: Set[int] = None
) -> Dict[str, float]:
    """
    综合评估推荐系统
    
    Args:
        recommendations: 每个用户的推荐列表 {user_id: [item_ids]}
        test_data: 测试集DataFrame，包含user_id, video_id, rating列
        k: 评估的top-k
        item_features: 物品特征字典（用于多样性计算）
        all_items: 所有物品集合（用于覆盖率计算）
    
    Returns:
        评估指标字典
    """
    metrics = {
        'ndcg@k': 0.0,
        'recall@k': 0.0,
        'precision@k': 0.0,
        'coverage': 0.0,
        'diversity': 0.0
    }
    
    # 构建ground truth
    test_dict = {}
    for _, row in test_data.iterrows():
        user_id = row['user_id']
        video_id = row['video_id']
        rating = row.get('rating', 1)
        
        # 只考虑高评分（>=4）作为相关物品
        if rating >= 4:
            if user_id not in test_dict:
                test_dict[user_id] = set()
            test_dict[user_id].add(video_id)
    
    # 计算每个用户的指标
    ndcg_scores = []
    recall_scores = []
    precision_scores = []
    
    for user_id, recs in recommendations.items():
        if user_id not in test_dict:
            continue
        
        relevant = test_dict[user_id]
        if len(relevant) == 0:
            continue
        
        # NDCG@K
        ndcg = ndcg_at_k(recs, relevant, k)
        ndcg_scores.append(ndcg)
        
        # Recall@K
        recall = recall_at_k(recs, relevant, k)
        recall_scores.append(recall)
        
        # Precision@K
        precision = precision_at_k(recs, relevant, k)
        precision_scores.append(precision)
    
    # 平均指标
    metrics['ndcg@k'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
    metrics['recall@k'] = np.mean(recall_scores) if recall_scores else 0.0
    metrics['precision@k'] = np.mean(precision_scores) if precision_scores else 0.0
    
    # Coverage
    if all_items is not None:
        metrics['coverage'] = coverage(recommendations, all_items)
    
    # Diversity
    metrics['diversity'] = diversity(recommendations, item_features)
    
    return metrics


def compare_systems(
    system1_name: str,
    system1_recommendations: Dict[int, List[int]],
    system2_name: str,
    system2_recommendations: Dict[int, List[int]],
    test_data: pd.DataFrame,
    k: int = 10,
    item_features: Dict[int, List[str]] = None,
    all_items: Set[int] = None
) -> pd.DataFrame:
    """
    对比两个推荐系统的性能
    
    Args:
        system1_name: 系统1名称
        system1_recommendations: 系统1的推荐结果
        system2_name: 系统2名称
        system2_recommendations: 系统2的推荐结果
        test_data: 测试数据
        k: 评估的top-k
        item_features: 物品特征
        all_items: 所有物品集合
    
    Returns:
        对比结果DataFrame
    """
    # 评估系统1
    metrics1 = evaluate_recommendations(
        system1_recommendations,
        test_data,
        k=k,
        item_features=item_features,
        all_items=all_items
    )
    
    # 评估系统2
    metrics2 = evaluate_recommendations(
        system2_recommendations,
        test_data,
        k=k,
        item_features=item_features,
        all_items=all_items
    )
    
    # 构建对比表格
    comparison = pd.DataFrame({
        'Metric': ['NDCG@' + str(k), 'Recall@' + str(k), 'Precision@' + str(k), 'Coverage', 'Diversity'],
        system1_name: [
            metrics1['ndcg@k'],
            metrics1['recall@k'],
            metrics1['precision@k'],
            metrics1['coverage'],
            metrics1['diversity']
        ],
        system2_name: [
            metrics2['ndcg@k'],
            metrics2['recall@k'],
            metrics2['precision@k'],
            metrics2['coverage'],
            metrics2['diversity']
        ]
    })
    
    # 计算差异
    comparison['Difference'] = comparison[system2_name] - comparison[system1_name]
    comparison['Improvement %'] = (
        (comparison[system2_name] - comparison[system1_name]) / 
        (comparison[system1_name] + 1e-8) * 100
    )
    
    return comparison


def main():
    """测试评估指标"""
    # 创建示例数据
    recommendations = {
        1: [1, 2, 3, 4, 5],
        2: [2, 3, 4, 5, 6],
        3: [1, 3, 5, 7, 9]
    }
    
    test_data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3],
        'video_id': [1, 2, 2, 3, 1, 3],
        'rating': [5, 4, 5, 4, 5, 4]
    })
    
    all_items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    
    # 评估
    metrics = evaluate_recommendations(recommendations, test_data, k=5, all_items=all_items)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()




