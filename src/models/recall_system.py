"""
多路召回系统：结合多种召回策略生成候选集
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from surprise import SVD
import pickle


class MultiRecallSystem:
    """多路召回系统"""
    
    def __init__(
        self,
        ratings_path: Path = Path("data/ratings.csv"),
        model_path: Path = Path("models/svd_model.pkl"),
        dataset_dir: Path = Path("data/ml-100k")
    ):
        """
        初始化多路召回系统
        
        Args:
            ratings_path: 评分数据路径
            model_path: SVD模型路径
            dataset_dir: 数据集目录
        """
        self.ratings_path = ratings_path
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        
        # 加载数据
        self.ratings = pd.read_csv(ratings_path)
        self.model = self._load_model()
        
        # 预计算热门视频和用户历史
        self.popular_videos = self._get_popular_videos()
        self.user_history = self._build_user_history()
        self.video_stats = self._compute_video_stats()
    
    def _load_model(self) -> SVD:
        """加载SVD模型"""
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)
    
    def _get_popular_videos(self, top_n: int = 1000) -> List[int]:
        """获取热门视频（按评分数量）"""
        video_counts = self.ratings['video_id'].value_counts()
        return video_counts.head(top_n).index.tolist()
    
    def _build_user_history(self) -> Dict[int, set]:
        """构建用户历史评分字典"""
        user_history = {}
        for user_id, group in self.ratings.groupby('user_id'):
            user_history[user_id] = set(group['video_id'].values)
        return user_history
    
    def _compute_video_stats(self) -> pd.DataFrame:
        """计算视频统计信息"""
        video_stats = self.ratings.groupby('video_id').agg({
            'rating': ['mean', 'count', 'std'],
            'user_id': 'nunique'
        }).reset_index()
        video_stats.columns = ['video_id', 'avg_rating', 'rating_count', 'rating_std', 'user_count']
        video_stats['rating_std'] = video_stats['rating_std'].fillna(0)
        return video_stats
    
    def recall_by_collaborative_filtering(
        self,
        user_id: int,
        top_k: int = 200
    ) -> List[Tuple[int, float]]:
        """
        召回方式1：协同过滤召回（基于SVD模型）
        
        Args:
            user_id: 用户ID
            top_k: 召回数量
            
        Returns:
            [(video_id, predicted_rating), ...]
        """
        if user_id not in self.user_history:
            return []
        
        user_rated = self.user_history[user_id]
        all_videos = self.ratings['video_id'].unique()
        
        predictions = []
        for video_id in all_videos:
            if video_id not in user_rated:
                try:
                    pred = self.model.predict(user_id, video_id)
                    predictions.append((video_id, pred.est))
                except:
                    continue
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def recall_by_popularity(
        self,
        user_id: int,
        top_k: int = 100
    ) -> List[Tuple[int, float]]:
        """
        召回方式2：热门召回
        
        Args:
            user_id: 用户ID
            top_k: 召回数量
            
        Returns:
            [(video_id, popularity_score), ...]
        """
        user_rated = self.user_history.get(user_id, set())
        
        # 排除用户已看过的视频
        candidates = [vid for vid in self.popular_videos if vid not in user_rated]
        
        # 使用评分数量作为热门度分数
        popularity_scores = []
        for video_id in candidates[:top_k]:
            count = self.video_stats[
                self.video_stats['video_id'] == video_id
            ]['rating_count'].values
            if len(count) > 0:
                popularity_scores.append((video_id, float(count[0])))
        
        return popularity_scores[:top_k]
    
    def recall_by_high_rating(
        self,
        user_id: int,
        top_k: int = 100,
        min_ratings: int = 10
    ) -> List[Tuple[int, float]]:
        """
        召回方式3：高评分召回（平均评分高且评分数量足够）
        
        Args:
            user_id: 用户ID
            top_k: 召回数量
            min_ratings: 最小评分数量阈值
            
        Returns:
            [(video_id, avg_rating), ...]
        """
        user_rated = self.user_history.get(user_id, set())
        
        # 筛选：评分数量 >= min_ratings
        high_quality = self.video_stats[
            (self.video_stats['rating_count'] >= min_ratings) &
            (~self.video_stats['video_id'].isin(user_rated))
        ].copy()
        
        # 按平均评分排序
        high_quality = high_quality.sort_values('avg_rating', ascending=False)
        
        results = [
            (row['video_id'], row['avg_rating'])
            for _, row in high_quality.head(top_k).iterrows()
        ]
        
        return results
    
    def recall_by_user_similarity(
        self,
        user_id: int,
        top_k: int = 100
    ) -> List[Tuple[int, float]]:
        """
        召回方式4：基于用户相似度的召回
        找到相似用户喜欢的视频
        
        Args:
            user_id: 用户ID
            top_k: 召回数量
            
        Returns:
            [(video_id, similarity_score), ...]
        """
        if user_id not in self.user_history:
            return []
        
        user_rated = self.user_history[user_id]
        
        # 计算用户相似度（基于共同评分的视频）
        user_similarities = {}
        user_items = self.user_history[user_id]
        
        for other_user_id, other_items in self.user_history.items():
            if other_user_id == user_id:
                continue
            
            # Jaccard相似度
            intersection = len(user_items & other_items)
            union = len(user_items | other_items)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0:
                    user_similarities[other_user_id] = similarity
        
        # 找到最相似的K个用户
        similar_users = sorted(
            user_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]  # 取前50个相似用户
        
        # 收集相似用户喜欢的视频
        candidate_scores = {}
        for similar_user_id, similarity in similar_users:
            similar_user_items = self.user_history[similar_user_id]
            for video_id in similar_user_items - user_rated:
                if video_id not in candidate_scores:
                    candidate_scores[video_id] = 0
                candidate_scores[video_id] += similarity
        
        # 排序并返回
        results = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return results
    
    def multi_recall(
        self,
        user_id: int,
        recall_nums: Dict[str, int] = None
    ) -> List[int]:
        """
        多路召回：结合多种召回策略
        
        Args:
            user_id: 用户ID
            recall_nums: 各路召回数量配置
                {
                    'cf': 200,      # 协同过滤
                    'popular': 100, # 热门
                    'high_rating': 100,  # 高评分
                    'similarity': 100   # 用户相似度
                }
        
        Returns:
            去重后的候选视频ID列表
        """
        if recall_nums is None:
            recall_nums = {
                'cf': 200,
                'popular': 100,
                'high_rating': 100,
                'similarity': 100
            }
        
        all_candidates = set()
        
        # 1. 协同过滤召回
        cf_candidates = self.recall_by_collaborative_filtering(
            user_id, recall_nums.get('cf', 200)
        )
        all_candidates.update([vid for vid, _ in cf_candidates])
        
        # 2. 热门召回
        popular_candidates = self.recall_by_popularity(
            user_id, recall_nums.get('popular', 100)
        )
        all_candidates.update([vid for vid, _ in popular_candidates])
        
        # 3. 高评分召回
        high_rating_candidates = self.recall_by_high_rating(
            user_id, recall_nums.get('high_rating', 100)
        )
        all_candidates.update([vid for vid, _ in high_rating_candidates])
        
        # 4. 用户相似度召回
        similarity_candidates = self.recall_by_user_similarity(
            user_id, recall_nums.get('similarity', 100)
        )
        all_candidates.update([vid for vid, _ in similarity_candidates])
        
        # 排除用户已看过的视频
        user_rated = self.user_history.get(user_id, set())
        final_candidates = list(all_candidates - user_rated)
        
        return final_candidates
    
    def get_recall_details(
        self,
        user_id: int,
        recall_nums: Dict[str, int] = None
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        获取各路召回的详细信息（用于调试和分析）
        
        Returns:
            各路召回的候选列表
        """
        if recall_nums is None:
            recall_nums = {
                'cf': 200,
                'popular': 100,
                'high_rating': 100,
                'similarity': 100
            }
        
        return {
            'collaborative_filtering': self.recall_by_collaborative_filtering(
                user_id, recall_nums.get('cf', 200)
            ),
            'popularity': self.recall_by_popularity(
                user_id, recall_nums.get('popular', 100)
            ),
            'high_rating': self.recall_by_high_rating(
                user_id, recall_nums.get('high_rating', 100)
            ),
            'user_similarity': self.recall_by_user_similarity(
                user_id, recall_nums.get('similarity', 100)
            )
        }


def main():
    """测试多路召回系统"""
    print("初始化多路召回系统...")
    recall_system = MultiRecallSystem()
    
    user_id = 1
    print(f"\n为用户 {user_id} 进行多路召回...")
    
    # 获取各路召回详情
    recall_details = recall_system.get_recall_details(user_id)
    
    print("\n各路召回结果统计:")
    for recall_type, candidates in recall_details.items():
        print(f"  {recall_type}: {len(candidates)} 个候选")
    
    # 多路召回合并
    candidates = recall_system.multi_recall(user_id)
    print(f"\n合并后总候选数: {len(candidates)}")
    
    print("\n多路召回系统测试完成!")


if __name__ == "__main__":
    main()

