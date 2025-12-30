"""
特征工程：为排序模型准备特征
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


class FeatureEngineering:
    """特征工程类"""
    
    def __init__(
        self,
        ratings_path: Path = Path("data/ratings.csv"),
        dataset_dir: Path = Path("data/ml-100k")
    ):
        """
        初始化特征工程
        
        Args:
            ratings_path: 评分数据路径
            dataset_dir: 数据集目录
        """
        self.ratings_path = ratings_path
        self.dataset_dir = dataset_dir
        
        # 加载数据
        self.ratings = pd.read_csv(ratings_path)
        
        # 预计算特征
        self.user_features = self._compute_user_features()
        self.video_features = self._compute_video_features()
        self.user_video_interaction = self._compute_user_video_interaction()
    
    def _compute_user_features(self) -> pd.DataFrame:
        """计算用户特征"""
        user_features = self.ratings.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max'],
            'video_id': 'nunique'
        }).reset_index()
        
        user_features.columns = [
            'user_id',
            'user_avg_rating',
            'user_rating_std',
            'user_rating_count',
            'user_min_rating',
            'user_max_rating',
            'user_unique_videos'
        ]
        
        # 填充缺失值
        user_features['user_rating_std'] = user_features['user_rating_std'].fillna(0)
        
        # 用户活跃度（评分数量）
        user_features['user_activity_level'] = pd.cut(
            user_features['user_rating_count'],
            bins=[0, 20, 50, 100, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return user_features
    
    def _compute_video_features(self) -> pd.DataFrame:
        """计算视频特征"""
        video_features = self.ratings.groupby('video_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max'],
            'user_id': 'nunique'
        }).reset_index()
        
        video_features.columns = [
            'video_id',
            'video_avg_rating',
            'video_rating_std',
            'video_rating_count',
            'video_min_rating',
            'video_max_rating',
            'video_unique_users'
        ]
        
        # 填充缺失值
        video_features['video_rating_std'] = video_features['video_rating_std'].fillna(0)
        
        # 视频流行度
        video_features['video_popularity'] = pd.cut(
            video_features['video_rating_count'],
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # 视频质量（基于平均评分）
        video_features['video_quality'] = pd.cut(
            video_features['video_avg_rating'],
            bins=[0, 2, 3, 4, 5],
            labels=['poor', 'fair', 'good', 'excellent']
        )
        
        return video_features
    
    def _compute_user_video_interaction(self) -> pd.DataFrame:
        """计算用户-视频交互特征"""
        # 用户对视频的平均评分（如果用户已评分过该视频）
        user_video_ratings = self.ratings.groupby(['user_id', 'video_id']).agg({
            'rating': 'mean'
        }).reset_index()
        user_video_ratings.columns = ['user_id', 'video_id', 'user_video_rating']
        
        return user_video_ratings
    
    def create_features(
        self,
        user_id: int,
        video_ids: List[int]
    ) -> pd.DataFrame:
        """
        为用户-视频对创建特征
        
        Args:
            user_id: 用户ID
            video_ids: 视频ID列表
            
        Returns:
            特征DataFrame
        """
        # 创建用户-视频对
        pairs = pd.DataFrame({
            'user_id': [user_id] * len(video_ids),
            'video_id': video_ids
        })
        
        # 合并用户特征
        pairs = pairs.merge(
            self.user_features,
            on='user_id',
            how='left'
        )
        
        # 合并视频特征
        pairs = pairs.merge(
            self.video_features,
            on='video_id',
            how='left'
        )
        
        # 合并交互特征
        pairs = pairs.merge(
            self.user_video_interaction,
            on=['user_id', 'video_id'],
            how='left'
        )
        
        # 添加交叉特征
        pairs = self._add_cross_features(pairs)
        
        # 添加SVD预测特征
        pairs = self._add_svd_features(pairs)
        
        return pairs
    
    def _add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交叉特征"""
        # 用户平均评分 vs 视频平均评分
        df['rating_diff'] = df['user_avg_rating'] - df['video_avg_rating']
        df['rating_ratio'] = df['user_avg_rating'] / (df['video_avg_rating'] + 1e-6)
        
        # 用户活跃度 vs 视频流行度
        df['activity_popularity_match'] = (
            df['user_rating_count'] * df['video_rating_count']
        ) / (df['user_rating_count'] + df['video_rating_count'] + 1e-6)
        
        # 用户评分标准差 vs 视频评分标准差
        df['std_diff'] = df['user_rating_std'] - df['video_rating_std']
        
        return df
    
    def _add_svd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加SVD预测特征（需要模型）"""
        # 这个特征会在排序模型中添加
        # 这里先返回原DataFrame
        return df
    
    def create_training_features(
        self,
        sample_size: int = 10000
    ) -> pd.DataFrame:
        """
        创建训练样本的特征
        
        Args:
            sample_size: 采样数量
            
        Returns:
            特征DataFrame（包含label）
        """
        # 从已有评分中采样正样本
        positive_samples = self.ratings.sample(
            min(sample_size, len(self.ratings)),
            random_state=42
        ).copy()
        positive_samples['label'] = 1
        
        # 生成负样本（用户未评分的视频）
        negative_samples = []
        users = self.ratings['user_id'].unique()
        user_rated = self.ratings.groupby('user_id')['video_id'].apply(set).to_dict()
        all_videos = set(self.ratings['video_id'].unique())
        
        for user_id in users[:min(100, len(users))]:  # 限制用户数量以加快速度
            rated = user_rated.get(user_id, set())
            candidates = list(all_videos - rated)
            
            if len(candidates) > 0:
                # 为每个用户采样一些负样本
                n_negative = min(10, len(candidates))
                sampled = np.random.choice(candidates, n_negative, replace=False)
                for video_id in sampled:
                    negative_samples.append({
                        'user_id': user_id,
                        'video_id': video_id,
                        'label': 0
                    })
        
        negative_df = pd.DataFrame(negative_samples)
        
        # 合并正负样本
        all_samples = pd.concat([positive_samples, negative_df], ignore_index=True)
        
        # 创建特征
        features = self.create_features(
            all_samples['user_id'].values[0],
            all_samples['video_id'].values.tolist()
        )
        
        # 简化：直接使用已有的特征计算
        # 实际应该为每个样本单独计算
        result = all_samples.merge(
            self.user_features,
            on='user_id',
            how='left'
        ).merge(
            self.video_features,
            on='video_id',
            how='left'
        )
        
        # 添加交叉特征
        result['rating_diff'] = result['user_avg_rating'] - result['video_avg_rating']
        result['activity_popularity_match'] = (
            result['user_rating_count'] * result['video_rating_count']
        ) / (result['user_rating_count'] + result['video_rating_count'] + 1e-6)
        
        return result


def main():
    """测试特征工程"""
    print("初始化特征工程...")
    fe = FeatureEngineering()
    
    print(f"\n用户特征统计:")
    print(f"  用户数: {len(fe.user_features)}")
    print(f"  特征列: {list(fe.user_features.columns)}")
    
    print(f"\n视频特征统计:")
    print(f"  视频数: {len(fe.video_features)}")
    print(f"  特征列: {list(fe.video_features.columns)}")
    
    # 测试为特定用户-视频对创建特征
    user_id = 1
    video_ids = [242, 302, 377, 51, 346]
    
    print(f"\n为用户 {user_id} 和视频 {video_ids} 创建特征...")
    features = fe.create_features(user_id, video_ids)
    print(f"\n生成的特征形状: {features.shape}")
    print(f"特征列: {list(features.columns)}")
    print(f"\n前5行特征:")
    print(features.head())
    
    print("\n特征工程测试完成!")


if __name__ == "__main__":
    main()

