"""
LightGBM 排序模型：对召回结果进行精排
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from surprise import SVD

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入LightGBM，如果失败则设置为None
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    lgb = None
    print(f"警告: LightGBM不可用 ({e})，排序功能将不可用")

from src.features.feature_engineering import FeatureEngineering
from src.models.recall_system import MultiRecallSystem


class RankingModel:
    """LightGBM排序模型"""
    
    def __init__(
        self,
        ratings_path: Path = Path("data/ratings.csv"),
        model_path: Path = Path("models/svd_model.pkl"),
        dataset_dir: Path = Path("data/ml-100k"),
        ranking_model_path: Path = Path("models/lgbm_ranking_model.pkl")
    ):
        """
        初始化排序模型
        
        Args:
            ratings_path: 评分数据路径
            model_path: SVD模型路径
            dataset_dir: 数据集目录
            ranking_model_path: 排序模型保存路径
        """
        self.ratings_path = ratings_path
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.ranking_model_path = ranking_model_path
        
        # 初始化组件
        self.feature_engineer = FeatureEngineering(ratings_path, dataset_dir)
        self.recall_system = MultiRecallSystem(ratings_path, model_path, dataset_dir)
        self.svd_model = self._load_svd_model()
        self.ranking_model = None
    
    def _load_svd_model(self) -> SVD:
        """加载SVD模型"""
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)
    
    def _add_svd_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加SVD预测分数作为特征"""
        predictions = []
        for _, row in df.iterrows():
            try:
                pred = self.svd_model.predict(row['user_id'], row['video_id'])
                predictions.append(pred.est)
            except:
                predictions.append(3.0)  # 默认值
        
        df['svd_predicted_rating'] = predictions
        return df
    
    def prepare_training_data(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            n_samples: 采样数量
            
        Returns:
            (特征DataFrame, 标签Series)
        """
        print("准备训练数据...")
        
        # 从已有评分中采样正样本
        ratings = pd.read_csv(self.ratings_path)
        positive_samples = ratings.sample(
            min(n_samples, len(ratings)),
            random_state=42
        ).copy()
        positive_samples['label'] = 1
        
        # 生成负样本
        negative_samples = []
        users = ratings['user_id'].unique()[:50]  # 限制用户数量
        user_rated = ratings.groupby('user_id')['video_id'].apply(set).to_dict()
        all_videos = set(ratings['video_id'].unique())
        
        for user_id in users:
            rated = user_rated.get(user_id, set())
            candidates = list(all_videos - rated)
            
            if len(candidates) > 0:
                n_negative = min(20, len(candidates))
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
        features_list = []
        for _, row in all_samples.iterrows():
            feats = self.feature_engineer.create_features(
                row['user_id'],
                [row['video_id']]
            )
            features_list.append(feats.iloc[0])
        
        features_df = pd.DataFrame(features_list)
        
        # 添加SVD预测特征
        features_df = self._add_svd_predictions(features_df)
        
        # 选择特征列（排除非数值列和ID列）
        feature_cols = [
            col for col in features_df.columns
            if col not in ['user_id', 'video_id', 'label', 
                          'user_activity_level', 'video_popularity', 'video_quality']
            and features_df[col].dtype in [np.int64, np.float64]
        ]
        
        X = features_df[feature_cols].fillna(0)
        y = all_samples['label']
        
        return X, y
    
    def train(
        self,
        n_samples: int = 5000,
        params: Dict = None
    ):
        """
        训练LightGBM模型
        
        Args:
            n_samples: 训练样本数量
            params: LightGBM参数
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM不可用，无法训练排序模型。请安装: brew install libomp")
        
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'random_state': 42
            }
        
        print("准备训练数据...")
        X, y = self.prepare_training_data(n_samples)
        
        print(f"训练数据形状: {X.shape}")
        print(f"特征列: {list(X.columns)}")
        
        # 划分训练集和验证集
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 创建LightGBM数据集
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM不可用")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        print("开始训练LightGBM模型...")
        self.ranking_model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'eval'],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
        )
        
        # 保存模型
        self.ranking_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ranking_model_path, 'wb') as f:
            pickle.dump(self.ranking_model, f)
        
        print(f"模型已保存到 {self.ranking_model_path}")
        
        # 特征重要性
        if hasattr(self.ranking_model, 'feature_importance'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.ranking_model.feature_importance()
            }).sort_values('importance', ascending=False)
        
        print("\nTop 10 重要特征:")
        print(feature_importance.head(10))
    
    def load_model(self):
        """加载已训练的模型"""
        if not self.ranking_model_path.exists():
            raise FileNotFoundError(
                f"排序模型不存在，请先运行 train() 方法"
            )
        
        with open(self.ranking_model_path, 'rb') as f:
            self.ranking_model = pickle.load(f)
    
    def predict(
        self,
        user_id: int,
        video_ids: List[int],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        对候选视频进行排序
        
        Args:
            user_id: 用户ID
            video_ids: 候选视频ID列表
            top_k: 返回Top K结果
            
        Returns:
            [(video_id, score), ...] 按分数排序
        """
        if self.ranking_model is None:
            self.load_model()
        
        # 创建特征
        features = self.feature_engineer.create_features(user_id, video_ids)
        
        # 添加SVD预测特征
        features = self._add_svd_predictions(features)
        
        # 选择特征列
        feature_cols = [
            col for col in features.columns
            if col not in ['user_id', 'video_id',
                          'user_activity_level', 'video_popularity', 'video_quality']
            and features[col].dtype in [np.int64, np.float64]
        ]
        
        X = features[feature_cols].fillna(0)
        
        # 预测
        scores = self.ranking_model.predict(X)
        
        # 组合结果
        results = list(zip(video_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        recall_nums: Dict[str, int] = None
    ) -> List[Tuple[int, float]]:
        """
        完整的推荐流程：召回 + 排序
        
        Args:
            user_id: 用户ID
            top_k: 最终推荐数量
            recall_nums: 召回数量配置
            
        Returns:
            [(video_id, score), ...] 排序后的推荐结果
        """
        # 1. 多路召回
        candidates = self.recall_system.multi_recall(user_id, recall_nums)
        
        if len(candidates) == 0:
            return []
        
        # 2. 排序
        ranked_results = self.predict(user_id, candidates, top_k)
        
        return ranked_results


def main():
    """测试排序模型"""
    print("初始化排序模型...")
    ranking_model = RankingModel()
    
    # 训练模型
    print("\n开始训练排序模型...")
    ranking_model.train(n_samples=3000)
    
    # 测试推荐
    user_id = 1
    print(f"\n为用户 {user_id} 生成推荐...")
    recommendations = ranking_model.recommend(user_id, top_k=10)
    
    print(f"\nTop 10 推荐结果:")
    for i, (video_id, score) in enumerate(recommendations, 1):
        print(f"{i}. 视频 {video_id}: 分数 {score:.4f}")
    
    print("\n排序模型测试完成!")


if __name__ == "__main__":
    main()

