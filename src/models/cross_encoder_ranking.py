"""
Cross-Encoder for Fine Ranking
使用预训练的Cross-Encoder进行精排
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pickle

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except (ImportError, OSError, PermissionError) as e:
    CROSS_ENCODER_AVAILABLE = False
    print(f"Warning: sentence-transformers not available. Cross-Encoder ranking will be disabled. ({type(e).__name__})")


class CrossEncoderRanker:
    """Cross-Encoder精排系统"""
    def __init__(
        self,
        ratings_path: Path,
        videos_path: Optional[Path] = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        self.ratings = pd.read_csv(ratings_path)
        self.device = device
        
        # 加载视频元数据
        if videos_path and videos_path.exists():
            self.videos = pd.read_csv(videos_path)
        else:
            self.videos = None
            print("Warning: videos.csv not found. Using minimal video metadata.")
        
        # 初始化Cross-Encoder模型
        if CROSS_ENCODER_AVAILABLE:
            print(f"Loading Cross-Encoder model: {model_name}")
            self.model = CrossEncoder(model_name, device=device)
            print("Cross-Encoder model loaded successfully")
        else:
            self.model = None
            print("Error: Cross-Encoder not available")
    
    def _get_video_text(self, video_id: int) -> str:
        """获取视频文本（标题+描述）"""
        if self.videos is not None:
            video_row = self.videos[self.videos['video_id'] == video_id]
            if not video_row.empty:
                title = str(video_row.iloc[0].get('title', ''))
                desc = str(video_row.iloc[0].get('description', ''))
                text = f"{title} {desc}".strip()
                return text if text else f"Video {video_id}"
        return f"Video {video_id}"
    
    def _get_user_query(self, user_id: int) -> str:
        """根据用户历史构建查询文本（改进版）"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        if len(user_ratings) == 0:
            return "user preferences"
        
        # 策略1：优先使用高评分视频（rating >= 4）
        high_rated = user_ratings[user_ratings['rating'] >= 4].sort_values('rating', ascending=False)
        
        # 策略2：如果没有高评分，使用最近观看的视频
        if len(high_rated) == 0:
            high_rated = user_ratings.sort_values('rating', ascending=False).head(10)
        
        # 构建查询：提取关键词和主题
        query_parts = []
        seen_texts = set()  # 避免重复
        
        # 使用前5个最高评分的视频（避免查询过长）
        for _, row in high_rated.head(5).iterrows():
            text = self._get_video_text(row['video_id'])
            # 只保留标题部分（通常更简洁、更有代表性）
            if self.videos is not None:
                video_row = self.videos[self.videos['video_id'] == row['video_id']]
                if not video_row.empty:
                    title = str(video_row.iloc[0].get('title', ''))
                    if title and title not in seen_texts:
                        query_parts.append(title)
                        seen_texts.add(title)
        
        # 如果还是没有，使用完整文本
        if len(query_parts) == 0:
            for _, row in high_rated.head(3).iterrows():
                text = self._get_video_text(row['video_id'])
                if text and text not in seen_texts:
                    query_parts.append(text)
                    seen_texts.add(text)
        
        # 组合查询，限制长度（避免过长导致模型性能下降）
        query = " ".join(query_parts)
        if len(query) > 500:  # 限制查询长度
            query = query[:500]
        
        return query if query else "user preferences"
    
    def rank(
        self,
        user_id: int,
        candidate_video_ids: List[int],
        top_k: Optional[int] = None,
        recall_scores: Optional[Dict[int, float]] = None
    ) -> List[Tuple[int, float]]:
        """
        使用Cross-Encoder对候选视频进行精排（改进版）
        
        Args:
            user_id: 用户ID
            candidate_video_ids: 候选视频ID列表（来自粗排）
            top_k: 返回top_k个结果，如果为None则返回全部
            recall_scores: 召回阶段的分数（可选，用于加权融合）
        
        Returns:
            List of (video_id, score) tuples, 按分数降序排列
        """
        if self.model is None:
            print("Error: Cross-Encoder model not available. Cannot perform ranking.")
            return []
        
        if len(candidate_video_ids) == 0:
            return []
        
        # 构建用户查询
        user_query = self._get_user_query(user_id)
        
        # 如果候选太多，先进行粗筛（只对前200个进行精排，提高效率）
        max_candidates = 200
        if len(candidate_video_ids) > max_candidates:
            # 如果有召回分数，使用召回分数排序
            if recall_scores:
                scored_candidates = [(vid, recall_scores.get(vid, 0.0)) for vid in candidate_video_ids]
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                candidate_video_ids = [vid for vid, _ in scored_candidates[:max_candidates]]
            else:
                # 否则随机选择（但保持顺序）
                candidate_video_ids = candidate_video_ids[:max_candidates]
        
        # 构建查询-视频对
        pairs = []
        for video_id in candidate_video_ids:
            video_text = self._get_video_text(video_id)
            pairs.append([user_query, video_text])
        
        # 批量计算分数（避免内存溢出）
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            try:
                scores = self.model.predict(batch_pairs, show_progress_bar=False)
                all_scores.extend(scores.tolist())
            except Exception as e:
                # 如果预测失败，使用默认分数
                print(f"Warning: Cross-Encoder prediction failed for batch {i}: {e}")
                all_scores.extend([0.0] * len(batch_pairs))
        
        # 组合结果
        results = list(zip(candidate_video_ids, all_scores))
        
        # 如果提供了召回分数，进行加权融合
        if recall_scores:
            # 归一化Cross-Encoder分数到0-1范围
            if len(all_scores) > 0:
                max_ce_score = max(all_scores)
                min_ce_score = min(all_scores)
                ce_range = max_ce_score - min_ce_score if max_ce_score > min_ce_score else 1.0
                
                # 归一化召回分数
                recall_values = [recall_scores.get(vid, 0.0) for vid in candidate_video_ids]
                if len(recall_values) > 0:
                    max_recall = max(recall_values)
                    min_recall = min(recall_values)
                    recall_range = max_recall - min_recall if max_recall > min_recall else 1.0
                    
                    # 加权融合：Cross-Encoder 70% + 召回分数 30%
                    combined_results = []
                    for (vid, ce_score), recall_val in zip(results, recall_values):
                        normalized_ce = (ce_score - min_ce_score) / ce_range if ce_range > 0 else 0.5
                        normalized_recall = (recall_val - min_recall) / recall_range if recall_range > 0 else 0.5
                        combined_score = normalized_ce * 0.7 + normalized_recall * 0.3
                        combined_results.append((vid, combined_score))
                    results = combined_results
        
        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k
        if top_k is not None:
            results = results[:top_k]
        
        return [(vid, float(score)) for vid, score in results]
    
    def rank_with_query(
        self,
        query: str,
        candidate_video_ids: List[int],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        使用自定义查询对候选视频进行精排
        
        Args:
            query: 查询文本
            candidate_video_ids: 候选视频ID列表
            top_k: 返回top_k个结果
        
        Returns:
            List of (video_id, score) tuples
        """
        if self.model is None:
            print("Error: Cross-Encoder model not available.")
            return []
        
        if len(candidate_video_ids) == 0:
            return []
        
        # 构建查询-视频对
        pairs = []
        for video_id in candidate_video_ids:
            video_text = self._get_video_text(video_id)
            pairs.append([query, video_text])
        
        # 批量计算分数
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            scores = self.model.predict(batch_pairs, show_progress_bar=False)
            all_scores.extend(scores.tolist())
        
        # 组合结果并排序
        results = list(zip(candidate_video_ids, all_scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return [(vid, float(score)) for vid, score in results]


def main():
    """测试Cross-Encoder排序"""
    from pathlib import Path
    
    data_dir = Path("data")
    ratings_path = data_dir / "ratings.csv"
    videos_path = data_dir / "videos.csv"
    
    # 初始化
    cross_encoder = CrossEncoderRanker(ratings_path, videos_path, device="cpu")
    
    # 测试
    if len(cross_encoder.ratings) > 0:
        test_user_id = cross_encoder.ratings['user_id'].iloc[0]
        # 获取一些候选视频
        candidate_videos = cross_encoder.ratings['video_id'].unique()[:50]
        
        print(f"\nTesting Cross-Encoder ranking for user {test_user_id}")
        print(f"Ranking {len(candidate_videos)} candidate videos...")
        results = cross_encoder.rank(test_user_id, candidate_videos.tolist(), top_k=10)
        
        print(f"\nTop 10 results:")
        for vid, score in results:
            print(f"  Video {vid}: {score:.4f}")

if __name__ == "__main__":
    main()

