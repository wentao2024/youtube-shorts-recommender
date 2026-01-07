"""
BM25 Ranking for Coarse Ranking
使用BM25算法进行粗排
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle
import re

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available. BM25 ranking will be disabled.")


class BM25Ranker:
    """BM25粗排系统"""
    def __init__(
        self,
        ratings_path: Path,
        videos_path: Optional[Path] = None,
        model_path: Optional[Path] = None
    ):
        self.ratings = pd.read_csv(ratings_path)
        
        # 加载视频元数据
        if videos_path and videos_path.exists():
            self.videos = pd.read_csv(videos_path)
        else:
            self.videos = None
            print("Warning: videos.csv not found. Using minimal video metadata.")
        
        self.bm25 = None
        self.video_ids = []
        self.video_texts = []
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            self._build_index()
            if model_path:
                self.save_model(model_path)
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词函数"""
        if not isinstance(text, str):
            text = str(text)
        # 转换为小写，移除标点，分词
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens
    
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
    
    def _build_index(self):
        """构建BM25索引"""
        if not BM25_AVAILABLE:
            print("Error: rank-bm25 not available. Cannot build BM25 index.")
            return
        
        print("Building BM25 index...")
        unique_videos = self.ratings['video_id'].unique()
        
        self.video_ids = []
        self.video_texts = []
        tokenized_texts = []
        
        for video_id in unique_videos:
            text = self._get_video_text(video_id)
            tokens = self._tokenize(text)
            
            self.video_ids.append(video_id)
            self.video_texts.append(text)
            tokenized_texts.append(tokens)
        
        if len(tokenized_texts) > 0:
            self.bm25 = BM25Okapi(tokenized_texts)
            print(f"BM25 index built for {len(self.video_ids)} videos")
        else:
            print("Warning: No videos found to index")
    
    def rank(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        根据查询文本对视频进行BM25排序
        
        Args:
            query: 查询文本（可以是用户历史观看的视频标题组合，或用户偏好描述）
            top_k: 返回top_k个结果
        
        Returns:
            List of (video_id, score) tuples
        """
        if self.bm25 is None:
            print("Error: BM25 index not built. Cannot perform ranking.")
            return []
        
        # 分词查询
        query_tokens = self._tokenize(query)
        if len(query_tokens) == 0:
            return []
        
        # 计算BM25分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 排序并返回top_k
        scored_videos = list(zip(self.video_ids, scores))
        scored_videos.sort(key=lambda x: x[1], reverse=True)
        
        results = [(vid, float(score)) for vid, score in scored_videos[:top_k]]
        return results
    
    def rank_by_user_history(self, user_id: int, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        根据用户历史观看记录构建查询，进行BM25排序
        
        Args:
            user_id: 用户ID
            top_k: 返回top_k个结果
        
        Returns:
            List of (video_id, score) tuples
        """
        # 获取用户历史观看的视频
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        if len(user_ratings) == 0:
            return []
        
        # 获取用户观看过的视频文本，构建查询
        watched_video_ids = user_ratings['video_id'].tolist()
        query_parts = []
        
        for vid in watched_video_ids[:20]:  # 最多使用20个历史视频
            text = self._get_video_text(vid)
            query_parts.append(text)
        
        query = " ".join(query_parts)
        return self.rank(query, top_k)
    
    def save_model(self, path: Path):
        """保存BM25模型"""
        if self.bm25 is None:
            print("Warning: BM25 index not built. Nothing to save.")
            return
        
        data = {
            'video_ids': self.video_ids,
            'video_texts': self.video_texts,
            'bm25': self.bm25
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"BM25 model saved to {path}")
    
    def load_model(self, path: Path):
        """加载BM25模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.video_ids = data['video_ids']
        self.video_texts = data['video_texts']
        self.bm25 = data['bm25']
        print(f"BM25 model loaded from {path}")


def main():
    """测试BM25排序"""
    from pathlib import Path
    
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    ratings_path = data_dir / "ratings.csv"
    videos_path = data_dir / "videos.csv"
    model_path = models_dir / "bm25_model.pkl"
    
    # 初始化
    bm25_ranker = BM25Ranker(ratings_path, videos_path, model_path)
    
    # 测试查询
    test_query = "action adventure movie"
    print(f"\nTesting BM25 ranking with query: '{test_query}'")
    results = bm25_ranker.rank(test_query, top_k=10)
    print(f"Top 10 results:")
    for vid, score in results:
        print(f"  Video {vid}: {score:.4f}")
    
    # 测试用户历史
    if len(bm25_ranker.ratings) > 0:
        test_user_id = bm25_ranker.ratings['user_id'].iloc[0]
        print(f"\nTesting BM25 ranking for user {test_user_id} (based on history)")
        results = bm25_ranker.rank_by_user_history(test_user_id, top_k=10)
        print(f"Top 10 results:")
        for vid, score in results:
            print(f"  Video {vid}: {score:.4f}")

if __name__ == "__main__":
    main()



