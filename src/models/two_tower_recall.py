"""
Two-Tower Model for Recall
双塔模型：用户塔 + 视频塔
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle
from tqdm import tqdm

# 尝试导入 PyTorch（可能因为架构不兼容而失败）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"Warning: PyTorch not available ({type(e).__name__}: {e})")
    print("Two-Tower model will be disabled. Please install PyTorch for your architecture.")
    # 创建虚拟类以避免导入错误
    class nn:
        class Module:
            pass
        class Embedding:
            pass
        class Linear:
            pass
        class Dropout:
            pass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Text encoding will be disabled.")


class UserTower(nn.Module):
    """用户塔：编码用户特征（ID + 统计特征）"""
    def __init__(self, num_users: int, user_embedding_dim: int = 128, hidden_dim: int = 256, user_feature_dim: int = 5):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Cannot create UserTower.")
        super().__init__()
        # 增加模型容量以提高表达能力
        self.embedding = nn.Embedding(num_users, user_embedding_dim)
        # 用户统计特征（平均评分、评分数量、评分标准差等）
        self.user_feature_dim = user_feature_dim
        input_dim = user_embedding_dim + user_feature_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)  # 减少dropout，提高模型容量
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重，使用Xavier初始化（更适合深度学习）"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        
    def forward(self, user_ids, user_features=None):
        x = self.embedding(user_ids)
        # 如果有用户特征，拼接
        if user_features is not None:
            x = torch.cat([x, user_features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # 添加数值稳定性检查
        x = torch.clamp(x, min=-10.0, max=10.0)
        return F.normalize(x, p=2, dim=1)  # L2归一化


class VideoTower(nn.Module):
    """视频塔：编码视频特征（文本+元数据）"""
    def __init__(
        self,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        metadata_dim: int = 15,  # 增加元数据维度
        hidden_dim: int = 256  # 增加隐藏层维度
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Cannot create VideoTower.")
        super().__init__()
        # 预训练的文本编码器（不参与梯度更新）
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.text_encoder = SentenceTransformer(text_encoder_name)
            text_dim = self.text_encoder.get_sentence_embedding_dimension()
        else:
            self.text_encoder = None
            text_dim = 384  # 默认维度
        
        input_dim = text_dim + metadata_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.15)  # 减少dropout
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重，使用Xavier初始化"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        
    def forward(self, text_embeds: torch.Tensor, video_metadata: torch.Tensor):
        """
        前向传播
        
        Args:
            text_embeds: 预计算的文本embeddings (batch_size, text_dim)
            video_metadata: 视频元数据 (batch_size, metadata_dim)
        
        Note: 文本embeddings应该在模型外部预计算，这样能保证：
            1. 训练的确定性
            2. 避免不可微的SentenceTransformer
            3. 性能提升
        """
        # 确保text_embeds已经是torch tensor
        if not isinstance(text_embeds, torch.Tensor):
            text_embeds = torch.tensor(text_embeds, dtype=torch.float32)
        
        # 归一化文本embedding（如果还没有）
        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        
        # 拼接文本和元数据
        x = torch.cat([text_embeds, video_metadata], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # 添加数值稳定性检查
        x = torch.clamp(x, min=-10.0, max=10.0)
        return F.normalize(x, p=2, dim=1)  # L2归一化


class TwoTowerModel(nn.Module):
    """双塔模型"""
    def __init__(
        self,
        num_users: int,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 128,  # 增加embedding维度
        hidden_dim: int = 256  # 增加hidden维度
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Cannot create TwoTowerModel.")
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim, hidden_dim, user_feature_dim=5)
        self.video_tower = VideoTower(text_encoder_name, metadata_dim=15, hidden_dim=hidden_dim)
        self.embedding_dim = hidden_dim
        
    def forward(self, user_ids, text_embeds, video_metadata, user_features=None, device="cpu"):
        """
        前向传播
        
        Args:
            user_ids: 用户索引 (batch_size,)
            text_embeds: 预计算的文本embeddings (batch_size, text_dim)
            video_metadata: 视频元数据 (batch_size, metadata_dim)
            user_features: 用户特征 (batch_size, feature_dim)
            device: 设备
        
        Returns:
            scores: 相似度分数 (batch_size, batch_size)
            user_embeds: 用户embeddings (batch_size, hidden_dim)
            video_embeds: 视频embeddings (batch_size, hidden_dim)
        """
        user_embeds = self.user_tower(user_ids, user_features)
        video_embeds = self.video_tower(text_embeds, video_metadata)
        # 计算相似度（内积）
        scores = torch.matmul(user_embeds, video_embeds.t())
        return scores, user_embeds, video_embeds


class TwoTowerRecall:
    """双塔模型召回系统"""
    def __init__(
        self,
        ratings_path: Path,
        videos_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        device: str = "cpu"
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Please install PyTorch for your architecture.")
        self.device = torch.device(device)
        self.ratings = pd.read_csv(ratings_path)
        
        # 加载视频元数据
        if videos_path and videos_path.exists():
            self.videos = pd.read_csv(videos_path)
        else:
            self.videos = None
            print("Warning: videos.csv not found. Using minimal video metadata.")
        
        # 构建用户和视频映射（初始映射，可能被load_model覆盖）
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.ratings['user_id'].unique())}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.video_to_idx = {vid: idx for idx, vid in enumerate(self.ratings['video_id'].unique())}
        self.idx_to_video = {idx: vid for vid, idx in self.video_to_idx.items()}
        
        # 预计算用户统计特征（用于训练和推理）
        self._compute_user_features()
        
        # 预计算视频文本embeddings（关键！）
        self._precompute_text_embeddings()
        
        self.model = None
        if model_path and model_path.exists():
            self.load_model(model_path)
            # 加载模型后，重新计算用户特征（基于新的映射）
            # 但保持self.ratings为当前数据（用于特征计算）
            self._compute_user_features()
    
    def _compute_user_features(self):
        """预计算用户统计特征"""
        user_features = {}
        for user_id in self.user_to_idx.keys():
            user_ratings = self.ratings[self.ratings['user_id'] == user_id]
            rating_count = len(user_ratings)
            avg_rating = user_ratings['rating'].mean() if rating_count > 0 else 0.0
            rating_std = user_ratings['rating'].std() if rating_count > 0 else 0.0
            max_rating = user_ratings['rating'].max() if rating_count > 0 else 0.0
            min_rating = user_ratings['rating'].min() if rating_count > 0 else 0.0
            
            # 归一化特征
            features = [
                min(rating_count / 100.0, 1.0),  # 评分数量
                avg_rating / 5.0,  # 平均评分
                min(rating_std / 2.0, 1.0),  # 评分标准差
                max_rating / 5.0,  # 最高评分
                min_rating / 5.0,  # 最低评分
            ]
            user_features[user_id] = features
        
        self.user_features = user_features
    
    def _precompute_text_embeddings(self):
        """预计算所有视频的文本embeddings"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers not available. Text embeddings will not be precomputed.")
            self.text_embeddings = {}
            return
        
        print("Precomputing text embeddings for all videos...")
        self.text_embeddings = {}
        
        # 获取所有视频ID和文本
        all_video_ids = list(self.video_to_idx.keys())
        video_texts = []
        for vid in all_video_ids:
            text = self._get_video_text(vid)
            video_texts.append(text)
        
        # 使用SentenceTransformer编码所有文本（只做一次！）
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = encoder.encode(
            video_texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 存储embeddings
        for vid, emb in zip(all_video_ids, embeddings):
            self.text_embeddings[vid] = emb
        
        print(f"✓ Precomputed {len(self.text_embeddings)} text embeddings")
    
    def _get_user_features(self, user_ids: List[int]) -> torch.Tensor:
        """获取用户特征"""
        features = []
        for user_id in user_ids:
            if user_id in self.user_features:
                features.append(self.user_features[user_id])
            else:
                features.append([0.0] * 5)  # 默认特征
        
        result = torch.tensor(features, dtype=torch.float32, device=self.device)
        result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
        return result
    
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
    
    def _get_video_metadata(self, video_ids: List[int]) -> torch.Tensor:
        """获取视频元数据特征（增强版）"""
        metadata = []
        for vid in video_ids:
            video_ratings = self.ratings[self.ratings['video_id'] == vid]
            rating_count = len(video_ratings)
            avg_rating = video_ratings['rating'].mean() if rating_count > 0 else 0.0
            rating_std = video_ratings['rating'].std() if rating_count > 0 else 0.0
            max_rating = video_ratings['rating'].max() if rating_count > 0 else 0.0
            min_rating = video_ratings['rating'].min() if rating_count > 0 else 0.0
            
            # 计算用户多样性（不同用户数量）
            unique_users = video_ratings['user_id'].nunique() if rating_count > 0 else 0
            
            # 如果有videos.csv，添加更多特征
            if self.videos is not None:
                video_row = self.videos[self.videos['video_id'] == vid]
                if not video_row.empty:
                    # 可以添加类别、年份等特征（如果有）
                    has_metadata = 1.0
                else:
                    has_metadata = 0.0
            else:
                has_metadata = 0.0
            
            # 归一化特征
            features = [
                min(rating_count / 100.0, 1.0),  # 评分数量
                avg_rating / 5.0,  # 平均评分
                min(rating_std / 2.0, 1.0),  # 评分标准差
                max_rating / 5.0,  # 最高评分
                min_rating / 5.0,  # 最低评分
                min(unique_users / 50.0, 1.0),  # 用户多样性
                has_metadata,  # 是否有元数据
                # 添加更多统计特征
                min(rating_count / 200.0, 1.0) if rating_count > 0 else 0.0,  # 评分数量（不同归一化）
                (avg_rating - 2.5) / 2.5 if rating_count > 0 else 0.0,  # 中心化的平均评分
                min(rating_std / 1.0, 1.0) if rating_count > 0 else 0.0,  # 评分标准差（不同归一化）
                # 添加时间相关特征（如果有时间戳）
                0.0, 0.0, 0.0, 0.0, 0.0  # 预留位置
            ]
            metadata.append(features[:15])  # 确保是15维
        
        result = torch.tensor(metadata, dtype=torch.float32, device=self.device)
        result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
        return result
    
    def train_epoch(
        self,
        train_data: List,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 256,
        negative_samples: int = 1,
        margin: float = 0.0,
        verbose: bool = True
    ) -> float:
        """训练一个epoch（用于外部优化器控制）"""
        total_loss = 0
        num_batches = 0
        
        # 随机打乱数据
        np.random.shuffle(train_data)
        
        # 使用tqdm显示进度
        total_batches = (len(train_data) + batch_size - 1) // batch_size
        iterator = range(0, len(train_data), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Training", total=total_batches, unit="batch")
        
        for i in iterator:
            batch = train_data[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            try:
                # 正样本
                user_ids = torch.tensor([x[0] for x in batch], device=self.device, dtype=torch.long)
                video_ids = [x[1] for x in batch]
                
                # 获取视频文本和元数据
                video_texts = [self._get_video_text(self.idx_to_video[vid]) for vid in video_ids]
                video_metadata = self._get_video_metadata([self.idx_to_video[vid] for vid in video_ids])
                
                if torch.isnan(video_metadata).any() or torch.isinf(video_metadata).any():
                    continue
                
                # 负采样（支持多个负样本）
                user_rated = {self.idx_to_video[vid] for vid in video_ids}
                all_video_ids_list = list(self.video_to_idx.keys())
                negative_video_ids_list = [vid for vid in all_video_ids_list if vid not in user_rated]
                
                # 为每个正样本采样多个负样本
                negative_samples_list = []
                for _ in range(negative_samples):
                    if len(negative_video_ids_list) < len(batch):
                        neg_ids = np.random.choice(
                            all_video_ids_list,
                            size=len(batch),
                            replace=True
                        ).tolist()
                    else:
                        neg_ids = np.random.choice(
                            negative_video_ids_list,
                            size=len(batch),
                            replace=False
                        ).tolist()
                    negative_samples_list.append(neg_ids)
                
                # 获取用户特征
                actual_user_ids = [self.idx_to_user[uid.item()] for uid in user_ids]
                user_features = self._get_user_features(actual_user_ids)
                
                # 计算正样本分数
                with torch.set_grad_enabled(True):  # 确保梯度计算开启
                    positive_scores, _, _ = self.model(
                        user_ids, video_texts, video_metadata, user_features=user_features, device=str(self.device)
                    )
                    positive_scores = torch.diag(positive_scores)
                    
                    # 计算所有负样本的分数并取平均
                    negative_scores_list = []
                    for neg_ids in negative_samples_list:
                        neg_texts = [self._get_video_text(vid) for vid in neg_ids]
                        neg_metadata = self._get_video_metadata(neg_ids)
                        
                        if torch.isnan(neg_metadata).any() or torch.isinf(neg_metadata).any():
                            continue
                        
                        neg_scores, _, _ = self.model(
                            user_ids, neg_texts, neg_metadata, user_features=user_features, device=str(self.device)
                        )
                        negative_scores_list.append(torch.diag(neg_scores))
                
                if len(negative_scores_list) == 0:
                    continue
                
                # 平均所有负样本的分数
                negative_scores = torch.stack(negative_scores_list).mean(dim=0)
                
                # BPR Loss with margin
                score_diff = positive_scores - negative_scores - margin
                score_diff = torch.clamp(score_diff, min=-10.0, max=10.0)
                sigmoid_diff = torch.sigmoid(score_diff)
                sigmoid_diff = torch.clamp(sigmoid_diff, min=1e-8, max=1.0 - 1e-8)
                loss = -torch.log(sigmoid_diff).mean()
                
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(
        self,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        negative_samples: int = 4,  # 增加负样本数量
        save_path: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        训练双塔模型（改进版）
        
        Key improvements:
        1. 使用预计算的文本embeddings（确定性，可微分）
        2. 改进的BPR loss（数值稳定）
        3. 多负样本采样
        4. 学习率调度
        5. 详细的训练监控
        """
        num_users = len(self.user_to_idx)
        num_videos = len(self.video_to_idx)
        
        print("=" * 70)
        print("Training Two-Tower Model (Improved)")
        print("=" * 70)
        print(f"Users: {num_users}, Videos: {num_videos}")
        print(f"Training samples: {len(self.ratings)}")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        print(f"Negative samples per positive: {negative_samples}")
        print("=" * 70)
        
        # 初始化模型
        self.model = TwoTowerModel(num_users).to(self.device)
        
        # 优化器 - 使用更稳定的超参数
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器 - 动态调整学习率
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=max(2, epochs // 3), 
            gamma=0.5
        )
        
        # 构建训练数据
        print("\nBuilding training data...")
        train_data = []
        for _, row in self.ratings.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            video_id = row['video_id']
            train_data.append((user_idx, video_id))
        
        # 创建视频ID到索引的映射（加速）
        video_id_to_idx = self.video_to_idx
        
        # 获取所有视频ID列表（用于负采样）
        all_video_ids = list(self.video_to_idx.keys())
        
        print(f"Total training samples: {len(train_data)}")
        
        # 训练循环
        best_loss = float('inf')
        no_improve_count = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # 随机打乱数据
            np.random.shuffle(train_data)
            
            # 进度条
            pbar = tqdm(range(0, len(train_data), batch_size), 
                       desc=f"Epoch {epoch+1}/{epochs}",
                       unit="batch") if verbose else range(0, len(train_data), batch_size)
            
            for batch_start in pbar:
                batch_end = min(batch_start + batch_size, len(train_data))
                batch = train_data[batch_start:batch_end]
                
                if len(batch) == 0:
                    continue
                
                try:
                    # 提取正样本
                    user_idxs = [x[0] for x in batch]
                    pos_video_ids = [x[1] for x in batch]
                    
                    user_ids_tensor = torch.tensor(user_idxs, device=self.device, dtype=torch.long)
                    user_features = self._get_user_features([self.idx_to_user[uid] for uid in user_idxs])
                    
                    # 获取正样本文本embeddings
                    pos_text_embeds = []
                    for vid in pos_video_ids:
                        if vid in self.text_embeddings:
                            pos_text_embeds.append(self.text_embeddings[vid])
                        else:
                            pos_text_embeds.append(np.zeros(384))
                    pos_text_embeds = torch.tensor(np.array(pos_text_embeds), 
                                                   device=self.device, dtype=torch.float32)
                    
                    # 获取正样本元数据
                    pos_metadata = self._get_video_metadata(pos_video_ids)
                    
                    # 前向传播 - 正样本
                    pos_scores, user_embeds, pos_video_embeds = self.model(
                        user_ids_tensor,
                        pos_text_embeds,
                        pos_metadata,
                        user_features=user_features,
                        device=str(self.device)
                    )
                    
                    # 取对角线（用户-对应视频对的分数）
                    pos_scores = torch.diag(pos_scores)
                    
                    # 负采样 - 每个正样本采样多个负样本
                    loss = torch.tensor(0.0, device=self.device)
                    
                    for neg_sample_idx in range(negative_samples):
                        # 负采样
                        neg_video_ids = []
                        pos_video_set = set(pos_video_ids)
                        
                        for _ in range(len(batch)):
                            # 采样一个未交互的视频
                            candidates = [v for v in all_video_ids if v not in pos_video_set]
                            if candidates:
                                neg_vid = np.random.choice(candidates)
                            else:
                                neg_vid = np.random.choice(all_video_ids)
                            neg_video_ids.append(neg_vid)
                        
                        # 获取负样本文本embeddings
                        neg_text_embeds = []
                        for vid in neg_video_ids:
                            if vid in self.text_embeddings:
                                neg_text_embeds.append(self.text_embeddings[vid])
                            else:
                                neg_text_embeds.append(np.zeros(384))
                        neg_text_embeds = torch.tensor(np.array(neg_text_embeds),
                                                       device=self.device, dtype=torch.float32)
                        
                        # 获取负样本元数据
                        neg_metadata = self._get_video_metadata(neg_video_ids)
                        
                        # 前向传播 - 负样本
                        neg_scores, _, _ = self.model(
                            user_ids_tensor,
                            neg_text_embeds,
                            neg_metadata,
                            user_features=user_features,
                            device=str(self.device)
                        )
                        
                        # 取对角线
                        neg_scores = torch.diag(neg_scores)
                        
                        # 改进的BPR Loss - 使用log-sigmoid以提高数值稳定性
                        # loss = -log(sigmoid(pos - neg))
                        score_diff = pos_scores - neg_scores
                        
                        # 数值稳定的log-sigmoid计算
                        # log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x)) 当x>0
                        #                  = x - softplus(x) = x - log(1 + exp(x)) 当x<0
                        log_sigmoid_loss = -F.softplus(-score_diff)
                        batch_loss = -log_sigmoid_loss.mean()
                        
                        loss = loss + batch_loss / negative_samples
                    
                    # 检查损失是否有效
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                        if verbose:
                            print(f"  Warning: Invalid loss detected: {loss.item()}, skipping batch")
                        continue
                    
                    # 梯度更新
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if verbose and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    if verbose:
                        print(f"  Error in batch: {e}")
                    continue
            
            # 计算平均损失
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            # 打印epoch结果
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Batches: {num_batches}")
            
            # 学习率调度
            scheduler.step()
            
            # 早停检测
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
                
                # 保存最佳模型
                if save_path:
                    self.save_model(save_path)
                    if verbose:
                        print(f"  ✓ Best model saved (loss: {best_loss:.4f})")
            else:
                no_improve_count += 1
                if no_improve_count >= 3:
                    if verbose:
                        print(f"\nEarly stopping: No improvement for 3 epochs")
                    break
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best loss: {best_loss:.4f}")
        if save_path:
            print(f"Model saved to: {save_path}")
        print("=" * 70)
    
    def recall(self, user_id: int, top_k: int = 100) -> List[Tuple[int, float]]:
        """使用双塔模型召回（改进版，使用预计算embeddings）"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        self.model.eval()
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            # 用户不在映射中，返回空列表
            return []
        
        # 获取用户embedding和特征
        user_tensor = torch.tensor([user_idx], device=self.device)
        user_features = self._get_user_features([user_id])
        with torch.no_grad():
            user_embed = self.model.user_tower(user_tensor, user_features)
        
        # 批量处理所有视频（避免内存溢出）
        all_video_ids = list(self.video_to_idx.keys())
        batch_size = 1000
        all_scores = []
        
        for i in range(0, len(all_video_ids), batch_size):
            batch_video_ids = all_video_ids[i:i+batch_size]
            
            # 获取预计算的文本embeddings
            text_embeds = []
            for vid in batch_video_ids:
                if vid in self.text_embeddings:
                    text_embeds.append(self.text_embeddings[vid])
                else:
                    text_embeds.append(np.zeros(384))
            text_embeds = torch.tensor(np.array(text_embeds), device=self.device, dtype=torch.float32)
            
            # 获取视频元数据
            video_metadata = self._get_video_metadata(batch_video_ids)
            
            with torch.no_grad():
                video_embeds = self.model.video_tower(text_embeds, video_metadata)
                scores = torch.matmul(user_embed, video_embeds.t()).cpu().numpy()[0]
                all_scores.extend(zip(batch_video_ids, scores))
        
        # 排序并返回top_k
        all_scores.sort(key=lambda x: x[1], reverse=True)
        results = [(vid, float(score)) for vid, score in all_scores[:top_k]]
        
        return results
    
    def save_model(self, path: Path):
        """保存模型和预计算的embeddings"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_to_idx': self.user_to_idx,
            'video_to_idx': self.video_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_video': self.idx_to_video,
            'text_embeddings': self.text_embeddings,  # 保存预计算的embeddings
            'user_features': self.user_features,
        }, path)
    
    def load_model(self, path: Path):
        """加载模型和预计算的embeddings"""
        # PyTorch 2.6+ requires weights_only=False for models with numpy objects
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(path, map_location=self.device)
        num_users = len(checkpoint['user_to_idx'])
        self.model = TwoTowerModel(num_users).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.user_to_idx = checkpoint['user_to_idx']
        self.video_to_idx = checkpoint['video_to_idx']
        self.idx_to_user = checkpoint.get('idx_to_user', {v: k for k, v in self.user_to_idx.items()})
        self.idx_to_video = checkpoint.get('idx_to_video', {v: k for k, v in self.video_to_idx.items()})
        self.text_embeddings = checkpoint.get('text_embeddings', {})  # 加载embeddings
        self.user_features = checkpoint.get('user_features', {})
        print(f"Model loaded from {path}")
        print(f"  Loaded {len(self.text_embeddings)} text embeddings")


def main():
    """测试双塔模型"""
    from pathlib import Path
    
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    ratings_path = data_dir / "ratings.csv"
    videos_path = data_dir / "videos.csv"
    model_path = models_dir / "two_tower_model.pth"
    
    # 初始化
    two_tower = TwoTowerRecall(ratings_path, videos_path, device="cpu")
    
    # 训练
    print("Training Two-Tower Model...")
    two_tower.train(epochs=5, batch_size=128, save_path=model_path)
    
    # 测试召回
    test_user_id = two_tower.idx_to_user[0]
    print(f"\nTesting recall for user {test_user_id}...")
    results = two_tower.recall(test_user_id, top_k=10)
    print(f"Top 10 recommendations:")
    for vid, score in results:
        print(f"  Video {vid}: {score:.4f}")

if __name__ == "__main__":
    main()


