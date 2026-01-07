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
        # 使用预训练文本编码器
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.text_encoder = SentenceTransformer(text_encoder_name)
            text_dim = self.text_encoder.get_sentence_embedding_dimension()
        else:
            self.text_encoder = None
            text_dim = 384  # 默认维度
        
        input_dim = text_dim + metadata_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)  # 减少dropout
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重，使用Xavier初始化"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        
    def forward(self, video_texts: List[str], video_metadata: torch.Tensor, device: str = "cpu"):
        # 编码文本
        if self.text_encoder is not None:
            text_embeds = self.text_encoder.encode(
                video_texts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
        else:
            # 如果没有文本编码器，使用零向量
            text_embeds = torch.zeros(len(video_texts), 384, device=device)
        
        # 归一化文本embedding
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
        
    def forward(self, user_ids, video_texts, video_metadata, user_features=None, device="cpu"):
        user_embeds = self.user_tower(user_ids, user_features)
        video_embeds = self.video_tower(video_texts, video_metadata, device)
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
        learning_rate: float = 0.0001,
        negative_samples: int = 1,
        save_path: Optional[Path] = None,
        verbose: bool = True
    ):
        """训练双塔模型"""
        num_users = len(self.user_to_idx)
        num_videos = len(self.video_to_idx)
        
        self.model = TwoTowerModel(num_users).to(self.device)
        # 使用更小的学习率和权重衰减
        # 注意：learning_rate 已经很小了（0.0001），不需要再乘以0.01
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 构建训练数据
        print("Building training data...")
        train_data = []
        for _, row in tqdm(self.ratings.iterrows(), total=len(self.ratings), desc="Processing ratings"):
            user_idx = self.user_to_idx[row['user_id']]
            video_idx = self.video_to_idx[row['video_id']]
            train_data.append((user_idx, video_idx, row['rating']))
        
        print(f"Training on {len(train_data)} samples...")
        
        # 训练循环
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # 随机打乱数据
            np.random.shuffle(train_data)
            
            for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
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
                    
                    # 检查元数据是否有效
                    if torch.isnan(video_metadata).any() or torch.isinf(video_metadata).any():
                        continue
                    
                    # 获取用户特征
                    actual_user_ids = [self.idx_to_user[uid.item()] for uid in user_ids]
                    user_features = self._get_user_features(actual_user_ids)
                    
                    # 负采样（简化版本，每个正样本配一个负样本）
                    user_rated = {self.idx_to_video[vid] for vid in video_ids}
                    all_video_ids_list = list(self.video_to_idx.keys())
                    negative_video_ids_list = [vid for vid in all_video_ids_list if vid not in user_rated]
                    
                    if len(negative_video_ids_list) < len(batch):
                        # 如果负样本不够，允许重复
                        negative_video_ids_list = np.random.choice(
                            all_video_ids_list,
                            size=len(batch),
                            replace=True
                        ).tolist()
                    else:
                        negative_video_ids_list = np.random.choice(
                            negative_video_ids_list,
                            size=len(batch),
                            replace=False
                        ).tolist()
                    
                    negative_texts = [self._get_video_text(vid) for vid in negative_video_ids_list]
                    negative_metadata = self._get_video_metadata(negative_video_ids_list)
                    
                    # 检查负样本元数据是否有效
                    if torch.isnan(negative_metadata).any() or torch.isinf(negative_metadata).any():
                        continue
                    
                    # 前向传播 - 正样本
                    positive_scores, user_embeds, positive_video_embeds = self.model(
                        user_ids,
                        video_texts,
                        video_metadata,
                        user_features=user_features,
                        device=str(self.device)
                    )
                    
                    # 检查输出是否有效
                    if torch.isnan(positive_scores).any() or torch.isinf(positive_scores).any():
                        continue
                    
                    positive_scores = torch.diag(positive_scores)  # 取对角线（用户-视频对）
                    
                    # 前向传播 - 负样本
                    negative_scores, _, negative_video_embeds = self.model(
                        user_ids,
                        negative_texts,
                        negative_metadata,
                        user_features=user_features,
                        device=str(self.device)
                    )
                    
                    # 检查输出是否有效
                    if torch.isnan(negative_scores).any() or torch.isinf(negative_scores).any():
                        continue
                    
                    negative_scores = torch.diag(negative_scores)
                    
                    # BPR Loss（使用更稳定的计算方式）
                    # loss = -log(sigmoid(positive_score - negative_score))
                    score_diff = positive_scores - negative_scores
                    
                    # 限制score_diff的范围，防止数值溢出
                    score_diff = torch.clamp(score_diff, min=-10.0, max=10.0)
                    
                    # 使用更稳定的sigmoid计算
                    sigmoid_diff = torch.sigmoid(score_diff)
                    sigmoid_diff = torch.clamp(sigmoid_diff, min=1e-8, max=1.0 - 1e-8)
                    loss = -torch.log(sigmoid_diff).mean()
                    
                    # 检查损失是否有效
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                        continue
                    
                    # 梯度裁剪（防止梯度爆炸）
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    # 捕获任何异常，继续下一个批次
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Batches: {num_batches}")
        
        # 保存模型
        if save_path:
            self.save_model(save_path)
            print(f"Model saved to {save_path}")
    
    def recall(self, user_id: int, top_k: int = 100) -> List[Tuple[int, float]]:
        """使用双塔模型召回"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        self.model.eval()
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            # 用户不在映射中，返回空列表
            # 这通常发生在评估时，测试用户不在训练集中
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
            video_texts = [self._get_video_text(vid) for vid in batch_video_ids]
            video_metadata = self._get_video_metadata(batch_video_ids)
            
            with torch.no_grad():
                video_embeds = self.model.video_tower(video_texts, video_metadata, device=str(self.device))
                scores = torch.matmul(user_embed, video_embeds.t()).cpu().numpy()[0]
                all_scores.extend(zip(batch_video_ids, scores))
        
        # 排序并返回top_k
        all_scores.sort(key=lambda x: x[1], reverse=True)
        # 保留原始分数（不截断），用于更好的排序
        results = [(vid, float(score)) for vid, score in all_scores[:top_k]]
        
        return results
    
    def save_model(self, path: Path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_to_idx': self.user_to_idx,
            'video_to_idx': self.video_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_video': self.idx_to_video,
        }, path)
    
    def load_model(self, path: Path):
        """加载模型"""
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
        print(f"Model loaded from {path}")


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


