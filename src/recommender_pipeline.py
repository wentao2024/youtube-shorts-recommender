import pandas as pd
import numpy as np
# import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from src.config import Config
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class RecallModel:
    def __init__(self):
        self.item_sim_matrix = defaultdict(dict)
        self.popular_items = []
        
    def fit(self, df):
        """
        Builds Item-Item co-occurrence matrix.
        df: train interaction data
        """
        print("Training Recall Model (ItemCF)...")
        # 1. Global Popularity (Fallback)
        self.popular_items = df[Config.LABEL_COL].groupby(df[Config.ITEM_ID_COL]).sum().sort_values(ascending=False).head(Config.RECALL_CANDIDATES_NUM).index.tolist()
        
        # 2. Simple Item-CF
        # Group items by user to find co-occurrences
        positive_df = df[df[Config.LABEL_COL] == 1]
        user_histories = positive_df.groupby(Config.USER_ID_COL)[Config.ITEM_ID_COL].apply(list)
        
        # Count co-occurrences (simplified for speed)
        co_counts = defaultdict(lambda: defaultdict(int))
        for history in user_histories:
            for i in range(len(history)):
                for j in range(i+1, len(history)):
                    item_i = history[i]
                    item_j = history[j]
                    co_counts[item_i][item_j] += 1
                    co_counts[item_j][item_i] += 1
        
        # Normalize to get similarity (conditional probability approach roughly)
        for i, related in co_counts.items():
            sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)[:20] # Keep top 20 neighbors
            self.item_sim_matrix[i] = {k: v for k, v in sorted_related}
            
    def retrieve(self, user_history_items, n=100):
        """
        Retrieves candidates based on user history.
        """
        candidates = defaultdict(float)
        
        # Method 1: ItemCF
        # For each item in recent history, add its neighbors
        for item in user_history_items[-5:]: # Use last 5 items
            if item in self.item_sim_matrix:
                for neighbor, score in self.item_sim_matrix[item].items():
                    candidates[neighbor] += score
        
        # Method 2: Fallback to Popularity
        for item in self.popular_items:
            if item not in candidates:
                candidates[item] = 0.001 # Small score for popularity fallback
                
        # Sort and return top N
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:n]
        return [x[0] for x in sorted_cands]

class RoughRanker:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', max_iter=100)
        self.le_feat1 = LabelEncoder()
        
    def fit(self, train_df):
        print("Training Rough Ranker (LogReg)...")
        # Features: numerical stats + onehot_feat1 (encoded) + duration
        X = self._extract_features(train_df)
        y = train_df[Config.LABEL_COL]
        self.model.fit(X, y)
        
    def predict(self, df):
        X = self._extract_features(df)
        return self.model.predict_proba(X)[:, 1]
        
    def _extract_features(self, df):
        # selected simple features
        return df[['duration_ms', 'reco_active_level', 'search_active_level']].fillna(0)

class FineRanker:
    def __init__(self):
        self.model = PredictionModelError = None
        # Using Sklearn GBDT as fallback for LGBM (due to missing libomp on some macs)
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.le_dict = {}

    def fit(self, train_df):
        print("Training Fine Ranker (Sklearn GBDT)...")
        
        # Use more features
        features = Config.USER_FEAT_COLS + ['duration_ms']
        
        X = train_df[features].copy()
        y = train_df[Config.LABEL_COL]
        
        # Simple Label Encoding for Sklearn
        cat_cols = ['onehot_feat1', 'onehot_feat2']
        for col in cat_cols:
            le = LabelEncoder()
            # Handle unknown categories by fitting on string conversion or similar (simplified here)
            X[col] = le.fit_transform(X[col].astype(str))
            self.le_dict[col] = le
            
        self.model.fit(X, y)
        
    def predict(self, df):
        features = Config.USER_FEAT_COLS + ['duration_ms']
        X = df[features].copy()
        
        cat_cols = ['onehot_feat1', 'onehot_feat2']
        for col in cat_cols:
            if col in self.le_dict:
                le = self.le_dict[col]
                # Handle unseen labels by mapping to a safe value or keeping as is if numeric (simplified)
                # For demo, we just apply transform and clip errors or re-fit (imperfect but works for demo)
                # Better: reuse dict mapping
                X[col] = X[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
            else:
                X[col] = 0
                
        return self.model.predict_proba(X)[:, 1]

class FullPipeline:
    def __init__(self):
        self.recall = RecallModel()
        self.rough = RoughRanker()
        self.fine = FineRanker()
        
    def train(self, train_df):
        # 1. Train Recall on all data
        self.recall.fit(train_df)
        
        # 2. Train Rough Ranker
        # In reality, we should construct a dataset with negative samples for rankers.
        # Here we just use the clicked/non-clicked samples from history as "Ranker Training Data"
        self.rough.fit(train_df)
        
        # 3. Train Fine Ranker
        self.fine.fit(train_df)
        
    def predict(self, user_features_df, all_user_history_map):
        """
        Simulate inference for test users.
        """
        pass
        
    def score_test_set_auc(self, test_df):
        return self.fine.predict(test_df)
