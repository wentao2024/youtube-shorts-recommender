import pandas as pd
from src.config import Config

class PopularityModel:
    def __init__(self):
        self.popular_items = []
        
    def fit(self, train_df):
        """
        Calculates the most popular items based on effective view counts.
        """
        # Count effective views per item
        item_counts = train_df[train_df[Config.LABEL_COL] == 1].groupby(Config.ITEM_ID_COL).size()
        
        # Sort by count descending
        sorted_items = item_counts.sort_values(ascending=False).reset_index()
        sorted_items.columns = [Config.ITEM_ID_COL, 'score']
        
        # Keep top K (enough to recommend to anyone)
        self.popular_items = sorted_items
        print(f"Popularity Baseline trained. Top item: {self.popular_items.iloc[0][Config.ITEM_ID_COL]} with score {self.popular_items.iloc[0]['score']}")
        
    def predict_rank(self, user_ids, k=10):
        """
        Returns the top K popular items for every user.
        Since this is non-personalized, everyone gets the same list.
        Returns: Dict {user_id: [item_id1, item_id2...]}
        """
        top_k_items = self.popular_items.head(k)[Config.ITEM_ID_COL].tolist()
        
        recommendations = {}
        for uid in user_ids:
            recommendations[uid] = top_k_items
            
        return recommendations

    def predict_proba(self, pair_df):
        """
        For AUC calculation: assign score based on global popularity.
        pair_df: DataFrame with user_id, item_id
        """
        # Merge with popularity scores
        # Fill missing items (cold start) with 0 score
        merged = pair_df.merge(self.popular_items, on=Config.ITEM_ID_COL, how='left')
        merged['score'] = merged['score'].fillna(0)
        
        # Normalize score to 0-1 for probability interpretation (MinMax or just raw score for AUC)
        # For AUC, raw score is fine.
        return merged['score'].values
