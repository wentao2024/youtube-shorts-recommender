import pandas as pd
import numpy as np
from src.config import Config

def load_and_clean(sample_nrows=None):
    """
    Loads data, cleans it, and generates labels.
    """
    print("Loading data...")
    # Load User Features
    user_df = pd.read_csv(Config.USER_FEATURES_PATH)
    
    # Load Interactions
    inter_df = pd.read_csv(Config.INTERACTION_PATH, nrows=sample_nrows)
    
    # Basic Cleaning: Ensure duration is valid
    inter_df = inter_df[inter_df['duration_ms'] > 0].copy()
    
    # Label Generation: Effective View
    # Condition 1: Completion Rate > Threshold
    completion_rate = inter_df['playing_time'] / inter_df['duration_ms']
    is_completed = completion_rate >= Config.PLAY_COMPLETION_THRESHOLD
    
    # Condition 2: Any positive interaction
    has_interaction = (inter_df['like'] == 1) | \
                      (inter_df['follow'] == 1) | \
                      (inter_df['forward'] == 1)
                      
    inter_df[Config.LABEL_COL] = (is_completed | has_interaction).astype(int)
    
    # Merge User Features
    # Inner join to ensure we have features for all users in interaction log
    full_df = inter_df.merge(user_df, on=Config.USER_ID_COL, how='inner')
    
    # Sort by timestamp for time-based split
    full_df = full_df.sort_values(by=Config.TIMESTAMP_COL)
    
    print(f"Data loaded: {len(full_df)} rows.")
    return full_df

def train_test_split_by_time(df, test_ratio=0.2):
    """
    Splits the dataframe into train and test sets strictly by time.
    """
    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
    return train_df, test_df

class FeatureGenerator:
    """
    Helper to generate additional features if needed.
    """
    @staticmethod
    def get_item_stats(train_df):
        # Calculate item popularity and average duration from TRAIN set only
        item_stats = train_df.groupby(Config.ITEM_ID_COL).agg({
            Config.LABEL_COL: ['count', 'mean'], # count = popularity, mean = effective_rate
            'duration_ms': 'mean'
        })
        item_stats.columns = ['pop_count', 'eff_rate', 'avg_duration']
        return item_stats.reset_index()

if __name__ == "__main__":
    # Test run
    df = load_and_clean(sample_nrows=10000)
    train, test = train_test_split_by_time(df)
    stats = FeatureGenerator.get_item_stats(train)
    print(stats.head())
