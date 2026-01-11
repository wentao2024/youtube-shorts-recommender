import os

class Config:
    # ROI
    # Use absolute path to be safe, or relative to running directory
    # Assuming run from project root
    DATA_DIR = './data'
    USER_FEATURES_PATH = os.path.join(DATA_DIR, 'user_features.csv')
    INTERACTION_PATH = os.path.join(DATA_DIR, 'rec_inter.csv')
    
    # Feature Columns
    USER_ID_COL = 'user_id'
    ITEM_ID_COL = 'item_id'
    TIMESTAMP_COL = 'timestamp'
    LABEL_COL = 'is_effective'
    
    # User features to use
    USER_FEAT_COLS = ['onehot_feat1', 'onehot_feat2', 'search_active_level', 'reco_active_level']
    
    # Interaction cols relevant for label definition
    INTERACTION_COLS = ['playing_time', 'duration_ms', 'like', 'follow', 'forward']
    
    # Thresholds
    # If playing_time / duration_ms > 0.8 OR any positive interaction -> Effective View
    PLAY_COMPLETION_THRESHOLD = 0.8
    
    # Pipeline Params
    RECALL_CANDIDATES_NUM = 100
    ROUGH_RANK_NUM = 50
    FINE_RANK_NUM = 10
    
    # Random Seed
    SEED = 42
