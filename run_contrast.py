import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from src.data_loader import load_and_clean, train_test_split_by_time
from src.baseline import PopularityModel
from src.recommender_pipeline import FullPipeline
from src.config import Config

def evaluate_model(y_true, y_scores, name="Model"):
    auc = roc_auc_score(y_true, y_scores)
    print(f"[{name}] AUC: {auc:.4f}")
    return auc

def main():
    print("=== Starting Recommender System Contrast Project ===")
    
    # 1. Load Data
    # Using 100k rows to ensure speed for demonstration. Remove sample_nrows for full run.
    full_df = load_and_clean(sample_nrows=500000) 
    
    print(f"Total Interactions: {len(full_df)}")
    print(f"Effective View Ratio: {full_df[Config.LABEL_COL].mean():.2%}")
    
    # 2. Split Data
    train_df, test_df = train_test_split_by_time(full_df)
    
    # 3. Baseline Model
    print("\n--- Training Baseline (Popularity) ---")
    baseline = PopularityModel()
    baseline.fit(train_df)
    
    # Evaluate Baseline
    print("Evaluating Baseline...")
    # Baseline scores are just popularity scores for the items in test set
    base_scores = baseline.predict_proba(test_df[[Config.USER_ID_COL, Config.ITEM_ID_COL]])
    base_auc = evaluate_model(test_df[Config.LABEL_COL], base_scores, name="Baseline")
    
    # 4. Advanced Pipeline
    print("\n--- Training Advanced Pipeline (Recall -> Rough -> Fine) ---")
    pipeline = FullPipeline()
    pipeline.train(train_df)
    
    # Evaluate Advanced Pipeline (Fine Ranker Scoring)
    print("Evaluating Advanced Recommender...")
    adv_scores = pipeline.score_test_set_auc(test_df)
    adv_auc = evaluate_model(test_df[Config.LABEL_COL], adv_scores, name="Advanced Recommender")
    
    # 5. Contrast Report
    print("\n=== Contrast Results ===")
    print(f"Baseline AUC: {base_auc:.4f}")
    print(f"Advanced AUC: {adv_auc:.4f}")
    improvement = (adv_auc - base_auc) / base_auc * 100
    print(f"Improvement: {improvement:.2f}%")
    
    if adv_auc > base_auc:
        print("SUCCESS: The Advanced Recommender outperformed the Baseline.")
    else:
        print("NOTE: improvement not observed. Tune features or increase data size.")

    # 6. Pipeline Demonstration (Inference Check)
    print("\n--- Pipeline Inference Demo (User ID: 6504) ---")
    # Just checking if we can retrieve and rank for a sample user
    sample_user_history = train_df[train_df[Config.USER_ID_COL] == 6504][Config.ITEM_ID_COL].tolist()
    if not sample_user_history:
        print("User 6504 not in training set, skipping demo.")
    else:
        candidates = pipeline.recall.retrieve(sample_user_history, n=50)
        print(f"Recall Stage retrieved {len(candidates)} candidates.")
        # Simulating Rough Ranker scoring on candidates (creating dummy features df for them)
        # In a real app, we'd fetch item features here.
        # This part is just a text confirmation of the architecture.
        print("Rough Ranker would filter top 20.")
        print("Fine Ranker would re-rank top 20.")

if __name__ == "__main__":
    main()
