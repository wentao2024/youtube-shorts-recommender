"""
Train SVD model on full MovieLens 100K dataset.
"""
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from pathlib import Path
import pickle


def load_ratings(data_path: Path = Path("data/ratings.csv")) -> pd.DataFrame:
    """Load processed ratings data."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found at {data_path}. Please run data_prep.py first."
        )
    
    print(f"Loading ratings from {data_path}...")
    ratings = pd.read_csv(data_path)
    print(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users and {ratings['video_id'].nunique()} videos.")
    return ratings


def create_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    """
    Create Surprise dataset from pandas DataFrame.
    
    Args:
        ratings: DataFrame with columns user_id, video_id, rating
        
    Returns:
        Surprise Dataset object
    """
    # Define rating scale (1-5 for MovieLens)
    reader = Reader(rating_scale=(1, 5))
    
    # Convert to Surprise format
    data = Dataset.load_from_df(
        ratings[['user_id', 'video_id', 'rating']],
        reader
    )
    
    return data


def train_svd_model(
    data: Dataset,
    model_path: Path = Path("models/svd_model.pkl"),
    random_state: int = 42
) -> SVD:
    """
    Train SVD model on full dataset.
    
    Args:
        data: Surprise Dataset object
        model_path: Path to save the trained model
        random_state: Random seed for reproducibility
        
    Returns:
        Trained SVD model
    """
    print("Training SVD model on full dataset...")
    
    # Build full trainset (no test split)
    trainset = data.build_full_trainset()
    
    # Initialize SVD with memory-efficient parameters
    model = SVD(
        n_factors=50,      # Reduced for lower memory usage
        n_epochs=20,
        lr_all=0.005,      # Learning rate
        reg_all=0.02,      # Regularization
        random_state=random_state
    )
    
    # Train the model
    model.fit(trainset)
    print("Model training complete!")
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    return model


def evaluate_model(data: Dataset, model: SVD) -> dict:
    """
    Evaluate model using cross-validation with RMSE.
    
    Args:
        data: Surprise Dataset object
        model: Trained SVD model
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating model with 5-fold cross-validation...")
    
    # Perform cross-validation
    cv_results = cross_validate(
        model,
        data,
        measures=['RMSE', 'MAE'],
        cv=5,
        verbose=True
    )
    
    # Calculate average metrics
    avg_rmse = cv_results['test_rmse'].mean()
    avg_mae = cv_results['test_mae'].mean()
    
    metrics = {
        'RMSE': avg_rmse,
        'MAE': avg_mae,
        'RMSE_std': cv_results['test_rmse'].std(),
        'MAE_std': cv_results['test_mae'].std()
    }
    
    print(f"\nCross-validation results:")
    print(f"Average RMSE: {avg_rmse:.4f} (+/- {cv_results['test_rmse'].std():.4f})")
    print(f"Average MAE: {avg_mae:.4f} (+/- {cv_results['test_mae'].std():.4f})")
    
    return metrics 
    


def main():
    """Main function to train and evaluate the model."""
    # Load data
    ratings = load_ratings()
    
    # Create Surprise dataset
    data = create_surprise_dataset(ratings)
    
    # Train model on full dataset
    model = train_svd_model(data)
    
    # Evaluate model with cross-validation
    metrics = evaluate_model(data, model)
    
    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()

