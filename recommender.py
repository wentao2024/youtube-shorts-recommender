"""
YouTube Shorts Recommendation System using Collaborative Filtering (SVD).
Optimized for low-memory systems (8GB RAM).
"""
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
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
    print(f"Loaded {len(ratings)} ratings.")
    return ratings


def create_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    """
    Create Surprise dataset from pandas DataFrame.
    Memory-efficient approach.
    """
    # Define rating scale (1-5 for MovieLens)
    reader = Reader(rating_scale=(1, 5))
    
    # Convert to Surprise format
    data = Dataset.load_from_df(
        ratings[['user_id', 'video_id', 'rating']],
        reader
    )
    
    return data


def train_model(
    data: Dataset,
    model_path: Path = Path("models/svd_model.pkl"),
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[SVD, Dataset]:
    """
    Train SVD model on the dataset.
    
    Args:
        data: Surprise Dataset object
        model_path: Path to save the trained model
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, testset)
    """
    print("Training SVD model...")
    
    # Split data into train and test sets
    trainset, testset = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )
    
    # Initialize SVD with memory-efficient parameters
    # n_factors: lower values use less memory
    # n_epochs: fewer epochs for faster training
    model = SVD(
        n_factors=50,      # Reduced from default 100 for lower memory
        n_epochs=20,       # Reduced from default 20
        lr_all=0.005,      # Learning rate
        reg_all=0.02,      # Regularization
        random_state=random_state
    )
    
    # Train the model
    model.fit(trainset)
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    return model, testset


def predict_rating(model: SVD, user_id: int, video_id: int) -> float:
    """
    Predict rating for a user-video pair.
    
    Args:
        model: Trained SVD model
        user_id: User ID
        video_id: Video ID
        
    Returns:
        Predicted rating
    """
    prediction = model.predict(user_id, video_id)
    return prediction.est


def get_top_recommendations(
    model: SVD,
    user_id: int,
    ratings: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """
    Get top N video recommendations for a user.
    
    Args:
        model: Trained SVD model
        user_id: User ID
        ratings: DataFrame with all ratings
        n: Number of recommendations to return
        
    Returns:
        DataFrame with top recommendations
    """
    # Get all unique video IDs
    all_videos = ratings['video_id'].unique()
    
    # Get videos already rated by user
    user_rated = ratings[ratings['user_id'] == user_id]['video_id'].values
    
    # Predict ratings for unrated videos only (memory efficient)
    predictions = []
    for video_id in all_videos:
        if video_id not in user_rated:
            pred_rating = predict_rating(model, user_id, video_id)
            predictions.append({
                'video_id': video_id,
                'predicted_rating': pred_rating
            })
    
    # Convert to DataFrame and sort
    recommendations = pd.DataFrame(predictions)
    recommendations = recommendations.sort_values(
        'predicted_rating',
        ascending=False
    ).head(n)
    
    return recommendations


def evaluate_model(model: SVD, testset: list) -> dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained SVD model
        testset: Test set from train_test_split
        
    Returns:
        Dictionary with evaluation metrics
    """
    from surprise import accuracy
    
    print("Evaluating model...")
    predictions = model.test(testset)
    
    # Calculate RMSE and MAE
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae
    }
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return metrics


def main():
    """Main function to train and evaluate the recommendation model."""
    # Load data
    ratings = load_ratings()
    
    # Create Surprise dataset
    data = create_surprise_dataset(ratings)
    
    # Train model
    model, testset = train_model(data)
    
    # Evaluate model
    metrics = evaluate_model(model, testset)
    
    # Example: Get recommendations for user 1
    print("\nGetting top 10 recommendations for user 1...")
    recommendations = get_top_recommendations(model, 1, ratings, n=10)
    print(recommendations)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

