"""
Get video recommendations for users using trained SVD model.
Includes fallback for cold start users (popular items).
"""
import pandas as pd
from surprise import SVD
from pathlib import Path
import pickle


def load_movie_titles(dataset_dir: Path = Path("data/ml-100k")) -> dict[int, str]:
    """
    Load movie titles from u.item file.
    
    Args:
        dataset_dir: Path to ml-100k directory
        
    Returns:
        Dictionary mapping video_id (movie_id) to movie title
    """
    u_item_path = dataset_dir / "u.item"
    
    if not u_item_path.exists():
        raise FileNotFoundError(
            f"u.item file not found at {u_item_path}. Please run data_prep.py first."
        )
    
    # u.item format: movie_id | movie_title | release_date | ...
    # Read with pipe separator, only first 2 columns
    movies = pd.read_csv(
        u_item_path,
        sep='|',
        header=None,
        encoding='latin-1',
        usecols=[0, 1],
        names=['video_id', 'title']
    )
    
    # Create dictionary mapping
    title_map = dict(zip(movies['video_id'], movies['title']))
    
    return title_map


def load_model(model_path: Path = Path("models/svd_model.pkl")) -> SVD:
    """
    Load trained SVD model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Trained SVD model
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run train.py first."
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def get_popular_items(
    ratings_path: Path = Path("data/ratings.csv"),
    top_k: int = 10
) -> list[int]:
    """
    Get top K most popular items (by number of ratings).
    Used as fallback for cold start users.
    
    Args:
        ratings_path: Path to ratings CSV
        top_k: Number of popular items to return
        
    Returns:
        List of video_ids (most popular first)
    """
    ratings = pd.read_csv(ratings_path)
    
    # Count ratings per video and sort
    video_counts = ratings['video_id'].value_counts()
    
    # Get top K video IDs
    popular_videos = video_counts.head(top_k).index.tolist()
    
    return popular_videos


def get_recommendations(
    user_id: int,
    top_k: int = 10,
    model_path: Path = Path("models/svd_model.pkl"),
    ratings_path: Path = Path("data/ratings.csv"),
    dataset_dir: Path = Path("data/ml-100k")
) -> list[str]:
    """
    Get top K video recommendations for a user.
    Returns movie titles (not IDs).
    Includes fallback for cold start users.
    
    Args:
        user_id: User ID to get recommendations for
        top_k: Number of recommendations to return
        model_path: Path to trained model
        ratings_path: Path to ratings CSV
        dataset_dir: Path to ml-100k directory
        
    Returns:
        List of movie titles (video titles)
    """
    # Load model and data
    model = load_model(model_path)
    ratings = pd.read_csv(ratings_path)
    title_map = load_movie_titles(dataset_dir)
    
    # Check if user exists in training data
    user_ratings = ratings[ratings['user_id'] == user_id]
    
    if len(user_ratings) == 0:
        # Cold start: return popular items
        print(f"User {user_id} not found in training data. Returning popular items.")
        popular_video_ids = get_popular_items(ratings_path, top_k)
        recommendations = [title_map.get(vid_id, f"Video {vid_id}") for vid_id in popular_video_ids]
        return recommendations
    
    # Get all unique video IDs
    all_videos = ratings['video_id'].unique()
    
    # Get videos already rated by user
    user_rated = set(user_ratings['video_id'].values)
    
    # Predict ratings for unrated videos
    predictions = []
    for video_id in all_videos:
        if video_id not in user_rated:
            try:
                pred_rating = model.predict(user_id, video_id).est
                predictions.append({
                    'video_id': video_id,
                    'predicted_rating': pred_rating
                })
            except Exception as e:
                # Skip if prediction fails
                continue
    
    # Convert to DataFrame and sort
    if not predictions:
        # Fallback to popular items if no predictions
        print(f"No predictions available. Returning popular items.")
        popular_video_ids = get_popular_items(ratings_path, top_k)
        recommendations = [title_map.get(vid_id, f"Video {vid_id}") for vid_id in popular_video_ids]
        return recommendations
    
    recommendations_df = pd.DataFrame(predictions)
    recommendations_df = recommendations_df.sort_values(
        'predicted_rating',
        ascending=False
    ).head(top_k)
    
    # Map video IDs to titles
    recommended_titles = [
        title_map.get(vid_id, f"Video {vid_id}")
        for vid_id in recommendations_df['video_id'].values
    ]
    
    return recommended_titles


def main():
    """Example usage of recommendation function."""
    # Example: Get recommendations for user 1
    user_id = 1
    top_k = 10
    
    print(f"Getting top {top_k} recommendations for user {user_id}...")
    recommendations = get_recommendations(user_id, top_k)
    
    print(f"\nTop {top_k} recommendations:")
    for i, title in enumerate(recommendations, 1):
        print(f"{i}. {title}")
    
    # Example: Cold start user (user ID that doesn't exist)
    print(f"\n\nTesting cold start user (user_id=99999)...")
    cold_start_recs = get_recommendations(99999, top_k)
    print(f"\nTop {top_k} recommendations (popular items):")
    for i, title in enumerate(cold_start_recs, 1):
        print(f"{i}. {title}")


if __name__ == "__main__":
    main()

