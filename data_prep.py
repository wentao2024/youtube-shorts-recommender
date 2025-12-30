"""
Download and preprocess MovieLens 100K dataset for YouTube Shorts recommendation simulation.
"""
import os
import urllib.request
import ssl
import zipfile
import pandas as pd
from pathlib import Path


def download_movielens_100k():
    """Download MovieLens 100K dataset if not already present."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    zip_path = data_dir / "ml-100k.zip"
    extract_dir = data_dir / "ml-100k"
    
    # Check if already extracted
    if extract_dir.exists() and (extract_dir / "u.data").exists():
        print("MovieLens 100K dataset already exists.")
        return extract_dir
    
    # Download if zip doesn't exist
    if not zip_path.exists():
        print("Downloading MovieLens 100K dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        # Create SSL context that doesn't verify certificates (for download only)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
        print("Download complete.")
    
    # Extract
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")
    
    return extract_dir


def preprocess_ratings(dataset_dir: Path) -> pd.DataFrame:
    """
    Preprocess MovieLens 100K ratings data.
    
    Args:
        dataset_dir: Path to the ml-100k directory
        
    Returns:
        DataFrame with columns: user_id, video_id (renamed from movie_id), rating
    """
    # Read u.data file (user_id, movie_id, rating, timestamp)
    ratings_file = dataset_dir / "u.data"
    print(f"Reading ratings from {ratings_file}...")
    
    # Column names: user_id, movie_id, rating, timestamp
    ratings = pd.read_csv(
        ratings_file,
        sep='\t',
        header=None,
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    # Rename movie_id to video_id for YouTube Shorts simulation
    ratings = ratings.rename(columns={'movie_id': 'video_id'})
    
    # Keep only necessary columns
    ratings = ratings[['user_id', 'video_id', 'rating']]
    
    print(f"Processed {len(ratings)} ratings from {ratings['user_id'].nunique()} users and {ratings['video_id'].nunique()} videos.")
    
    return ratings


def save_processed_data(ratings: pd.DataFrame, output_path: Path = Path("data/ratings.csv")):
    """Save processed ratings to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(output_path, index=False)
    print(f"Saved processed ratings to {output_path}")


def main():
    """Main function to download and preprocess data."""
    print("Starting data preparation...")
    
    # Download dataset
    dataset_dir = download_movielens_100k()
    
    # Preprocess ratings
    ratings = preprocess_ratings(dataset_dir)
    
    # Save processed data
    save_processed_data(ratings)
    
    print("Data preparation complete!")


if __name__ == "__main__":
    main()

