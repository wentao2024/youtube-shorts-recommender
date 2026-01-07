"""
从 MovieLens u.item 文件创建 videos.csv
用于提供视频元数据（标题等）给 BM25 和 Two-Tower 模型
"""
import pandas as pd
from pathlib import Path


def create_videos_csv():
    """从 u.item 文件创建 videos.csv"""
    data_dir = Path("data")
    ml100k_dir = data_dir / "ml-100k"
    u_item_file = ml100k_dir / "u.item"
    output_file = data_dir / "videos.csv"
    
    if not u_item_file.exists():
        print(f"Error: {u_item_file} not found!")
        print("Please run data_prep.py first to download the dataset.")
        return False
    
    print(f"Reading movie information from {u_item_file}...")
    
    # u.item 格式：
    # movie_id | movie_title | release_date | video_release_date | IMDb_URL | 
    # genre1 | genre2 | ... | genre19
    # 使用 | 分隔，编码可能是 latin-1
    
    try:
        # 尝试读取 u.item 文件
        movies = pd.read_csv(
            u_item_file,
            sep='|',
            header=None,
            encoding='latin-1',
            engine='python',
            names=[
                'movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url',
                'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
                'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
                'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'
            ]
        )
        
        # 创建 videos.csv，只保留必要信息
        videos = pd.DataFrame({
            'video_id': movies['movie_id'],
            'title': movies['movie_title'],
            'description': '',  # MovieLens 没有描述，留空
            'release_date': movies['release_date']
        })
        
        # 保存
        videos.to_csv(output_file, index=False)
        print(f"✅ Created {output_file}")
        print(f"   Total videos: {len(videos)}")
        print(f"   Sample titles:")
        for i, row in videos.head(5).iterrows():
            print(f"     {row['video_id']}: {row['title']}")
        
        return True
        
    except Exception as e:
        print(f"Error reading u.item: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Creating videos.csv from MovieLens u.item")
    print("=" * 60)
    print()
    
    success = create_videos_csv()
    
    if success:
        print()
        print("=" * 60)
        print("✅ videos.csv created successfully!")
        print("=" * 60)
        print()
        print("Now BM25 and Two-Tower models can use video titles")
        print("for better text-based recommendations.")
    else:
        print()
        print("=" * 60)
        print("❌ Failed to create videos.csv")
        print("=" * 60)
        print()
        print("BM25 will still work, but with minimal metadata.")



