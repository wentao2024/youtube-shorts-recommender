"""
KuaiRec数据集预处理
KuaiRec: A Fully-observed Dataset for Recommender Systems
https://kuairec.com/
GitHub: https://github.com/chongminggao/KuaiRec
"""
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm
import os
import subprocess
import shutil

def download_kuairec_from_github(data_dir: Path, force: bool = False):
    """
    从GitHub下载KuaiRec数据集
    
    Args:
        data_dir: 数据目录
        force: 是否强制重新下载
    
    Returns:
        KuaiRec数据目录路径
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    kuairec_dir = data_dir / "kuairec"
    kuairec_git_dir = data_dir / "KuaiRec"
    
    # 检查是否已存在
    if kuairec_dir.exists() and not force:
        print(f"✓ KuaiRec directory already exists: {kuairec_dir}")
        return kuairec_dir
    
    print("=" * 70)
    print("Downloading KuaiRec from GitHub")
    print("=" * 70)
    print("\nRepository: https://github.com/chongminggao/KuaiRec")
    
    # 方法1: 使用git clone
    if shutil.which("git"):
        print("\n📥 Method 1: Using git clone...")
        try:
            if kuairec_git_dir.exists() and force:
                shutil.rmtree(kuairec_git_dir)
            
            if not kuairec_git_dir.exists():
                subprocess.run(
                    ["git", "clone", "https://github.com/chongminggao/KuaiRec.git"],
                    cwd=data_dir,
                    check=True
                )
                print("✓ Git clone successful")
            else:
                print("✓ Repository already cloned")
            
            # 检查是否有数据文件
            data_files = list(kuairec_git_dir.glob("*.csv"))
            if data_files:
                print(f"✓ Found {len(data_files)} CSV files in repository")
                # 如果GitHub仓库包含数据文件，直接使用
                if (kuairec_git_dir / "big_matrix.csv").exists():
                    print("✓ Found big_matrix.csv in GitHub repository")
                    return kuairec_git_dir
            else:
                print("⚠ No CSV data files found in GitHub repository")
                print("  GitHub may only contain code, not the actual dataset")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Git clone failed: {e}")
        except Exception as e:
            print(f"⚠ Error: {e}")
    
    # 方法2: 提示手动下载
    print("\n" + "=" * 70)
    print("Alternative Download Methods")
    print("=" * 70)
    print("\n📥 Option 1: Download from official website")
    print("   Visit: https://kuairec.com/")
    print("   Register and download the full dataset")
    print(f"   Extract to: {kuairec_dir}")
    
    print("\n📥 Option 2: Check GitHub for data files")
    print("   Visit: https://github.com/chongminggao/KuaiRec")
    print("   Look for data files or download links")
    print("   Note: Large datasets may be stored elsewhere (e.g., Google Drive)")
    
    print("\n📁 Expected file structure:")
    print("   kuairec/")
    print("     ├── big_matrix.csv          # 全观测交互矩阵 (required)")
    print("     ├── small_matrix.csv         # 稀疏交互矩阵 (optional)")
    print("     ├── user_features.csv       # 用户特征 (optional)")
    print("     └── item_features.csv       # 视频特征 (optional)")
    
    return kuairec_dir


def load_kuairec_interactions(kuairec_dir: Path, use_big_matrix: bool = True):
    """
    加载KuaiRec交互数据
    
    Args:
        kuairec_dir: KuaiRec数据目录（可能包含data子目录，如KuaiRec 2.0）
        use_big_matrix: 是否使用big_matrix（全观测数据，99.6%稠密度）
    
    Returns:
        DataFrame with columns: user_id, video_id, rating, timestamp
    """
    # 检查是否有data子目录（KuaiRec 2.0格式）
    data_subdir = kuairec_dir / "data"
    if data_subdir.exists():
        kuairec_dir = data_subdir
        print(f"Found data subdirectory, using: {kuairec_dir}")
    
    if use_big_matrix:
        matrix_file = kuairec_dir / "big_matrix.csv"
        print("Loading big_matrix.csv (fully observed, 99.6% density)...")
    else:
        matrix_file = kuairec_dir / "small_matrix.csv"
        print("Loading small_matrix.csv (sparse observations)...")
    
    if not matrix_file.exists():
        print(f"\n❌ Error: Required file not found: {matrix_file}")
        print("\n" + "=" * 70)
        print("FILE NOT FOUND")
        print("=" * 70)
        print(f"\nExpected file: {matrix_file}")
        print("\nPlease ensure:")
        print("1. Dataset is downloaded from https://kuairec.com/")
        print(f"2. Files are extracted to: {kuairec_dir}")
        print("3. File names match exactly (case-sensitive)")
        print("\nAvailable files in directory:")
        if kuairec_dir.exists():
            files = list(kuairec_dir.glob("*.csv"))
            if files:
                for f in files:
                    print(f"  - {f.name}")
            else:
                print("  (no CSV files found)")
        print("\n" + "=" * 70)
        raise FileNotFoundError(f"Required file not found: {matrix_file}")
    
    # 读取CSV文件
    print(f"Reading {matrix_file}...")
    
    # 检查文件格式：是矩阵格式还是长格式
    # 先读取前几行判断
    sample = pd.read_csv(matrix_file, nrows=5)
    
    # KuaiRec 2.0使用长格式：user_id, video_id, play_duration, video_duration, time, date, timestamp, watch_ratio
    if 'user_id' in sample.columns and 'video_id' in sample.columns:
        print("Detected long format (KuaiRec 2.0 style)...")
        # 长格式，分块读取大文件
        print("Reading full dataset (this may take a while for large files)...")
        
        # 估算文件大小
        file_size_mb = matrix_file.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        
        # 如果文件很大，分块读取
        if file_size_mb > 500:  # 大于500MB
            print("Large file detected, reading in chunks...")
            chunks = []
            chunk_size = 100000  # 每次读取10万行
            for chunk in tqdm(pd.read_csv(matrix_file, chunksize=chunk_size), desc="Reading chunks"):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(matrix_file)
        
        # 转换watch_ratio或play_duration为rating
        # 使用watch_ratio作为评分（0-1范围，可以映射到1-5）
        if 'watch_ratio' in df.columns:
            # watch_ratio: 观看时长/视频时长，映射到1-5评分
            # 0-0.2 -> 1, 0.2-0.4 -> 2, 0.4-0.6 -> 3, 0.6-0.8 -> 4, 0.8+ -> 5
            df['rating'] = pd.cut(
                df['watch_ratio'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, float('inf')],
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
        elif 'play_duration' in df.columns and 'video_duration' in df.columns:
            # 计算watch_ratio
            df['watch_ratio'] = df['play_duration'] / df['video_duration']
            df['rating'] = pd.cut(
                df['watch_ratio'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, float('inf')],
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
        else:
            # 如果没有观看比例，使用默认评分
            df['rating'] = 3.0
        
        # 提取timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].fillna(0).astype(int)
        elif 'time' in df.columns:
            # 如果有time列，尝试转换
            df['timestamp'] = 0
        else:
            df['timestamp'] = 0
        
        # 只保留需要的列
        df = df[['user_id', 'video_id', 'rating', 'timestamp']].copy()
        
    else:
        # 矩阵格式（旧版本KuaiRec）
        print("Detected matrix format (old KuaiRec style)...")
        matrix = pd.read_csv(matrix_file, index_col=0)
        
        print(f"Matrix shape: {matrix.shape}")
        print(f"Users: {matrix.shape[0]}, Videos: {matrix.shape[1]}")
        
        # 转换为长格式
        print("Converting to long format...")
        interactions = []
        
        for user_idx, user_id in enumerate(tqdm(matrix.index, desc="Processing users")):
            for video_idx, video_id in enumerate(matrix.columns):
                rating = matrix.iloc[user_idx, video_idx]
                
                # 跳过缺失值
                if pd.isna(rating) or rating == 0:
                    continue
                
                interactions.append({
                    'user_id': int(user_id),
                    'video_id': int(video_id),
                    'rating': float(rating),
                    'timestamp': 0
                })
        
        df = pd.DataFrame(interactions)
    
    print(f"\nTotal interactions: {len(df):,}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique videos: {df['video_id'].nunique()}")
    print(f"Density: {len(df) / (df['user_id'].nunique() * df['video_id'].nunique()):.4f}")
    
    return df


def load_kuairec_features(kuairec_dir: Path):
    """
    加载KuaiRec特征数据
    
    Args:
        kuairec_dir: KuaiRec数据目录（可能包含data子目录）
    
    Returns:
        user_features: DataFrame with user features
        video_features: DataFrame with video features
    """
    # 检查是否有data子目录（KuaiRec 2.0格式）
    data_subdir = kuairec_dir / "data"
    if data_subdir.exists():
        kuairec_dir = data_subdir
    
    user_features = None
    video_features = None
    
    # 尝试多个可能的文件名
    user_files = [
        kuairec_dir / "user_features.csv",
        kuairec_dir / "user_features_raw.csv"
    ]
    
    video_files = [
        kuairec_dir / "item_features.csv",
        kuairec_dir / "kuairec_caption_category.csv",
        kuairec_dir / "item_categories.csv"
    ]
    
    # 加载用户特征
    for user_file in user_files:
        if user_file.exists():
            print(f"\nLoading user features from {user_file}...")
            try:
                try:
                    user_features = pd.read_csv(user_file, encoding='utf-8')
                except (UnicodeDecodeError, pd.errors.ParserError):
                    try:
                        user_features = pd.read_csv(user_file, encoding='gbk')
                    except Exception:
                        try:
                            user_features = pd.read_csv(
                                user_file,
                                encoding='utf-8',
                                engine='python',
                                on_bad_lines='skip'
                            )
                        except TypeError:
                            # pandas < 1.3.0
                            user_features = pd.read_csv(
                                user_file,
                                encoding='utf-8',
                                engine='python',
                                error_bad_lines=False,
                                warn_bad_lines=False
                            )
                print(f"User features shape: {user_features.shape}")
                break
            except Exception as e:
                print(f"Error loading {user_file}: {e}")
                print("  Skipping this file, trying next...")
                continue
    
    if user_features is None:
        print(f"\nUser features file not found in {kuairec_dir}")
    
    # 加载视频特征
    for video_file in video_files:
        if video_file.exists():
            print(f"Loading video features from {video_file}...")
            try:
                # 尝试多种读取方式处理格式问题
                try:
                    # 方式1: 标准读取
                    video_features = pd.read_csv(video_file, encoding='utf-8')
                except (UnicodeDecodeError, pd.errors.ParserError):
                    try:
                        # 方式2: 尝试其他编码
                        video_features = pd.read_csv(video_file, encoding='gbk')
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        try:
                            # 方式3: 使用Python引擎（更容错）
                            video_features = pd.read_csv(
                                video_file, 
                                encoding='utf-8',
                                engine='python',
                                sep=',',
                                quotechar='"',
                                on_bad_lines='skip'  # pandas >= 1.3.0
                            )
                        except TypeError:
                            # pandas < 1.3.0 使用旧参数
                            try:
                                video_features = pd.read_csv(
                                    video_file,
                                    encoding='utf-8',
                                    engine='python',
                                    sep=',',
                                    quotechar='"',
                                    error_bad_lines=False,
                                    warn_bad_lines=False
                                )
                            except Exception as e:
                                print(f"Warning: Failed to read {video_file}: {e}")
                                print("  Trying with minimal options...")
                                # 方式4: 最宽松的方式
                                video_features = pd.read_csv(
                                    video_file,
                                    encoding='utf-8',
                                    engine='python',
                                    sep=',',
                                    quotechar='"',
                                    error_bad_lines=False,
                                    warn_bad_lines=False,
                                    skipinitialspace=True
                                )
                        except Exception as e:
                            print(f"Warning: Failed to read {video_file}: {e}")
                            print("  Trying with minimal options...")
                            # 方式4: 最宽松的方式
                            try:
                                video_features = pd.read_csv(
                                    video_file,
                                    encoding='utf-8',
                                    engine='python',
                                    sep=',',
                                    quotechar='"',
                                    error_bad_lines=False,
                                    warn_bad_lines=False,
                                    skipinitialspace=True
                                )
                            except Exception:
                                # 最后尝试：只读取前几列
                                print("  Trying to read with limited columns...")
                                video_features = pd.read_csv(
                                    video_file,
                                    encoding='utf-8',
                                    engine='python',
                                    usecols=[0, 1, 2, 3],  # 只读前4列
                                    error_bad_lines=False,
                                    warn_bad_lines=False
                                )
                
                # 确保有video_id列
                if 'item_id' in video_features.columns:
                    video_features = video_features.rename(columns={'item_id': 'video_id'})
                elif 'video_id' not in video_features.columns:
                    print(f"Warning: {video_file} does not have video_id column")
                    print(f"  Available columns: {list(video_features.columns)}")
                    # 尝试使用第一列作为video_id
                    if len(video_features.columns) > 0:
                        first_col = video_features.columns[0]
                        if 'id' in first_col.lower() or 'video' in first_col.lower():
                            video_features = video_features.rename(columns={first_col: 'video_id'})
                            print(f"  Using {first_col} as video_id")
                
                print(f"Video features shape: {video_features.shape}")
                break
            except Exception as e:
                print(f"Error loading {video_file}: {e}")
                print("  Skipping this file, trying next...")
                continue
    
    if video_features is None:
        print(f"Video features file not found or could not be loaded from {kuairec_dir}")
        print("  Note: Videos.csv will not be created, but ratings.csv will still be generated")
        print("  BM25 and Cross-Encoder may not work without video features")
    
    return user_features, video_features


def create_videos_csv_from_features(video_features: pd.DataFrame, output_path: Path):
    """
    从特征数据创建videos.csv（用于BM25和Cross-Encoder）
    
    Args:
        video_features: 视频特征DataFrame
        output_path: 输出路径
    """
    if video_features is None:
        print("No video features available, skipping videos.csv creation")
        return
    
    print(f"\nCreating videos.csv from features...")
    
    # 创建videos.csv格式
    videos_df = pd.DataFrame()
    videos_df['video_id'] = video_features['video_id']
    
    # KuaiRec 2.0格式：kuairec_caption_category.csv
    # 包含：video_id, manual_cover_text, caption, topic_tag, category信息
    
    # 标题：使用manual_cover_text或caption
    if 'manual_cover_text' in video_features.columns:
        videos_df['title'] = video_features['manual_cover_text'].fillna('')
    elif 'caption' in video_features.columns:
        videos_df['title'] = video_features['caption'].fillna('')
    elif 'title' in video_features.columns:
        videos_df['title'] = video_features['title'].fillna('')
    elif 'name' in video_features.columns:
        videos_df['title'] = video_features['name'].fillna('')
    else:
        videos_df['title'] = 'Video ' + videos_df['video_id'].astype(str)
    
    # 描述：组合caption, topic_tag, category信息
    description_parts = []
    if 'caption' in video_features.columns:
        description_parts.append(video_features['caption'].fillna(''))
    if 'topic_tag' in video_features.columns:
        description_parts.append(video_features['topic_tag'].fillna(''))
    if 'first_level_category_name' in video_features.columns:
        description_parts.append(video_features['first_level_category_name'].fillna(''))
    if 'second_level_category_name' in video_features.columns:
        description_parts.append(video_features['second_level_category_name'].fillna(''))
    
    if description_parts:
        videos_df['description'] = description_parts[0]
        for part in description_parts[1:]:
            videos_df['description'] = videos_df['description'] + ' ' + part
        videos_df['description'] = videos_df['description'].str.strip()
    elif 'description' in video_features.columns:
        videos_df['description'] = video_features['description'].fillna('')
    elif 'desc' in video_features.columns:
        videos_df['description'] = video_features['desc'].fillna('')
    else:
        videos_df['description'] = ''
    
    # 添加类别信息（如果有）
    for col in ['first_level_category_name', 'second_level_category_name', 
                'third_level_category_name', 'category', 'category_name']:
        if col in video_features.columns:
            videos_df[col] = video_features[col]
    
    videos_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(videos_df)} videos")
    print(f"  - Videos with title: {(videos_df['title'] != '').sum()}")
    print(f"  - Videos with description: {(videos_df['description'] != '').sum()}")


def prepare_kuairec(
    kuairec_dir: Path = None,
    output_dir: Path = Path("data"),
    use_big_matrix: bool = True,
    download_from_github: bool = False
):
    """
    预处理KuaiRec数据集
    
    Args:
        kuairec_dir: KuaiRec数据目录（如果为None，会提示下载）
        output_dir: 输出目录
        use_big_matrix: 是否使用big_matrix（全观测数据）
        download_from_github: 是否尝试从GitHub下载
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果没有指定目录，尝试下载
    if kuairec_dir is None:
        if download_from_github:
            kuairec_dir = download_kuairec_from_github(output_dir)
        else:
            print("=" * 70)
            print("KuaiRec Dataset Download Instructions")
            print("=" * 70)
            print("\n📥 Download Options:")
            print("   1. From GitHub: python3 data_prep_kuairec.py --download_github")
            print("   2. From official website: https://kuairec.com/")
            print("\n💡 Tip: GitHub may only contain code, not the full dataset")
            print("   For the complete dataset, visit: https://kuairec.com/")
            print("=" * 70)
            kuairec_dir = output_dir / "kuairec"
    
    kuairec_dir = Path(kuairec_dir)
    
    if not kuairec_dir.exists():
        print(f"\n❌ Error: KuaiRec directory not found: {kuairec_dir}")
        print("\n" + "=" * 70)
        print("DOWNLOAD INSTRUCTIONS")
        print("=" * 70)
        print("\n1. Visit KuaiRec website:")
        print("   https://kuairec.com/")
        print("\n2. Register and download the dataset")
        print("\n3. Extract the dataset to:")
        print(f"   {kuairec_dir}")
        print("\n4. Required files:")
        print("   - big_matrix.csv (required)")
        print("   - small_matrix.csv (optional)")
        print("   - item_features.csv (optional, for BM25/Cross-Encoder)")
        print("   - user_features.csv (optional)")
        print("\n5. After extraction, run this script again:")
        print(f"   python3 data_prep_kuairec.py --kuairec_dir {kuairec_dir}")
        print("\n" + "=" * 70)
        return
    
    # 加载交互数据
    ratings_df = load_kuairec_interactions(kuairec_dir, use_big_matrix=use_big_matrix)
    
    # 保存ratings.csv
    ratings_path = output_dir / "ratings_kuairec.csv"
    ratings_df.to_csv(ratings_path, index=False)
    print(f"\n✓ Saved ratings to {ratings_path}")
    
    # 加载特征数据
    user_features, video_features = load_kuairec_features(kuairec_dir)
    
    # 创建videos.csv（如果特征可用）
    if video_features is not None and len(video_features) > 0:
        try:
            videos_path = output_dir / "videos_kuairec.csv"
            create_videos_csv_from_features(video_features, videos_path)
            print(f"✓ Saved video features to {videos_path}")
        except Exception as e:
            print(f"⚠ Warning: Failed to create videos.csv: {e}")
            print("  Ratings data is still available, but BM25/Cross-Encoder may not work")
    else:
        print("⚠ Warning: No video features available")
        print("  Ratings data is saved, but videos.csv was not created")
        print("  BM25 and Cross-Encoder will not work without video features")
    
    # 保存用户特征（如果需要）
    if user_features is not None:
        user_features_path = output_dir / "user_features_kuairec.csv"
        user_features.to_csv(user_features_path, index=False)
        print(f"✓ Saved user features to {user_features_path}")
    
    # 数据统计
    print("\n" + "=" * 70)
    print("KuaiRec Dataset Statistics")
    print("=" * 70)
    print(f"Users: {ratings_df['user_id'].nunique():,}")
    print(f"Videos: {ratings_df['video_id'].nunique():,}")
    print(f"Interactions: {len(ratings_df):,}")
    print(f"Density: {len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['video_id'].nunique()):.4f}")
    print(f"Avg ratings per user: {len(ratings_df) / ratings_df['user_id'].nunique():.1f}")
    print(f"Avg ratings per video: {len(ratings_df) / ratings_df['video_id'].nunique():.1f}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Run diagnostic tool:")
    print("   python3 diagnostic_tool.py  # (modify to use ratings_kuairec.csv)")
    print("\n2. Train models:")
    print("   python3 train_advanced.py  # (modify to use kuairec data)")
    print("\n3. Evaluate:")
    print("   python3 evaluate_comparison.py  # (modify to use kuairec data)")
    
    return ratings_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare KuaiRec dataset")
    parser.add_argument(
        "--kuairec_dir",
        type=str,
        default=None,
        help="Path to KuaiRec dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--use_small_matrix",
        action="store_true",
        help="Use small_matrix.csv instead of big_matrix.csv (sparse data)"
    )
    parser.add_argument(
        "--download_github",
        action="store_true",
        help="Try to download from GitHub (may not include full dataset)"
    )
    
    args = parser.parse_args()
    
    kuairec_dir = Path(args.kuairec_dir) if args.kuairec_dir else None
    output_dir = Path(args.output_dir)
    
    prepare_kuairec(
        kuairec_dir=kuairec_dir,
        output_dir=output_dir,
        use_big_matrix=not args.use_small_matrix,
        download_from_github=args.download_github
    )

