"""
推荐系统诊断工具
快速定位性能问题
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# 可选的可视化库
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class RecommenderDiagnostics:
    """推荐系统诊断器"""
    
    def __init__(self, ratings_path: Path):
        self.ratings = pd.read_csv(ratings_path)
        self.issues = []
    
    def run_full_diagnostics(self):
        """运行完整诊断"""
        print("=" * 70)
        print("RECOMMENDER SYSTEM DIAGNOSTICS")
        print("=" * 70)
        
        print("\n1. Data Quality Check")
        self.check_data_quality()
        
        print("\n2. Sparsity Analysis")
        self.analyze_sparsity()
        
        print("\n3. Cold Start Analysis")
        self.analyze_cold_start()
        
        print("\n4. Popularity Bias Check")
        self.check_popularity_bias()
        
        print("\n5. Dataset Suitability for Deep Learning")
        self.check_dl_suitability()
        
        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)
        
        if len(self.issues) == 0:
            print("✓ No major issues detected!")
        else:
            print(f"⚠ Found {len(self.issues)} issues:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        print("\n" + "=" * 70)
        return self.issues
    
    def check_data_quality(self):
        """检查数据质量"""
        print("-" * 70)
        n_users = self.ratings['user_id'].nunique()
        n_items = self.ratings['video_id'].nunique()
        n_ratings = len(self.ratings)
        
        print(f"Users: {n_users:,}")
        print(f"Items: {n_items:,}")
        print(f"Ratings: {n_ratings:,}")
        print(f"Sparsity: {1 - n_ratings / (n_users * n_items):.4f}")
        
        # 检查是否有足够数据训练深度学习
        if n_ratings < 50000:
            issue = f"⚠ Low data volume ({n_ratings:,} ratings). Deep learning may underperform."
            print(f"  {issue}")
            self.issues.append(issue)
        else:
            print(f"  ✓ Sufficient data volume for deep learning")
        
        # 检查评分分布
        rating_dist = self.ratings['rating'].value_counts().sort_index()
        print(f"\nRating distribution:")
        for rating, count in rating_dist.items():
            pct = count / len(self.ratings) * 100
            print(f"  {rating}: {count:,} ({pct:.1f}%)")
        
        # 检查是否有偏态
        if rating_dist.max() / rating_dist.sum() > 0.5:
            issue = "⚠ Highly skewed rating distribution"
            print(f"  {issue}")
            self.issues.append(issue)
    
    def analyze_sparsity(self):
        """分析稀疏性"""
        print("-" * 70)
        user_counts = self.ratings['user_id'].value_counts()
        item_counts = self.ratings['video_id'].value_counts()
        
        print(f"User activity:")
        print(f"  Mean: {user_counts.mean():.1f} ratings/user")
        print(f"  Median: {user_counts.median():.1f} ratings/user")
        print(f"  Min: {user_counts.min()}")
        print(f"  Max: {user_counts.max()}")
        
        print(f"\nItem popularity:")
        print(f"  Mean: {item_counts.mean():.1f} ratings/item")
        print(f"  Median: {item_counts.median():.1f} ratings/item")
        print(f"  Min: {item_counts.min()}")
        print(f"  Max: {item_counts.max()}")
        
        # 检查是否有很多低活跃用户
        low_activity_users = (user_counts < 5).sum()
        if low_activity_users / len(user_counts) > 0.3:
            issue = f"⚠ High cold-start risk: {low_activity_users/len(user_counts)*100:.1f}% users have <5 ratings"
            print(f"  {issue}")
            self.issues.append(issue)
    
    def analyze_cold_start(self):
        """分析冷启动问题"""
        print("-" * 70)
        user_counts = self.ratings['user_id'].value_counts()
        item_counts = self.ratings['video_id'].value_counts()
        
        # 用户冷启动
        cold_users = (user_counts < 10).sum()
        cold_user_pct = cold_users / len(user_counts) * 100
        print(f"Cold-start users (<10 ratings): {cold_users} ({cold_user_pct:.1f}%)")
        
        # 物品冷启动
        cold_items = (item_counts < 5).sum()
        cold_item_pct = cold_items / len(item_counts) * 100
        print(f"Cold-start items (<5 ratings): {cold_items} ({cold_item_pct:.1f}%)")
        
        if cold_user_pct > 30:
            issue = f"⚠ High user cold-start: {cold_user_pct:.1f}% users"
            print(f"  {issue}")
            self.issues.append(issue)
            print("  Recommendation: Use content-based features (BM25, metadata)")
        
        if cold_item_pct > 30:
            issue = f"⚠ High item cold-start: {cold_item_pct:.1f}% items"
            print(f"  {issue}")
            self.issues.append(issue)
            print("  Recommendation: Use item metadata and BM25")
    
    def check_popularity_bias(self):
        """检查热门偏差"""
        print("-" * 70)
        item_counts = self.ratings['video_id'].value_counts()
        
        # 计算基尼系数
        gini = self._compute_gini(item_counts.values)
        print(f"Item popularity Gini coefficient: {gini:.3f}")
        
        if gini > 0.8:
            issue = f"⚠ Severe popularity bias (Gini={gini:.3f})"
            print(f"  {issue}")
            self.issues.append(issue)
            print("  Recommendation: Use diversity-aware ranking")
        
        # Top 20% items占多少交互
        top_20_pct_items = int(len(item_counts) * 0.2)
        top_20_interactions = item_counts.head(top_20_pct_items).sum()
        top_20_ratio = top_20_interactions / item_counts.sum()
        
        print(f"Top 20% items account for {top_20_ratio*100:.1f}% of interactions")
        
        if top_20_ratio > 0.8:
            issue = f"⚠ Heavy-tailed distribution: top 20% items have {top_20_ratio*100:.1f}% interactions"
            print(f"  {issue}")
            self.issues.append(issue)
    
    def check_dl_suitability(self):
        """检查是否适合深度学习"""
        print("-" * 70)
        n_users = self.ratings['user_id'].nunique()
        n_items = self.ratings['video_id'].nunique()
        n_ratings = len(self.ratings)
        
        # 检查1: 数据量
        print("Deep Learning Suitability Check:")
        if n_ratings < 100000:
            print(f"  ✗ Data volume: {n_ratings:,} (recommend >100K)")
            self.issues.append("Deep learning may underperform due to limited data")
        else:
            print(f"  ✓ Data volume: {n_ratings:,}")
        
        # 检查2: 用户数和物品数
        if n_users < 1000 or n_items < 1000:
            print(f"  ✗ Scale: {n_users} users, {n_items} items (recommend >1000 each)")
            self.issues.append("Small scale may not benefit from deep learning complexity")
        else:
            print(f"  ✓ Scale: {n_users} users, {n_items} items")
        
        # 检查3: 每用户平均评分
        avg_ratings_per_user = n_ratings / n_users
        if avg_ratings_per_user < 20:
            print(f"  ✗ Avg ratings/user: {avg_ratings_per_user:.1f} (recommend >20)")
            self.issues.append("Low user activity may limit deep learning effectiveness")
        else:
            print(f"  ✓ Avg ratings/user: {avg_ratings_per_user:.1f}")
        
        # 总结建议
        print("\nRecommendation:")
        if len([i for i in self.issues if 'deep learning' in i.lower()]) > 0:
            print("  → Start with traditional CF (SVD)")
            print("  → Use deep learning as enhancement, not replacement")
            print("  → Consider ensemble approach")
        else:
            print("  ✓ Dataset suitable for deep learning")
    
    def _compute_gini(self, values):
        """计算基尼系数"""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    def visualize_diagnostics(self, save_path: Path = None):
        """可视化诊断结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 用户活跃度分布
            user_counts = self.ratings['user_id'].value_counts()
            axes[0, 0].hist(user_counts, bins=50, edgecolor='black')
            axes[0, 0].set_title('User Activity Distribution')
            axes[0, 0].set_xlabel('Number of Ratings')
            axes[0, 0].set_ylabel('Number of Users')
            axes[0, 0].axvline(user_counts.median(), color='r', linestyle='--', 
                              label=f'Median: {user_counts.median():.0f}')
            axes[0, 0].legend()
            
            # 2. 物品流行度分布
            item_counts = self.ratings['video_id'].value_counts()
            axes[0, 1].hist(item_counts, bins=50, edgecolor='black')
            axes[0, 1].set_title('Item Popularity Distribution')
            axes[0, 1].set_xlabel('Number of Ratings')
            axes[0, 1].set_ylabel('Number of Items')
            axes[0, 1].axvline(item_counts.median(), color='r', linestyle='--',
                              label=f'Median: {item_counts.median():.0f}')
            axes[0, 1].legend()
            
            # 3. 评分分布
            rating_dist = self.ratings['rating'].value_counts().sort_index()
            axes[1, 0].bar(rating_dist.index, rating_dist.values, edgecolor='black')
            axes[1, 0].set_title('Rating Distribution')
            axes[1, 0].set_xlabel('Rating')
            axes[1, 0].set_ylabel('Count')
            
            # 4. 长尾分布（物品流行度）
            sorted_counts = item_counts.sort_values(ascending=False).values
            axes[1, 1].plot(range(len(sorted_counts)), sorted_counts)
            axes[1, 1].set_title('Long-tail Distribution (Item Popularity)')
            axes[1, 1].set_xlabel('Item Rank')
            axes[1, 1].set_ylabel('Number of Ratings')
            axes[1, 1].set_yscale('log')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            plt.show()
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")
            print("  (This is optional - text diagnostics are still available)")


def quick_fix_suggestions(issues: List[str]):
    """根据诊断结果给出快速修复建议"""
    print("\n" + "=" * 70)
    print("QUICK FIX SUGGESTIONS")
    print("=" * 70)
    
    if not issues:
        print("No issues detected - system is healthy!")
        return
    
    suggestions = []
    
    for issue in issues:
        if 'data volume' in issue.lower() or 'limited data' in issue.lower():
            suggestions.append(
                "1. Use data augmentation:\n"
                "   - Implicit feedback (views, clicks)\n"
                "   - User demographics if available\n"
                "   - Item metadata"
            )
        
        if 'cold-start' in issue.lower() or 'cold start' in issue.lower():
            suggestions.append(
                "2. Enhance cold-start handling:\n"
                "   - Increase BM25 weight for new users\n"
                "   - Use popularity-based fallback\n"
                "   - Add content-based features"
            )
        
        if 'popularity bias' in issue.lower():
            suggestions.append(
                "3. Reduce popularity bias:\n"
                "   - Use diversity-aware reranking\n"
                "   - Sample negatives from less popular items\n"
                "   - Add exploration bonus to long-tail items"
            )
        
        if 'deep learning' in issue.lower():
            suggestions.append(
                "4. Optimize deep learning approach:\n"
                "   - Start with ensemble (70% CF + 30% DL)\n"
                "   - Pre-train on larger dataset if available\n"
                "   - Use simpler model architecture\n"
                "   - Focus on feature engineering"
            )
    
    # 去重并打印
    unique_suggestions = list(dict.fromkeys(suggestions))
    for sugg in unique_suggestions:
        print(f"\n{sugg}")


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recommender System Diagnostics")
    parser.add_argument(
        "--ratings",
        type=str,
        default="data/ratings.csv",
        help="Path to ratings CSV file (default: data/ratings.csv)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization"
    )
    
    args = parser.parse_args()
    
    # 运行诊断
    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        print(f"Error: {ratings_path} not found!")
        print("\nAvailable options:")
        print("  - MovieLens-100K: data/ratings.csv")
        print("  - KuaiRec 2.0: data/ratings_kuairec.csv")
        print("\nPlease run data_prep.py or data_prep_kuairec.py first")
        exit(1)
    
    print(f"Using ratings file: {ratings_path}")
    diagnostics = RecommenderDiagnostics(ratings_path)
    issues = diagnostics.run_full_diagnostics()
    
    # 可视化（如果matplotlib可用且未禁用）
    if not args.no_viz:
        try:
            output_name = f"diagnostics_report_{ratings_path.stem}.png"
            diagnostics.visualize_diagnostics(save_path=Path(output_name))
        except Exception as e:
            print(f"\nNote: Visualization skipped ({e})")
    
    # 获取修复建议
    quick_fix_suggestions(issues)

