# 🎬 YouTube Shorts Recommender

A production-ready recommendation system for short videos using collaborative filtering, multi-path recall, and machine learning ranking models.

## 📸 Screenshots

### Recommendations
![Recommendations](https://github.com/wentao2024/youtube-shorts-recommender/blob/main/screenshots/recommendations.png?raw=true)

### Recall Analysis
![Recall Analysis](https://github.com/wentao2024/youtube-shorts-recommender/blob/main/screenshots/recall-analysis.png?raw=true)

### About Page
![About Page](https://github.com/wentao2024/youtube-shorts-recommender/blob/main/screenshots/about.png?raw=true)

## ✨ Features

### Traditional System (MovieLens)
- **Multi-path Recall System**: Combines collaborative filtering, popularity, high-rating, and user similarity strategies
- **Personalized Recommendations**: SVD-based collaborative filtering algorithm
- **Cold Start Handling**: Returns popular items for new users
- **Feature Engineering**: Comprehensive feature extraction for ranking
- **LightGBM Ranking**: Advanced ranking model (optional)

### Advanced System (MicroLens)
- **Two-Tower Model**: Deep learning-based recall using user and video embeddings
- **BM25 Text Retrieval**: Content-based ranking using BM25 algorithm
- **Cross-Encoder Reranking**: Fine-grained ranking with pre-trained transformer models
- **Comprehensive Evaluation**: NDCG@K, Recall@K, Coverage, Diversity metrics
- **System Comparison**: Compare traditional vs advanced system performance

### Infrastructure
- **RESTful API**: FastAPI-based backend with interactive documentation
- **Streamlit Demo**: Simple and intuitive web interface
- **Dual Dataset Support**: Works with both MovieLens-100K and MicroLens-100K

## 🏗️ Architecture

### Traditional System Architecture
```
User Request
    ↓
Multi-path Recall System
    ├─ Collaborative Filtering (SVD)
    ├─ Popularity-based
    ├─ High Rating
    └─ User Similarity
    ↓
Feature Engineering
    ↓
Ranking Model (LightGBM)
    ↓
Top K Recommendations
```

### Advanced System Architecture
```
User Request
    ↓
Multi-path Recall System
    ├─ Traditional Methods (CF, Popularity, etc.)
    ├─ Two-Tower Model (Deep Learning)
    └─ BM25 Text Retrieval
    ↓
Coarse Ranking (BM25)
    ↓
Fine Ranking (Cross-Encoder)
    ↓
Top K Recommendations
```

### Evaluation Pipeline
```
Test Data
    ↓
Generate Recommendations (Traditional)
    ↓
Generate Recommendations (Advanced)
    ↓
Calculate Metrics
    ├─ NDCG@10
    ├─ Recall@10
    ├─ Precision@10
    ├─ Coverage
    └─ Diversity
    ↓
Compare & Report
```

## 📊 Dataset

### MovieLens 100K (Traditional System)
- **Users**: 943
- **Videos**: 1,682
- **Ratings**: 100,000
- **Rating Scale**: 1-5
- **Use Case**: Traditional collaborative filtering evaluation

### MicroLens 100K (Advanced System)
- **Purpose**: Designed specifically for short video recommendations
- **Features**: Video metadata, descriptions, user interactions
- **Use Case**: Deep learning and content-based methods
- **Download**: [MicroLens-100K Dataset](https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/wentaoma2024/youtube-shorts-recommender.git
cd youtube-shorts-recommender
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare data**

For MovieLens (Traditional System):
```bash
python data_prep.py
```

For MicroLens (Advanced System):
```bash
python data_prep_microlens.py
# Note: You need to download MicroLens dataset first
```

4. **Train models**

Traditional system:
```bash
python train.py
```

Advanced system:
```bash
python train_advanced.py
# This trains Two-Tower and BM25 models
```

5. **Run Streamlit demo**
```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` for the Streamlit interface.

**Optional: Start API server**
```bash
cd src/api
uvicorn app:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

**Compare System Performance**
```bash
python evaluate_comparison.py
# This will evaluate both traditional and advanced systems
# and generate comparison reports
```

## 📁 Project Structure

```
youtube-shorts-recommender/
├── src/
│   ├── models/
│   │   ├── recall_system.py          # Multi-path recall system
│   │   ├── ranking_model.py         # LightGBM ranking model
│   │   ├── two_tower_recall.py      # Two-Tower model for recall
│   │   ├── bm25_ranking.py          # BM25 coarse ranking
│   │   └── cross_encoder_ranking.py # Cross-Encoder fine ranking
│   ├── features/
│   │   └── feature_engineering.py   # Feature extraction
│   ├── evaluation/
│   │   └── metrics.py                # Evaluation metrics
│   └── api/
│       └── app.py                    # FastAPI application
├── data/                             # Dataset storage
├── models/                           # Saved models
├── data_prep.py                      # MovieLens preprocessing
├── data_prep_microlens.py            # MicroLens preprocessing
├── train.py                          # Traditional model training
├── train_advanced.py                 # Advanced model training
├── evaluate_comparison.py            # System comparison evaluation
├── streamlit_app.py                  # Streamlit demo
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🔌 API Endpoints

### Get Recommendations
```bash
POST /recommend
{
  "user_id": 1,
  "top_k": 10
}
```

### Recall Analysis
```bash
POST /recall
{
  "user_id": 1
}
```

### User Statistics
```bash
GET /user/{user_id}/stats
```

### Health Check
```bash
GET /health
```

See full API documentation at `/docs` when the server is running.


## 🛠️ Technology Stack

### Traditional System
- **Backend**: FastAPI, Python 3.12
- **ML Framework**: scikit-surprise, LightGBM
- **Data Processing**: pandas, numpy
- **Frontend**: Streamlit

### Advanced System
- **Deep Learning**: PyTorch, Transformers
- **NLP**: sentence-transformers, rank-bm25
- **Text Retrieval**: BM25 algorithm
- **Reranking**: Cross-Encoder models

## 📈 Performance

### Traditional System (MovieLens)
- **RMSE**: ~0.94 (5-fold cross-validation)
- **MAE**: ~0.74
- **Response Time**: 1-3 seconds per request
- **Memory Optimized**: Suitable for 8GB RAM systems

### Advanced System (MicroLens)
- **NDCG@10**: Evaluated on test set
- **Recall@10**: Evaluated on test set
- **Coverage**: Percentage of catalog covered
- **Diversity**: Recommendation diversity score

### System Comparison
Run `python evaluate_comparison.py` to compare:
- Traditional vs Advanced system performance
- Metrics: NDCG@10, Recall@10, Precision@10, Coverage, Diversity
- Performance improvement analysis

## 📝 License

This project uses the MovieLens 100K dataset, provided by GroupLens Research.

## 🔬 Evaluation & Comparison

### Running System Comparison

1. **Train both systems**:
```bash
# Train traditional system
python train.py

# Train advanced system
python train_advanced.py
```

2. **Run comparison evaluation**:
```bash
python evaluate_comparison.py
```

This will:
- Evaluate traditional system on test set
- Evaluate advanced system on test set
- Calculate metrics: NDCG@10, Recall@10, Precision@10, Coverage, Diversity
- Generate comparison report with improvement percentages
- Save results to CSV file

### Evaluation Metrics

- **NDCG@10**: Normalized Discounted Cumulative Gain at top 10
- **Recall@10**: Percentage of relevant items found in top 10
- **Precision@10**: Percentage of top 10 items that are relevant
- **Coverage**: Percentage of catalog items recommended
- **Diversity**: Average diversity of recommendation lists

## 📚 Usage Examples

### Using Traditional System
```python
from src.models.recall_system import MultiRecallSystem

recall_system = MultiRecallSystem(
    use_advanced_models=False
)
recommendations = recall_system.multi_recall(user_id=1, top_k=10)
```

### Using Advanced System
```python
from src.models.recall_system import MultiRecallSystem

recall_system = MultiRecallSystem(
    use_advanced_models=True  # Enable Two-Tower and BM25
)
recommendations = recall_system.multi_recall(
    user_id=1,
    recall_nums={
        'cf': 100,
        'popular': 50,
        'two_tower': 200,
        'bm25': 200
    }
)
```

### Using Cross-Encoder for Reranking
```python
from src.models.cross_encoder_ranking import CrossEncoderRanker

cross_encoder = CrossEncoderRanker(ratings_path, videos_path)
reranked = cross_encoder.rank(
    user_id=1,
    candidate_video_ids=[1, 2, 3, 4, 5],
    top_k=10
)
```

## 👤 Author

**Wentao Ma**
- GitHub: [@wentaoma2024](https://github.com/wentaoma2024)

## 🙏 Acknowledgments

- MovieLens dataset by GroupLens Research
- MicroLens dataset by Westlake University
- FastAPI for the excellent web framework
- Streamlit for the demo interface
- Hugging Face for pre-trained transformer models
