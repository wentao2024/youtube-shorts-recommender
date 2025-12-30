# 🎬 YouTube Shorts Recommender

A production-ready recommendation system for short videos using collaborative filtering, built with FastAPI and deployed for easy access.

## 🌐 Live Demo

- **API Documentation**: [https://your-recommender.onrender.com/docs](https://your-recommender.onrender.com/docs)
- **Streamlit Demo**: [https://your-recommender.streamlit.app](https://your-recommender.streamlit.app)

## ✨ Features

- **Multi-path Recall System**: Combines collaborative filtering, popularity, high-rating, and user similarity strategies
- **Personalized Recommendations**: SVD-based collaborative filtering algorithm
- **Cold Start Handling**: Returns popular items for new users
- **Feature Engineering**: Comprehensive feature extraction for ranking
- **LightGBM Ranking**: Advanced ranking model (optional)
- **RESTful API**: FastAPI-based backend with interactive documentation
- **Streamlit Demo**: Simple and intuitive web interface
- **Production Ready**: Deployed on Render/Railway for easy access

## 🏗️ Architecture

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

## 📊 Dataset

- **Dataset**: MovieLens 100K (used as proxy for YouTube Shorts)
- **Users**: 943
- **Videos**: 1,682
- **Ratings**: 100,000
- **Rating Scale**: 1-5

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
```bash
python data_prep.py
```

4. **Train model**
```bash
python train.py
```

5. **Start API server**
```bash
cd src/api
python app.py
```

6. **Run Streamlit demo** (in another terminal)
```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` for the Streamlit interface.

## 📁 Project Structure

```
youtube-shorts-recommender/
├── src/
│   ├── models/
│   │   ├── recall_system.py      # Multi-path recall system
│   │   └── ranking_model.py      # LightGBM ranking model
│   ├── features/
│   │   └── feature_engineering.py  # Feature extraction
│   └── api/
│       └── app.py                 # FastAPI application
├── data/                          # Dataset storage
├── models/                        # Saved models
├── frontend/                      # Web frontend (optional)
├── data_prep.py                   # Data preprocessing
├── train.py                       # Model training
├── streamlit_app.py               # Streamlit demo
├── requirements.txt               # Dependencies
└── README.md                      # This file
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

## 🚢 Deployment

### Render.com (Recommended)

1. Fork this repository
2. Go to [Render.com](https://render.com)
3. Create a new Web Service
4. Connect your GitHub repository
5. Use these settings:
   - **Build Command**: `pip install -r requirements.txt && python data_prep.py && python train.py`
   - **Start Command**: `cd src/api && uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Deploy!

### Railway.app

1. Fork this repository
2. Go to [Railway.app](https://railway.app)
3. Create a new project from GitHub
4. Railway will auto-detect the configuration
5. Deploy!

### Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Upload your code
3. Add `requirements.txt`
4. Set Space SDK to "Streamlit"
5. Deploy!

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.12
- **ML Framework**: scikit-surprise, LightGBM
- **Data Processing**: pandas, numpy
- **Frontend**: Streamlit, HTML/CSS/JavaScript
- **Deployment**: Render.com / Railway.app

## 📈 Performance

- **RMSE**: ~0.94 (5-fold cross-validation)
- **MAE**: ~0.74
- **Response Time**: 1-3 seconds per request
- **Memory Optimized**: Suitable for 8GB RAM systems

## 📝 License

This project uses the MovieLens 100K dataset, provided by GroupLens Research.

## 👤 Author

**Wentao Ma**
- GitHub: [@wentaoma2024](https://github.com/wentaoma2024)

## 🙏 Acknowledgments

- MovieLens dataset by GroupLens Research
- FastAPI for the excellent web framework
- Streamlit for the demo interface
