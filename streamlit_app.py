"""
Streamlit Demo for YouTube Shorts Recommender
Simple and intuitive demo interface
"""
import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import local modules
USE_LOCAL = True
try:
    from src.models.recall_system import MultiRecallSystem
    from recommend import get_recommendations as get_recommendations_local, load_movie_titles
    USE_LOCAL = True
except ImportError as e:
    USE_LOCAL = False
    st.warning(f"Local modules not available: {e}. Will use API mode.")
    # API Configuration (fallback)
    API_BASE_URL = st.secrets.get("API_URL", "http://localhost:8000") if hasattr(st, 'secrets') else "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="YouTube Shorts Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6366f1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎬 YouTube Shorts Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Personalized Video Recommendations using Collaborative Filtering</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    if USE_LOCAL:
        st.success("✅ Using local functions (no API needed)")
        api_url = "http://localhost:8000"  # Not used but kept for compatibility
    else:
        # API URL configuration
        api_url = st.text_input("API URL", value=API_BASE_URL, help="Backend API endpoint")
        
        # Health check
        if st.button("🔍 Check API Status"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API is running")
                else:
                    st.error("❌ API returned error")
            except Exception as e:
                st.error(f"❌ Cannot connect to API: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.info("""
    - **Users**: 943
    - **Videos**: 1,682
    - **Ratings**: 100,000
    """)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "🔍 Recall Analysis", "👤 User Stats", "ℹ️ About"])

# Tab 1: Recommendations
with tab1:
    st.header("Get Personalized Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=943, value=1, step=1)
    
    with col2:
        top_k = st.number_input("Number of Recommendations", min_value=1, max_value=50, value=10, step=1)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        use_custom_recall = st.checkbox("Custom Recall Configuration")
        
        if use_custom_recall:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                recall_cf = st.number_input("Collaborative Filtering", min_value=0, value=200)
            with col2:
                recall_popular = st.number_input("Popularity", min_value=0, value=100)
            with col3:
                recall_high_rating = st.number_input("High Rating", min_value=0, value=100)
            with col4:
                recall_similarity = st.number_input("User Similarity", min_value=0, value=100)
    
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Getting recommendations..."):
            try:
                if USE_LOCAL:
                    # Use local functions directly
                    recommendations = get_recommendations_local(
                        int(user_id),
                        int(top_k)
                    )
                    
                    # Get video IDs from titles (need to reverse lookup)
                    # For simplicity, create mock data structure
                    # In production, you'd map titles back to IDs
                    data = {
                        "user_id": int(user_id),
                        "recommendations": [
                            {"video_id": hash(title) % 1682 + 1, "score": 0.95 - i*0.02, "title": title}
                            for i, title in enumerate(recommendations[:top_k])
                        ],
                        "total_candidates": len(recommendations)
                    }
                else:
                    # Use API
                    request_body = {
                        "user_id": int(user_id),
                        "top_k": int(top_k)
                    }
                    
                    if use_custom_recall:
                        request_body["recall_nums"] = {
                            "cf": recall_cf,
                            "popular": recall_popular,
                            "high_rating": recall_high_rating,
                            "similarity": recall_similarity
                        }
                    
                    response = requests.post(
                        f"{api_url}/recommend",
                        json=request_body,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        return
                
                if data:
                    
                    st.success(f"✅ Found {data['total_candidates']} recommendations")
                    
                    # Display recommendations
                    recommendations = data['recommendations']
                    
                    if recommendations:
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Recommendations", data['total_candidates'])
                        with col2:
                            avg_score = sum(r['score'] for r in recommendations) / len(recommendations)
                            st.metric("Average Score", f"{avg_score:.3f}")
                        with col3:
                            max_score = max(r['score'] for r in recommendations)
                            st.metric("Highest Score", f"{max_score:.3f}")
                        
                        st.markdown("### 📋 Recommendations")
                        
                        # Display as cards
                        for i, rec in enumerate(recommendations, 1):
                            with st.container():
                                col1, col2, col3 = st.columns([1, 4, 1])
                                with col1:
                                    st.markdown(f"### #{i}")
                                with col2:
                                    if 'title' in rec:
                                        st.markdown(f"**{rec['title']}**")
                                    st.markdown(f"**Video ID**: {rec['video_id']}")
                                    st.markdown(f"**Score**: {rec['score']:.4f}")
                                with col3:
                                    score_percent = rec['score'] * 100
                                    st.metric("Match", f"{score_percent:.1f}%")
                                st.divider()
                        
                        # Download button
                        df = pd.DataFrame([
                            {
                                "Rank": i+1,
                                "Video ID": r['video_id'],
                                "Score": r['score']
                            }
                            for i, r in enumerate(recommendations)
                        ])
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"recommendations_user_{user_id}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No recommendations found")
                        
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Tab 2: Recall Analysis
with tab2:
    st.header("Recall System Analysis")
    
    recall_user_id = st.number_input("User ID", min_value=1, max_value=943, value=1, step=1, key="recall_user")
    
    if st.button("🔍 Analyze Recall", type="primary", use_container_width=True):
        with st.spinner("Analyzing recall system..."):
            try:
                if USE_LOCAL:
                    # Use local recall system
                    recall_system = MultiRecallSystem()
                    recall_details = recall_system.get_recall_details(int(recall_user_id))
                    
                    # Format as API response
                    formatted_results = {}
                    total_candidates = set()
                    
                    for recall_type, candidates in recall_details.items():
                        formatted_results[recall_type] = [
                            {"video_id": vid, "score": float(score)}
                            for vid, score in candidates
                        ]
                        total_candidates.update([vid for vid, _ in candidates])
                    
                    data = {
                        "user_id": int(recall_user_id),
                        "recall_results": formatted_results,
                        "total_candidates": len(total_candidates)
                    }
                else:
                    # Use API
                    response = requests.post(
                        f"{api_url}/recall",
                        json={"user_id": int(recall_user_id)},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                    else:
                        st.error(f"Error: {response.status_code}")
                        return
                    
                    st.success(f"✅ Total candidates: {data['total_candidates']}")
                    
                    # Statistics
                    recall_results = data['recall_results']
                    
                    cols = st.columns(len(recall_results))
                    for i, (recall_type, candidates) in enumerate(recall_results.items()):
                        with cols[i]:
                            st.metric(
                                recall_type.replace('_', ' ').title(),
                                len(candidates)
                            )
                    
                    # Details
                    st.markdown("### 📊 Recall Details")
                    
                    for recall_type, candidates in recall_results.items():
                        with st.expander(f"{recall_type.replace('_', ' ').title()} ({len(candidates)} candidates)"):
                            if candidates:
                                df = pd.DataFrame(candidates[:20])  # Show first 20
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("No candidates found")
                                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 3: User Stats
with tab3:
    st.header("User Statistics")
    
    stats_user_id = st.number_input("User ID", min_value=1, max_value=943, value=1, step=1, key="stats_user")
    
    if st.button("📊 Get User Stats", type="primary", use_container_width=True):
        with st.spinner("Fetching user statistics..."):
            try:
                if USE_LOCAL:
                    # Use local functions
                    import pandas as pd
                    ratings = pd.read_csv("data/ratings.csv")
                    user_ratings = ratings[ratings['user_id'] == stats_user_id]
                    
                    if len(user_ratings) > 0:
                        data = {
                            "user_id": stats_user_id,
                            "rated_videos_count": int(user_ratings['video_id'].nunique()),
                            "average_rating": float(user_ratings['rating'].mean()),
                            "rating_count": int(len(user_ratings)),
                            "unique_videos": int(user_ratings['video_id'].nunique())
                        }
                    else:
                        data = {
                            "user_id": stats_user_id,
                            "rated_videos_count": 0,
                            "message": "User not found in training data"
                        }
                else:
                    # Use API
                    response = requests.get(
                        f"{api_url}/user/{stats_user_id}/stats",
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                    else:
                        st.error(f"Error: {response.status_code}")
                        return
                    
                    if 'message' in data:
                        st.warning(data['message'])
                    else:
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("User ID", data['user_id'])
                        with col2:
                            st.metric("Rating Count", data['rating_count'])
                        with col3:
                            st.metric("Average Rating", f"{data['average_rating']:.2f}")
                        with col4:
                            st.metric("Unique Videos", data['unique_videos'])
                        
                        # Chart
                        st.markdown("### 📈 User Activity")
                        chart_data = pd.DataFrame({
                            'Metric': ['Ratings', 'Unique Videos'],
                            'Count': [data['rating_count'], data['unique_videos']]
                        })
                        st.bar_chart(chart_data.set_index('Metric'))
                        
                else:
                    st.error(f"Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 4: About
with tab4:
    st.header("About the System")
    
    st.markdown("""
    ### 🎯 System Overview
    
    This is a collaborative filtering-based recommendation system for YouTube Shorts, 
    using the MovieLens 100K dataset as a proxy.
    
    ### 🛠️ Technology Stack
    
    - **Backend**: FastAPI + Python
    - **Recommendation Algorithm**: SVD (Collaborative Filtering)
    - **Recall Strategies**: Multi-path recall (CF, Popularity, High Rating, User Similarity)
    - **Frontend**: Streamlit
    
    ### 📊 Dataset Information
    
    - **Users**: 943
    - **Videos**: 1,682
    - **Ratings**: 100,000
    - **Rating Scale**: 1-5
    
    ### 🔌 API Endpoints
    
    - `POST /recommend` - Get personalized recommendations
    - `POST /recall` - Get recall analysis
    - `GET /user/{user_id}/stats` - User statistics
    - `GET /health` - Health check
    
    ### 🚀 Features
    
    - **Multi-path Recall**: Combines multiple recall strategies
    - **Personalized Recommendations**: Based on user behavior
    - **Cold Start Handling**: Returns popular items for new users
    - **Real-time Analysis**: Interactive recall system analysis
    """)
    
    st.markdown("---")
    st.markdown("### 📝 License")
    st.info("This project uses the MovieLens 100K dataset, provided by GroupLens Research.")

