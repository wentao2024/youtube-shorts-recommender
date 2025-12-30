"""
FastAPI 推荐服务
"""
import sys
from pathlib import Path as PathLib
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn

# 添加项目根目录到路径
project_root = PathLib(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入排序模型，如果失败则只使用召回系统
try:
    from src.models.ranking_model import RankingModel, LIGHTGBM_AVAILABLE
    RANKING_MODEL_AVAILABLE = LIGHTGBM_AVAILABLE
    if not LIGHTGBM_AVAILABLE:
        print("警告: LightGBM不可用，排序功能将不可用，只使用召回系统")
except (ImportError, OSError) as e:
    print(f"警告: 排序模型不可用 ({e})，将只使用召回系统")
    RANKING_MODEL_AVAILABLE = False
    RankingModel = None
    LIGHTGBM_AVAILABLE = False

from src.models.recall_system import MultiRecallSystem
from fastapi.middleware.cors import CORSMiddleware


# 初始化FastAPI应用
app = FastAPI(
    title="🎬 YouTube Shorts Recommender API",
    description="""
    ## 🚀 Recommendation System API Service
    
    A collaborative filtering-based recommendation system for YouTube Shorts, 
    using the MovieLens 100K dataset as a proxy.
    
    ### ✨ Key Features
    
    - **Personalized Recommendations**: Generate personalized video recommendations based on user behavior
    - **Multi-path Recall**: Combines collaborative filtering, popularity, high-rating, and user similarity strategies
    - **Smart Ranking**: Uses machine learning models for precise ranking of candidate videos
    
    ### 📊 Dataset Information
    
    - Users: 943
    - Videos: 1,682
    - Ratings: 100,000
    
    ### 🎯 Quick Start
    
    1. Use `/recommend` endpoint to get recommendations
    2. Use `/recall` endpoint to view recall process
    3. Use `/user/{user_id}/stats` to view user statistics
    
    ### 📝 Notes
    
    - User ID range: 1-943
    - Recommended top_k: 10-50
    - Response time: 1-3 seconds
    """,
    version="1.0.0",
    contact={
        "name": "YouTube Shorts Recommender",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
    tags_metadata=[
        {
            "name": "Recommendations",
            "description": "Core recommendation functionality for generating personalized video recommendations",
        },
        {
            "name": "Recall",
            "description": "Recall system endpoints for viewing multi-path recall results",
        },
        {
            "name": "User",
            "description": "User information query endpoints",
        },
        {
            "name": "System",
            "description": "System health check and status endpoints",
        },
    ]
)

# 添加CORS中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
ranking_model: Optional[RankingModel] = None
recall_system: Optional[MultiRecallSystem] = None
use_ranking_model = False  # 是否使用排序模型


# 请求/响应模型
class RecommendationRequest(BaseModel):
    """Recommendation request model"""
    user_id: int = Field(
        ...,
        description="User ID, range: 1-943",
        example=1,
        ge=1,
        le=943
    )
    top_k: int = Field(
        default=10,
        description="Number of recommendations to return, recommended range: 5-50",
        example=10,
        ge=1,
        le=100
    )
    recall_nums: Optional[Dict[str, int]] = Field(
        default=None,
        description="Recall configuration for each path (optional)",
        example={
            "cf": 200,
            "popular": 100,
            "high_rating": 100,
            "similarity": 100
        }
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "top_k": 10,
                "recall_nums": {
                    "cf": 200,
                    "popular": 100,
                    "high_rating": 100,
                    "similarity": 100
                }
            }
        }


class RecommendationResponse(BaseModel):
    """Recommendation response model"""
    user_id: int = Field(..., description="User ID")
    recommendations: List[Dict[str, float]] = Field(
        ...,
        description="Recommendation list, sorted by score (high to low)",
        example=[
            {"video_id": 512, "score": 0.95},
            {"video_id": 513, "score": 0.92}
        ]
    )
    total_candidates: int = Field(..., description="Total number of recommendations returned")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "recommendations": [
                    {"video_id": 512, "score": 0.95},
                    {"video_id": 513, "score": 0.92},
                    {"video_id": 514, "score": 0.89}
                ],
                "total_candidates": 10
            }
        }


class RecallRequest(BaseModel):
    """Recall request model"""
    user_id: int = Field(
        ...,
        description="User ID, range: 1-943",
        example=1,
        ge=1,
        le=943
    )
    recall_nums: Optional[Dict[str, int]] = Field(
        default=None,
        description="Recall configuration for each path",
        example={
            "cf": 200,
            "popular": 100,
            "high_rating": 100,
            "similarity": 100
        }
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "recall_nums": {
                    "cf": 200,
                    "popular": 100,
                    "high_rating": 100,
                    "similarity": 100
                }
            }
        }


class RecallResponse(BaseModel):
    """Recall response model"""
    user_id: int
    recall_results: Dict[str, List[Dict[str, float]]]
    total_candidates: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    global ranking_model, recall_system, use_ranking_model
    
    try:
        print("Initializing recommendation system...")
        
        # If ranking model is available, try to load it
        if RANKING_MODEL_AVAILABLE and RankingModel is not None:
            try:
                ranking_model = RankingModel()
                
                # Try to load trained ranking model
                if ranking_model.ranking_model_path.exists():
                    ranking_model.load_model()
                    use_ranking_model = True
                    print("✓ Ranking model loaded successfully")
                else:
                    print("⚠️  Ranking model not found, using recall system only")
                    use_ranking_model = False
                
                # Initialize recall system
                recall_system = ranking_model.recall_system
            except Exception as e:
                print(f"⚠️  Ranking model initialization failed: {e}")
                print("Using recall system only")
                use_ranking_model = False
                # Directly initialize recall system (using absolute path from project root)
                project_root = PathLib(__file__).parent.parent.parent
                recall_system = MultiRecallSystem(
                    ratings_path=project_root / "data" / "ratings.csv",
                    model_path=project_root / "models" / "svd_model.pkl",
                    dataset_dir=project_root / "data" / "ml-100k"
                )
        else:
            # Use recall system only (using absolute path from project root)
            print("Ranking model unavailable, using recall system only")
            project_root = PathLib(__file__).parent.parent.parent
            recall_system = MultiRecallSystem(
                ratings_path=project_root / "data" / "ratings.csv",
                model_path=project_root / "models" / "svd_model.pkl",
                dataset_dir=project_root / "data" / "ml-100k"
            )
        
        print("✓ Recall system initialized successfully")
        print("Model loading complete, service ready")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get(
    "/",
    response_model=HealthResponse,
    tags=["System"],
    summary="Root",
    description="Quick check if API service is running normally"
)
async def root():
    """
    Root endpoint
    
    Used to quickly check if the API service is started and running normally.
    
    - **Returns**: Basic status information
    """
    return HealthResponse(
        status="ok",
        message="YouTube Shorts Recommender API is running"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check if all components of the recommendation system are loaded normally",
    responses={
        200: {
            "description": "服务正常",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "message": "Service is healthy"
                    }
                }
            }
        },
        503: {
            "description": "服务异常",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Model not loaded"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Health check endpoint
    
    Check the health status of the recommendation system, including:
    - Whether models are loaded successfully
    - Whether recall system is available
    - Whether service is running normally
    """
    if ranking_model is None:
        return HealthResponse(
            status="error",
            message="Model not loaded"
        )
    
    return HealthResponse(
        status="ok",
        message="Service is healthy"
    )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get Personalized Recommendations",
    description="""
    ## 🎯 Core Recommendation Endpoint
    
    Generate personalized video recommendations for users, combining multi-path recall and smart ranking.
    
    ### Workflow
    1. **Multi-path Recall**: Recall candidate videos from collaborative filtering, popularity, high-rating, user similarity channels
    2. **Feature Engineering**: Extract user features, video features, cross features
    3. **Smart Ranking**: Use machine learning models to rank candidate videos
    4. **Return Top K**: Return the K highest-scoring recommendations
    
    ### Use Cases
    - **New Users**: Automatically return popular videos (cold start)
    - **Existing Users**: Return personalized recommendations based on historical behavior
    """,
    responses={
        200: {
            "description": "推荐成功",
            "content": {
                "application/json": {
                    "example": {
                        "user_id": 1,
                        "recommendations": [
                            {"video_id": 512, "score": 0.95},
                            {"video_id": 513, "score": 0.92},
                            {"video_id": 514, "score": 0.89}
                        ],
                        "total_candidates": 10
                    }
                }
            }
        },
        500: {
            "description": "服务器错误",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error"
                    }
                }
            }
        }
    }
)
async def get_recommendations(request: RecommendationRequest):
    """
    Get recommendation results (recall + ranking)
    
    ### Parameters
    
    - **user_id**: User ID (1-943)
    - **top_k**: Number of recommendations to return (recommended: 5-50)
    - **recall_nums**: Recall configuration for each path (optional)
    
    ### Returns
    
    - **recommendations**: Recommendation list, sorted by score (high to low)
    - **score**: Recommendation score (0-1, higher is better)
    - **video_id**: Video ID (corresponds to movie ID in MovieLens dataset)
    """
    if recall_system is None:
        raise HTTPException(status_code=503, detail="Recall system not loaded")
    
    try:
        # 检查是否使用排序模型
        if use_ranking_model and ranking_model is not None and ranking_model.ranking_model is not None:
            # 完整的召回 + 排序流程
            results = ranking_model.recommend(
                request.user_id,
                request.top_k,
                request.recall_nums
            )
            
            recommendations = [
                {"video_id": vid, "score": float(score)}
                for vid, score in results
            ]
        else:
            # 只使用召回系统（按热门度排序）
            candidates = recall_system.multi_recall(
                request.user_id,
                request.recall_nums
            )
            
            # 获取视频统计信息用于排序
            video_stats = recall_system.video_stats
            candidate_scores = []
            
            for vid in candidates:
                stats = video_stats[video_stats['video_id'] == vid]
                if len(stats) > 0:
                    # 使用评分数量和平均评分的组合作为分数
                    score = stats.iloc[0]['rating_count'] * 0.7 + stats.iloc[0]['avg_rating'] * 10
                    candidate_scores.append((vid, score))
                else:
                    candidate_scores.append((vid, 0.0))
            
            # 按分数排序
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = [
                {"video_id": vid, "score": float(score)}
                for vid, score in candidate_scores[:request.top_k]
            ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            total_candidates=len(recommendations)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/recall",
    response_model=RecallResponse,
    tags=["Recall"],
    summary="Get Recall Candidates",
    description="""
    ## 🔍 Recall Analysis Endpoint
    
    Get candidate videos from each recall path without ranking. Used for analyzing and debugging the recall system.
    
    ### Recall Strategy Description
    
    1. **collaborative_filtering**: Collaborative filtering recall
       - Based on SVD model prediction of user ratings
       - Score = Predicted rating (1-5)
    
    2. **popularity**: Popularity-based recall
       - Sorted by number of ratings
       - Score = Rating count
    
    3. **high_rating**: High-rating recall
       - Videos with high average rating and sufficient rating count
       - Score = Average rating (1-5)
    
    4. **user_similarity**: User similarity recall
       - Based on videos liked by similar users
       - Score = Similarity score (0-1)
    
    ### Difference from /recommend
    
    - `/recommend`: Recall + Ranking → Returns final recommendations
    - `/recall`: Recall only → Shows recall results from each path
    """,
    responses={
        200: {
            "description": "召回成功",
            "content": {
                "application/json": {
                    "example": {
                        "user_id": 1,
                        "recall_results": {
                            "collaborative_filtering": [
                                {"video_id": 512, "score": 4.5}
                            ],
                            "popularity": [
                                {"video_id": 50, "score": 583}
                            ],
                            "high_rating": [
                                {"video_id": 318, "score": 4.5}
                            ],
                            "user_similarity": [
                                {"video_id": 100, "score": 0.8}
                            ]
                        },
                        "total_candidates": 300
                    }
                }
            }
        }
    }
)
async def get_recall_candidates(request: RecallRequest):
    """
    Get recall candidates (without ranking)
    
    ### Use Cases
    
    - Analyze the effectiveness of each recall path
    - Debug the recall system
    - Understand the internal workings of the recommendation system
    - A/B test different recall strategies
    """
    if recall_system is None:
        raise HTTPException(status_code=503, detail="Recall system not loaded")
    
    try:
        # 获取各路召回详情
        recall_details = recall_system.get_recall_details(
            request.user_id,
            request.recall_nums
        )
        
        # 格式化结果
        formatted_results = {}
        total_candidates = set()
        
        for recall_type, candidates in recall_details.items():
            formatted_results[recall_type] = [
                {"video_id": vid, "score": float(score)}
                for vid, score in candidates
            ]
            total_candidates.update([vid for vid, _ in candidates])
        
        return RecallResponse(
            user_id=request.user_id,
            recall_results=formatted_results,
            total_candidates=len(total_candidates)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/user/{user_id}/stats",
    tags=["User"],
    summary="Get User Statistics",
    description="""
    ## 👤 User Information Query Endpoint
    
    Get user statistics in the system, including:
    - Rating count
    - Average rating
    - Number of rated videos
    
    ### Use Cases
    
    - Understand user activity
    - Analyze user preferences
    - User analysis before personalized recommendations
    - User profiling
    """,
    responses={
        200: {
            "description": "查询成功",
            "content": {
                "application/json": {
                    "example": {
                        "user_id": 1,
                        "rated_videos_count": 39,
                        "average_rating": 3.62,
                        "rating_count": 39,
                        "unique_videos": 39
                    }
                }
            }
        },
        404: {
            "description": "用户不存在",
            "content": {
                "application/json": {
                    "example": {
                        "user_id": 99999,
                        "rated_videos_count": 0,
                        "message": "User not found in training data"
                    }
                }
            }
        }
    }
)
async def get_user_stats(
    user_id: int = Path(
        ...,
        description="User ID, range: 1-943",
        examples=[1, 100, 500],
        ge=1,
        le=943
    )
):
    """
    Get user statistics
    
    ### Return Fields
    
    - **rated_videos_count**: Number of videos rated by user
    - **average_rating**: User's average rating (1-5)
    - **rating_count**: Total number of ratings
    - **unique_videos**: Number of unique videos rated
    """
    if recall_system is None:
        raise HTTPException(status_code=503, detail="System not loaded")
    
    try:
        user_history = recall_system.user_history.get(user_id, set())
        
        # 从ratings数据计算用户统计（使用项目根目录的绝对路径）
        import pandas as pd
        project_root = PathLib(__file__).parent.parent.parent
        ratings = pd.read_csv(project_root / "data" / "ratings.csv")
        user_ratings = ratings[ratings['user_id'] == user_id]
        
        if len(user_ratings) > 0:
            return {
                "user_id": user_id,
                "rated_videos_count": len(user_history),
                "average_rating": float(user_ratings['rating'].mean()),
                "rating_count": int(len(user_ratings)),
                "unique_videos": int(user_ratings['video_id'].nunique())
            }
        else:
            return {
                "user_id": user_id,
                "rated_videos_count": 0,
                "message": "User not found in training data"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

