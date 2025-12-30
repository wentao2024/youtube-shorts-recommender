# Files to Push to GitHub

## ✅ Essential Files (Must Include)

```
youtube-shorts-recommender/
├── src/                          ✅ Source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── recall_system.py
│   │   └── ranking_model.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   └── api/
│       ├── __init__.py
│       └── app.py
├── data_prep.py                  ✅ Data preparation
├── train.py                      ✅ Model training
├── recommend.py                  ✅ Basic recommendation
├── streamlit_app.py              ✅ Streamlit demo (MAIN DEMO)
├── requirements.txt              ✅ Dependencies
├── README.md                     ✅ Main documentation (English)
├── .gitignore                    ✅ Git ignore rules
├── render.yaml                   ✅ Render deployment config
├── railway.json                  ✅ Railway deployment config
├── Dockerfile                    ✅ Docker config
└── DEPLOYMENT.md                 ✅ Deployment guide
```

## ❌ Files to Exclude (in .gitignore)

- `data/ml-100k/` - Large dataset (will be downloaded during build)
- `data/*.csv` - Processed data (will be generated)
- `data/ml-100k.zip` - Downloaded zip file
- `models/*.pkl` - Trained models (will be generated during build)
- `.venv/` - Virtual environment
- `__pycache__/` - Python cache
- `frontend/` - Optional (can exclude if using Streamlit only)
- All Chinese documentation files (keep only English)

## 📝 Optional Files (Can Include)

- `frontend/` - Web frontend (optional, Streamlit is simpler)
- `test_*.py` - Test files
- `explore_u_data.py` - Data exploration script
- `QUICK_START.md` - Quick start guide

## 🎯 Minimal Setup for GitHub

**Minimum files needed:**
1. `src/` - All source code
2. `data_prep.py` - Data preparation
3. `train.py` - Model training
4. `streamlit_app.py` - **Main demo** (this is the key!)
5. `recommend.py` - Recommendation functions
6. `requirements.txt` - Dependencies
7. `README.md` - Documentation
8. `.gitignore` - Ignore rules
9. Deployment configs (`render.yaml`, `railway.json`, `Dockerfile`)

**Total size**: ~500KB (without data/models)

