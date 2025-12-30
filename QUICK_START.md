# 🚀 Quick Start Guide

## For Local Testing

### 1. Streamlit Demo (Easiest) ⭐

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data and train model
python data_prep.py
python train.py

# Start Streamlit demo
streamlit run streamlit_app.py
```

Visit: `http://localhost:8501`

**That's it!** No need to start API separately - Streamlit handles everything.

---

### 2. API + Frontend

**Terminal 1 - API:**
```bash
cd src/api
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python3 -m http.server 8080
```

Visit: `http://localhost:8080`

---

## For Deployment

### Render.com (Recommended)

1. Fork repository to your GitHub
2. Go to [Render.com](https://render.com)
3. Create Web Service → Connect GitHub repo
4. Build Command: `pip install -r requirements.txt && python data_prep.py && python train.py`
5. Start Command: `cd src/api && uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Deploy!

### Railway.app

1. Fork repository
2. Go to [Railway.app](https://railway.app)
3. New Project → Deploy from GitHub
4. Auto-detects configuration
5. Deploy!

### Hugging Face Spaces (Streamlit)

1. Create Space on [Hugging Face](https://huggingface.co/spaces)
2. SDK: Streamlit
3. Upload `streamlit_app.py` and `requirements.txt`
4. Deploy!

---

## Push to GitHub

```bash
./push_to_github.sh
```

Or manually:
```bash
git init
git remote add origin https://github.com/wentaoma2024/youtube-shorts-recommender.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

---

## After Deployment

1. Update `README.md` with live links
2. Add links to your resume
3. Share with interviewers! 🎉

