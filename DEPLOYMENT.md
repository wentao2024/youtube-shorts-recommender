# 🚀 Deployment Guide

## Free Deployment Options

### Option 1: Render.com (Recommended) ⭐

**Steps:**

1. **Fork this repository** to your GitHub account

2. **Go to [Render.com](https://render.com)** and sign up/login

3. **Create a New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the forked repository

4. **Configure the service:**
   - **Name**: `youtube-shorts-recommender` (or any name you like)
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt && python data_prep.py && python train.py
     ```
   - **Start Command**: 
     ```bash
     cd src/api && uvicorn app:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: Free tier is fine

5. **Deploy!**
   - Click "Create Web Service"
   - Wait for build and deployment (5-10 minutes)
   - Your API will be available at: `https://your-app-name.onrender.com`

6. **Update API URL in Streamlit**
   - Edit `streamlit_app.py`
   - Change `API_BASE_URL` to your Render URL

**Note**: Render free tier spins down after 15 minutes of inactivity. First request may take 30-60 seconds.

---

### Option 2: Railway.app

**Steps:**

1. **Fork this repository** to your GitHub account

2. **Go to [Railway.app](https://railway.app)** and sign up/login

3. **Create a New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

4. **Railway will auto-detect the configuration**
   - It will use the `railway.json` file
   - Or manually set:
     - **Start Command**: `cd src/api && uvicorn app:app --host 0.0.0.0 --port $PORT`

5. **Deploy!**
   - Railway will automatically build and deploy
   - Your API will be available at: `https://your-app-name.up.railway.app`

**Note**: Railway free tier gives $5 credit per month.

---

### Option 3: Hugging Face Spaces (For Streamlit)

**Steps:**

1. **Go to [Hugging Face Spaces](https://huggingface.co/spaces)**

2. **Create a New Space**
   - Click "Create new Space"
   - Name: `youtube-shorts-recommender`
   - SDK: **Streamlit**
   - Visibility: Public

3. **Upload files**
   - Upload all project files
   - Make sure `streamlit_app.py` is in the root
   - Upload `requirements.txt`

4. **Configure Space**
   - Add environment variables if needed
   - Set API URL in `streamlit_app.py` to your deployed API

5. **Deploy!**
   - Hugging Face will automatically build and deploy
   - Your app will be at: `https://huggingface.co/spaces/your-username/youtube-shorts-recommender`

---

## Environment Variables

For production deployment, you may want to set:

- `API_URL`: Backend API URL (for Streamlit)
- `PYTHON_VERSION`: Python version (3.12)

---

## Post-Deployment Checklist

- [ ] Test API endpoints at `/docs`
- [ ] Test Streamlit demo (if deployed)
- [ ] Update README with live links
- [ ] Add links to your resume/portfolio
- [ ] Test from different devices

---

## Troubleshooting

### Build Fails
- Check build logs for errors
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

### API Not Responding
- Check if service is running (not spun down)
- Verify port configuration
- Check CORS settings

### Data/Models Not Found
- Ensure data files are included in deployment
- Or use build command to download/prepare data
- Models should be generated during build

---

## Quick Links Template

After deployment, add to your README:

```markdown
## 🌐 Live Demo

- **API Documentation**: https://your-app.onrender.com/docs
- **Streamlit Demo**: https://your-app.streamlit.app
```

