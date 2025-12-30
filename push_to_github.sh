#!/bin/bash

# Script to prepare and push project to GitHub

echo "=========================================="
echo "Preparing project for GitHub"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Check remote
if ! git remote | grep -q origin; then
    echo "Adding remote repository..."
    git remote add origin https://github.com/wentaoma2024/youtube-shorts-recommender.git
fi

echo ""
echo "Files to be committed:"
git status --short

echo ""
read -p "Continue with commit and push? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

# Add all files
echo ""
echo "Staging files..."
git add .

# Commit
echo ""
echo "Creating commit..."
git commit -m "Initial commit: YouTube Shorts Recommender System

- Multi-path recall system (CF, Popularity, High Rating, User Similarity)
- FastAPI backend with interactive API documentation
- Streamlit demo interface
- LightGBM ranking model (optional)
- Feature engineering pipeline
- Deployment ready (Render/Railway)
- Full English documentation
- Production-ready ML recommendation system"

# Push
echo ""
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "=========================================="
echo "✅ Successfully pushed to GitHub!"
echo "=========================================="
echo ""
echo "Repository: https://github.com/wentaoma2024/youtube-shorts-recommender"
echo ""
echo "Next steps:"
echo "1. Deploy to Render.com or Railway.app"
echo "2. Update README.md with live demo links"
echo "3. Add repository description and topics"

