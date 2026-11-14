#!/bin/bash
# Automated GitHub Repository Initializer & Push Script for Vishal Pandey
# Usage: ./push_to_github.sh <your-github-username>

USERNAME=${1:-"vishalpandey"}
echo "Initializing Git repository for $USERNAME..."
git init
git add .
git commit -m "Initial commit: Add XAUUSD Quant Backtester, AI Placement/ATS Scanner, and Food Delivery ETA ML projects"
git branch -M main
echo "Now create an empty repository named 'vishal-quant-ai-portfolio' on https://github.com/new"
echo "Then run:"
echo "git remote add origin https://github.com/$USERNAME/vishal-quant-ai-portfolio.git"
echo "git push -u origin main"
