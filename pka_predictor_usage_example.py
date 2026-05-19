# @title 1. Initialize pKa Predictor from GitHub
import os

# Define your GitHub details (Replace with your actual username and repo name)
GITHUB_USER = "your_github_username"
REPO_NAME = "pka-colab-deploy"

# Pull the helper script and model file directly using raw URLs
print("📥 Fetching model files from GitHub...")
!wget -q https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/pka_predictor.py
!wget -q https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/pka_lightgbm_model.pkl

print("📦 Installing core dependencies...")
!pip install -q rdkit lightgbm

# Import your custom class from the downloaded script
from pka_predictor import PkaPredictor

# Instantiate the predictor
pka_engine = PkaPredictor(model_path='pka_lightgbm_model.pkl')
print("✅ pKa Engine is live and deployed successfully!")