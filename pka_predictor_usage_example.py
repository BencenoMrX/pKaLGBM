# How to load in Google Colab for example
import os

# Define your GitHub details (Replace with your actual username and repo name)
# BencenoMrX
# pKaLGBM
GITHUB_USER = "your_github_username"
REPO_NAME = "pka-colab-deploy"

# Pull the helper script and model file directly using raw URLs
print("Fetching model files from GitHub...")
!wget -q https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/pka_predictor.py
!wget -q https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/pka_lightgbm_model.pkl

print("Installing dependencies...")
!pip install -q rdkit lightgbm

# Import custom class from the downloaded script
from pka_predictor import PkaPredictor

# Initiate the pKa predictor
pka_engine = PkaPredictor(model_path='pka_lightgbm_model.pkl')

##### Usage example
# Single prediction
print(pka_engine.predict("C1=CC=C(C=C1)O")) # Phenol -> Output: ~9.95

# High-throughput batch prediction on thousands of molecules
list_of_smiles = ["CC(=O)O", "CCN", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]

predictions = pka_engine.predict_batch(list_of_smiles)
print(predictions)
