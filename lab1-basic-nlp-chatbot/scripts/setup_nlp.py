"""Jednorazowe pobranie zasobow NLP: punkt_tab (NLTK) i model spaCy."""
import subprocess
import sys

import nltk

from src.config import load_settings

if __name__ == "__main__":
    model = load_settings(require_token=False).spacy_model
    print("Pobieram NLTK punkt_tab...")
    nltk.download("punkt_tab")
    print(f"Pobieram model spaCy {model}...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
    print("Gotowe.")
