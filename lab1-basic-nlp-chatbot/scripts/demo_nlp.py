"""Demo bez Telegrama:  python -m scripts.demo_nlp "Przygotowac prezentacje" [operacja]"""
import sys

from src.config import load_settings
from src.nlp.service import NlpService

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Przygotować prezentację z przetwarzania języka naturalnego"
    op = sys.argv[2] if len(sys.argv) > 2 else "all"
    s = load_settings(require_token=False)
    print(NlpService(s.xgram, s.spacy_model).run(op, text, [text]))
