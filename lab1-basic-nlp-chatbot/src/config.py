"""Konfiguracja aplikacji wczytywana ze zmiennych srodowiskowych / pliku .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def xgram_size(album_number: str | int) -> int:
    """Wylicza wartosc x dla indywidualnego x-gramu z numeru albumu.

    Zasada z polecenia (patrz ostatnia cyfra numeru albumu):
      * 4-9 -> x = ta cyfra
      * 0-3 -> x = cyfra + 4
    """
    digits = [c for c in str(album_number) if c.isdigit()]
    if not digits:
        raise ValueError("ALBUM_NUMBER musi zawierac przynajmniej jedna cyfre")
    last = int(digits[-1])
    return last if last >= 4 else last + 4


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    album_number: str
    db_path: Path
    spacy_model: str
    log_level: str

    @property
    def xgram(self) -> int:
        return xgram_size(self.album_number)


def load_settings(require_token: bool = True) -> Settings:
    load_dotenv()  # wczytuje .env z biezacego katalogu (jesli istnieje)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if require_token and not token:
        raise RuntimeError(
            "Brak TELEGRAM_BOT_TOKEN. Skopiuj .env.example jako .env i wpisz token od @BotFather."
        )
    return Settings(
        telegram_token=token,
        album_number=os.getenv("ALBUM_NUMBER", "123456").strip(),
        db_path=Path(os.getenv("DB_PATH", "data/tasks.db")),
        spacy_model=os.getenv("SPACY_MODEL", "pl_core_news_sm").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
