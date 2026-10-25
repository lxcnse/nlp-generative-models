"""Punkt wejscia: sklada aplikacje (konfiguracja -> baza -> NLP -> bot) i uruchamia polling."""
from __future__ import annotations

import logging

from telegram.ext import Application, ApplicationBuilder

from src.bot.handlers import post_init, register_handlers
from src.config import Settings, load_settings
from src.nlp.service import NlpService
from src.storage.repository import TaskRepository

log = logging.getLogger("todobot")


def build_application(settings: Settings) -> Application:
    repo = TaskRepository(settings.db_path)
    repo.init_schema()                                    # tworzy tabele, jesli brak
    nlp = NlpService(settings.xgram, settings.spacy_model)

    app = ApplicationBuilder().token(settings.telegram_token).post_init(post_init).build()
    # bot_data to wspolny slownik dostepny w kazdym handlerze (proste "wstrzykiwanie zaleznosci")
    app.bot_data["repo"] = repo
    app.bot_data["nlp"] = nlp
    register_handlers(app)
    return app


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # nie loguj tokenu w URL-ach
    log.info("Start bota (x-gram = %d-gram, baza: %s)", settings.xgram, settings.db_path)
    build_application(settings).run_polling()


if __name__ == "__main__":
    main()
