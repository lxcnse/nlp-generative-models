"""Handlery komend Telegrama.

Kazdy handler: (1) pobiera argumenty, (2) wola repozytorium / serwis NLP,
(3) odpowiada. Logika biznesowa jest poza tym plikiem.
"""
from __future__ import annotations

import asyncio
import html
import logging
from functools import wraps
from typing import Awaitable, Callable

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from src.bot.messages import HELP_TEXT, STATUS_ICONS
from src.bot.parsing import parse_add_args, split_first_word
from src.errors import BotError, InvalidInputError
from src.nlp.service import NlpService
from src.storage.models import STATUSES, Task
from src.storage.repository import TaskRepository

log = logging.getLogger(__name__)
MAX_CHUNK = 3500  # limit Telegrama to 4096 znakow; zostawiamy zapas na znaczniki HTML

Handler = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]


# ------------------------------------------------------------------ pomocnicze
def safe(func: Handler) -> Handler:
    """Dekorator: oczekiwane bledy (BotError) -> czytelny komunikat, reszta -> log."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await func(update, context)
        except BotError as exc:
            await update.effective_message.reply_text(f"⚠️ {exc}")
        except Exception:
            log.exception("Nieoczekiwany blad w %s", func.__name__)
            await update.effective_message.reply_text("❌ Wystapil nieoczekiwany blad.")

    return wrapper


def _repo(context: ContextTypes.DEFAULT_TYPE) -> TaskRepository:
    return context.application.bot_data["repo"]


def _nlp(context: ContextTypes.DEFAULT_TYPE) -> NlpService:
    return context.application.bot_data["nlp"]


def _owner(update: Update) -> int:
    return update.effective_user.id


def _args(update: Update) -> str:
    """Tekst po komendzie ('/add cos tam' -> 'cos tam'), z zachowaniem spacji."""
    parts = (update.effective_message.text or "").split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _chunks(text: str, limit: int = MAX_CHUNK) -> list[str]:
    """Dzieli dlugi tekst na kawalki <= limit, tnac na granicach linii."""
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:                    # pojedyncza gigantyczna linia
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = ""
        current += line
    if current:
        chunks.append(current)
    return chunks or [""]


async def _reply_pre(update: Update, text: str) -> None:
    for chunk in _chunks(text):
        await update.effective_message.reply_text(
            f"<pre>{html.escape(chunk)}</pre>", parse_mode=ParseMode.HTML
        )


def _format_task(task: Task) -> str:
    icon = STATUS_ICONS.get(task.status, "⬜")
    line = f"{icon} <code>{task.id}</code>\n    {html.escape(task.description)}"
    line += f"\n    <i>priorytet: {task.priority}</i>"
    if task.relation:
        line += f" · <i>podzadanie: <code>{task.relation[:8]}</code></i>"
    return line


# -------------------------------------------------------------------- komendy
@safe
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


@safe
async def tasks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tasks = _repo(context).list_tasks(_owner(update))
    if not tasks:
        await update.effective_message.reply_text("Brak zadan. Dodaj pierwsze: /add Kupic mleko")
        return
    body = "\n\n".join(_format_task(t) for t in tasks)
    for chunk in _chunks(body):
        await update.effective_message.reply_text(chunk, parse_mode=ParseMode.HTML)


@safe
async def add_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = parse_add_args(_args(update))
    task = _repo(context).create(_owner(update), args.description, args.parent, args.priority)
    await update.effective_message.reply_text(
        f"Task created: <code>{task.id}</code>", parse_mode=ParseMode.HTML
    )


@safe
async def update_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    task_ref, new_text = split_first_word(_args(update))
    if not task_ref or not new_text:
        raise InvalidInputError("Uzycie: /update {id} {nowy opis}")
    task = _repo(context).update_description(_owner(update), task_ref, new_text)
    await update.effective_message.reply_text(
        f"Task updated: <code>{task.id}</code>", parse_mode=ParseMode.HTML
    )


@safe
async def delete_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    task_ref, _ = split_first_word(_args(update))
    if not task_ref:
        raise InvalidInputError("Uzycie: /delete {id}")
    task = _repo(context).delete(_owner(update), task_ref)
    await update.effective_message.reply_text(
        f"Task deleted: <code>{task.id}</code>", parse_mode=ParseMode.HTML
    )


@safe
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    task_ref, status = split_first_word(_args(update))
    if not task_ref or not status:
        raise InvalidInputError(f"Uzycie: /status {{id}} {'|'.join(STATUSES)}")
    task = _repo(context).set_status(_owner(update), task_ref, status.strip().lower())
    await update.effective_message.reply_text(f"{STATUS_ICONS[task.status]} Status: {task.status}")


@safe
async def ops_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [f"{name:<10} {desc}" for name, desc in _nlp(context).available_operations().items()]
    await _reply_pre(update, "Uzycie: /task {id} {operacja}\n\n" + "\n".join(lines))


@safe
async def task_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    task_ref, operation = split_first_word(_args(update))
    operation = operation.split()[0] if operation else ""
    if not task_ref or not operation:
        raise InvalidInputError("Uzycie: /task {id} {operacja}. Lista operacji: /ops")

    repo, nlp, owner = _repo(context), _nlp(context), _owner(update)
    task = repo.get(owner, task_ref)
    corpus = [t.description for t in repo.list_tasks(owner)]
    # NLP bywa wolne (ladowanie modeli) - w osobnym watku, zeby nie blokowac petli zdarzen bota.
    result = await asyncio.to_thread(nlp.run, operation, task.description, corpus)
    await _reply_pre(update, f"Zadanie {task.id}\n\n{result}")


# ---------------------------------------------------------------- rejestracja
COMMAND_MENU = [
    BotCommand("tasks", "lista zadan"),
    BotCommand("add", "dodaj zadanie"),
    BotCommand("update", "zmien opis zadania"),
    BotCommand("delete", "usun zadanie"),
    BotCommand("status", "zmien status"),
    BotCommand("task", "operacja NLP na zadaniu"),
    BotCommand("ops", "lista operacji NLP"),
    BotCommand("help", "pomoc"),
]


async def post_init(application: Application) -> None:
    await application.bot.set_my_commands(COMMAND_MENU)


def register_handlers(app: Application) -> None:
    for name, func in [
        ("start", start_cmd), ("help", start_cmd), ("tasks", tasks_cmd), ("add", add_cmd),
        ("update", update_cmd), ("delete", delete_cmd), ("status", status_cmd),
        ("ops", ops_cmd), ("task", task_cmd),
    ]:
        app.add_handler(CommandHandler(name, func))
