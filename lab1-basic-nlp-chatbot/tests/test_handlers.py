"""Test handlerow z atrapami obiektow Telegrama (bez sieci i bez tokenu)."""
import asyncio
import html
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot import handlers
from src.nlp.service import NlpService
from src.storage.repository import TaskRepository


@pytest.fixture
def ctx(tmp_path):
    repo = TaskRepository(tmp_path / "h.db")
    repo.init_schema()
    app = SimpleNamespace(bot_data={"repo": repo, "nlp": NlpService(6)})
    return SimpleNamespace(application=app)


def make_update(text, user_id=7):
    msg = MagicMock(text=text)
    msg.reply_text = AsyncMock()
    return SimpleNamespace(effective_message=msg, effective_user=SimpleNamespace(id=user_id))


def replies(update):
    return [c.args[0] for c in update.effective_message.reply_text.call_args_list]


def test_full_flow(ctx):
    u = make_update("/add Przygotować prezentację z przetwarzania języka naturalnego")
    asyncio.run(handlers.add_cmd(u, ctx))
    assert replies(u)[0].startswith("Task created: <code>")
    task_id = ctx.application.bot_data["repo"].list_tasks(7)[0].id

    u = make_update(f"/task {task_id} bigram")
    asyncio.run(handlers.task_cmd(u, ctx))
    assert "('przygotować', 'prezentację')" in html.unescape(replies(u)[0])

    u = make_update(f"/update {task_id[:8]} Nowy opis")
    asyncio.run(handlers.update_cmd(u, ctx))
    u = make_update("/tasks")
    asyncio.run(handlers.tasks_cmd(u, ctx))
    assert "Nowy opis" in replies(u)[0]

    u = make_update(f"/delete {task_id}")
    asyncio.run(handlers.delete_cmd(u, ctx))
    assert ctx.application.bot_data["repo"].list_tasks(7) == []


def test_user_errors_become_messages(ctx):
    u = make_update("/task")
    asyncio.run(handlers.task_cmd(u, ctx))
    assert replies(u)[0].startswith("⚠️ Uzycie")
    u = make_update("/delete abcd")
    asyncio.run(handlers.delete_cmd(u, ctx))
    assert "Nie znaleziono" in replies(u)[0]


def test_chunks_respect_limit():
    text = "\n".join(["x" * 100] * 100)
    assert all(len(c) <= 3500 for c in handlers._chunks(text))
    assert "".join(handlers._chunks(text)) == text
