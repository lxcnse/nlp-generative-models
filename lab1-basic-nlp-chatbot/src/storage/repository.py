"""Warstwa dostepu do danych (wzorzec Repository) oparta o sqlite3.

Reszta aplikacji nie zna SQL-a - rozmawia tylko z klasa TaskRepository.
"""
from __future__ import annotations

import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from src.errors import (
    AmbiguousTaskIdError,
    InvalidInputError,
    InvalidRelationError,
    TaskNotFoundError,
)
from src.storage.models import PRIORITIES, STATUSES, Task

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    owner_id    INTEGER NOT NULL,
    description TEXT NOT NULL,
    relation    TEXT REFERENCES tasks(id) ON DELETE SET NULL,
    status      TEXT NOT NULL DEFAULT 'todo',
    priority    TEXT NOT NULL DEFAULT 'medium',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner_id);
"""

_ID_REF = re.compile(r"^[0-9a-f-]{4,36}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    # ------------------------------------------------------------------ infra
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Nowe polaczenie na kazda operacje - bezpieczne dla watkow
        (NLP/IO bywa uruchamiane przez asyncio.to_thread)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # bez tego SQLite ignoruje klucze obce
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # --------------------------------------------------------------- pomocnicze
    @staticmethod
    def _resolve_id(conn: sqlite3.Connection, owner_id: int, ref: str) -> str:
        """Zamienia pelne UUID lub jego prefiks (min. 4 znaki) na pelne ID."""
        ref = ref.strip().lower()
        if not _ID_REF.match(ref):
            raise InvalidInputError(f"'{ref}' nie wyglada jak ID zadania (UUID lub jego poczatek).")
        rows = conn.execute(
            "SELECT id FROM tasks WHERE owner_id = ? AND id LIKE ?", (owner_id, ref + "%")
        ).fetchall()
        if not rows:
            raise TaskNotFoundError(f"Nie znaleziono zadania o ID '{ref}'.")
        if len(rows) > 1:
            raise AmbiguousTaskIdError(f"Prefiks '{ref}' pasuje do wielu zadan - podaj dluzszy.")
        return rows[0]["id"]

    @staticmethod
    def _to_task(row: sqlite3.Row) -> Task:
        return Task(**dict(row))

    # -------------------------------------------------------------------- CRUD
    def create(
        self,
        owner_id: int,
        description: str,
        parent_ref: Optional[str] = None,
        priority: str = "medium",
    ) -> Task:
        if priority not in PRIORITIES:
            raise InvalidInputError(f"Priorytet musi byc jednym z: {', '.join(PRIORITIES)}")
        now = _now()
        with self._connect() as conn:
            parent_id = None
            if parent_ref:
                try:
                    parent_id = self._resolve_id(conn, owner_id, parent_ref)
                except (TaskNotFoundError, AmbiguousTaskIdError) as exc:
                    raise InvalidRelationError(f"Zadanie nadrzedne: {exc}") from exc
            task = Task(
                id=str(uuid.uuid4()),
                owner_id=owner_id,
                description=description,
                relation=parent_id,
                priority=priority,
                created_at=now,
                updated_at=now,
            )  # walidacja Pydantic dziala TUTAJ, przed zapisem
            conn.execute(
                "INSERT INTO tasks (id, owner_id, description, relation, status, priority,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task.id, owner_id, task.description, task.relation, task.status,
                 task.priority, now, now),
            )
        return task

    def list_tasks(self, owner_id: int) -> list[Task]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE owner_id = ? ORDER BY created_at, rowid", (owner_id,)
            ).fetchall()
        return [self._to_task(r) for r in rows]

    def get(self, owner_id: int, ref: str) -> Task:
        with self._connect() as conn:
            task_id = self._resolve_id(conn, owner_id, ref)
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._to_task(row)

    def update_description(self, owner_id: int, ref: str, description: str) -> Task:
        with self._connect() as conn:
            task_id = self._resolve_id(conn, owner_id, ref)
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            task = self._to_task(row)
            updated = task.model_copy(update={"description": description})
            Task.model_validate(updated.model_dump())  # ponowna walidacja nowej tresci
            conn.execute(
                "UPDATE tasks SET description = ?, updated_at = ? WHERE id = ?",
                (updated.description.strip(), _now(), task_id),
            )
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._to_task(row)

    def set_status(self, owner_id: int, ref: str, status: str) -> Task:
        if status not in STATUSES:
            raise InvalidInputError(f"Status musi byc jednym z: {', '.join(STATUSES)}")
        with self._connect() as conn:
            task_id = self._resolve_id(conn, owner_id, ref)
            conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now(), task_id),
            )
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._to_task(row)

    def delete(self, owner_id: int, ref: str) -> Task:
        with self._connect() as conn:
            task_id = self._resolve_id(conn, owner_id, ref)
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return self._to_task(row)
