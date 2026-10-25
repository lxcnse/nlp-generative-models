"""Model zadania (Pydantic) - waliduje dane wchodzace i wychodzace z bazy."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

Status = Literal["todo", "in_progress", "done"]
Priority = Literal["low", "medium", "high"]

STATUSES: tuple[str, ...] = ("todo", "in_progress", "done")
PRIORITIES: tuple[str, ...] = ("low", "medium", "high")


class Task(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str                                   # UUID jako tekst
    owner_id: int                             # kto jest wlascicielem (Telegram user id)
    description: str                          # tresc zadania
    relation: Optional[str] = None            # id zadania nadrzednego (opcjonalnie)
    status: Status = "todo"
    priority: Priority = "medium"
    created_at: datetime
    updated_at: datetime

    @field_validator("id", "relation")
    @classmethod
    def _must_be_uuid(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return str(uuid.UUID(value))  # rzuca ValueError dla blednego UUID

    @field_validator("description")
    @classmethod
    def _description_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Opis zadania nie moze byc pusty")
        if len(value) > 1000:
            raise ValueError("Opis zadania jest za dlugi (max 1000 znakow)")
        return value
