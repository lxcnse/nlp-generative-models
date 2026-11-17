import pytest

from src.errors import (
    AmbiguousTaskIdError, InvalidInputError, InvalidRelationError, TaskNotFoundError,
)
from src.storage.repository import TaskRepository


@pytest.fixture
def repo(tmp_path):
    r = TaskRepository(tmp_path / "t.db")
    r.init_schema()
    return r


def test_create_and_list(repo):
    t = repo.create(1, "Kupic mleko")
    assert len(t.id) == 36 and t.status == "todo"
    assert [x.id for x in repo.list_tasks(1)] == [t.id]


def test_tasks_are_isolated_per_owner(repo):
    t = repo.create(1, "Moje")
    assert repo.list_tasks(2) == []
    with pytest.raises(TaskNotFoundError):
        repo.get(2, t.id)


def test_prefix_lookup_update_delete(repo):
    t = repo.create(1, "Stare")
    assert repo.get(1, t.id[:8]).id == t.id
    assert repo.update_description(1, t.id[:8], "Nowe").description == "Nowe"
    repo.delete(1, t.id)
    assert repo.list_tasks(1) == []


def test_relation_and_set_null_on_parent_delete(repo):
    parent = repo.create(1, "Rodzic")
    child = repo.create(1, "Dziecko", parent_ref=parent.id[:8])
    assert child.relation == parent.id
    repo.delete(1, parent.id)
    assert repo.get(1, child.id).relation is None


def test_invalid_parent(repo):
    with pytest.raises(InvalidRelationError):
        repo.create(1, "X", parent_ref="deadbeef")


def test_validation_errors(repo):
    t = repo.create(1, "Ok")
    with pytest.raises(InvalidInputError):
        repo.get(1, "zzzz")
    with pytest.raises(InvalidInputError):
        repo.set_status(1, t.id, "nieznany")
    with pytest.raises(InvalidInputError):
        repo.create(1, "X", priority="ultra")
    with pytest.raises(ValueError):
        repo.create(1, "   ")


def test_ambiguous_prefix(repo):
    now = "2026-01-01T00:00:00+00:00"
    with repo._connect() as conn:       # wstawiamy dwa ID o wspolnym prefiksie
        for uid in ("aaaa0000-0000-4000-8000-000000000001",
                    "aaaa0000-0000-4000-8000-000000000002"):
            conn.execute(
                "INSERT INTO tasks (id, owner_id, description, created_at, updated_at)"
                " VALUES (?, 1, 'x', ?, ?)", (uid, now, now))
    with pytest.raises(AmbiguousTaskIdError):
        repo.get(1, "aaaa0000")
    assert repo.get(1, "aaaa0000-0000-4000-8000-000000000002").description == "x"
