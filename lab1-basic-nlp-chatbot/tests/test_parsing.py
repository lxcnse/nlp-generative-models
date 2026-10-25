import pytest

from src.bot.parsing import parse_add_args, split_first_word
from src.errors import InvalidInputError


def test_parse_add_plain():
    a = parse_add_args("Kupić mleko")
    assert (a.description, a.parent, a.priority) == ("Kupić mleko", None, "medium")


def test_parse_add_flags():
    a = parse_add_args("Kupić mleko --parent 550e8400 --prio HIGH")
    assert (a.description, a.parent, a.priority) == ("Kupić mleko", "550e8400", "high")


def test_parse_add_errors():
    with pytest.raises(InvalidInputError):
        parse_add_args("--prio high")
    with pytest.raises(InvalidInputError):
        parse_add_args("x --prio super")


def test_split_first_word():
    assert split_first_word("abc  def ghi") == ("abc", "def ghi")
    assert split_first_word("abc") == ("abc", "")
    assert split_first_word("") == ("", "")
