import pytest

from src.config import xgram_size


@pytest.mark.parametrize(
    "album, expected",
    [("123456", 6), ("100000", 4), ("100001", 5), ("100002", 6), ("100003", 7),
     ("100004", 4), ("100009", 9)],
)
def test_xgram_size(album, expected):
    assert xgram_size(album) == expected


def test_xgram_requires_digit():
    with pytest.raises(ValueError):
        xgram_size("abc")
