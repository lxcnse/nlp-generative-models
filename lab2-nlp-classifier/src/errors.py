"""Wlasne wyjatki. Wszystkie dziedzicza po BotError, dzieki czemu handler
moze je jednym `except` zamienic na czytelny komunikat dla uzytkownika."""


class BotError(Exception):
    """Bazowy blad 'oczekiwany' - jego tresc mozna pokazac uzytkownikowi."""


class InvalidInputError(BotError):
    """Uzytkownik podal bledne argumenty komendy."""


class TaskNotFoundError(BotError):
    """Nie znaleziono zadania o podanym ID."""


class AmbiguousTaskIdError(BotError):
    """Podany prefiks ID pasuje do wielu zadan."""


class InvalidRelationError(BotError):
    """Bledna relacja miedzy zadaniami."""
