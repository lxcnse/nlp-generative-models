"""teksty wyswietlane uzytkownikowi (HTML - parse_mode Telegrama)"""

HELP_TEXT = """<b>Bot TODO + NLP</b>

<b>Zarzadzanie zadaniami</b>
/tasks - lista zadan
/add {opis} [--parent {id}] [--prio low|medium|high] - dodaj zadanie
/update {id} {nowy opis} - zmien opis
/delete {id} - usun zadanie
/status {id} todo|in_progress|done - zmien status

<b>NLP</b>
/task {id} {operacja} - wykonaj operacje na opisie zadania
/ops - lista operacji NLP

ID mozesz podac w calosci (UUID) albo jako pierwsze 4+ znakow."""

STATUS_ICONS = {"todo": "⬜", "in_progress": "🔄", "done": "✅"}
