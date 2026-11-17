"""teksty wyswietlane uzytkownikowi (HTML - parse_mode Telegrama)"""

HELP_TEXT = """<b>Bot TODO + NLP + klasyfikacja</b>

<b>Zarzadzanie zadaniami</b>
/tasks - lista zadan
/add {opis} [--parent {id}] [--prio low|medium|high] - dodaj zadanie
/update {id} {nowy opis} - zmien opis
/delete {id} - usun zadanie
/status {id} todo|in_progress|done - zmien status

<b>NLP</b>
/task {id} {operacja} - wykonaj operacje na opisie zadania
/ops - lista operacji NLP

<b>Klasyfikacja</b>
/prepare_data_for_classification [todo|agnews|20newsgroups] [--seed] - etykiety / wczytanie zbioru
/label {id} {klasa} - recznie nadaj (nadpisz) klase zadania
/classifier cleaning "tekst" bow|tfidf|embedding {metoda} - klasyfikacja tekstu
/evaluate bow|tfidf|embedding {metoda} - macierz pomylek i metryki
/validate bow|tfidf|embedding {metoda} - test na wlasnych zdaniach spoza zbioru
/methods - wybrane metody i parametry reprezentacji

ID mozesz podac w calosci (UUID) albo jako pierwsze 4+ znakow."""

STATUS_ICONS = {"todo": "⬜", "in_progress": "🔄", "done": "✅"}
