import re

# Kompiliere ein RegEx-Muster für häufige nicht-lyrische Muster am Anfang einer Datei
# Dieses Muster sucht nach Zeilen, die z.B. mit Ziffern (Contributors),
# oder spezifischen Phrasen wie 'LyricsTaken from', 'Read More', 'Translations' beginnen oder enthalten.
# Wir machen es optional am Anfang der Zeile (^) oder irgendwo in der Zeile.
NON_LYRIC_START_PATTERNS = re.compile(
    r"^\s*\d+\s*Contributors|"  # Beginnt mit Ziffern und "Contributors" (mit optionalen Leerzeichen)
    r"^\s*\d+\s*Contributor|"
    r"^\s*Translations|"      # Beginnt mit "Translations"
    r"LyricsTaken from|"      # Enthält "LyricsTaken from"
    r"Lyrics"
    r"Read More",             # Enthält "Read More"
    re.IGNORECASE              # Ignoriert Groß-/Kleinschreibung
)


def clean_lyrics(text):
    """
    Bereinigt Songtexte: Entfernt HTML-Zeilenumbrüche, Abschnitt-Header ([...]),
    häufige Einleitungsbeschreibungen/Metadaten, Satzzeichen und konvertiert in Kleinbuchstaben.
    Behält Zeilenumbrüche zwischen Lyric-Zeilen bei, bis zur finalen Glättung.
    """
    if not text:
        return ""

    # 1. HTML-Zeilenumbrüche ersetzen
    text = text.replace("<br>", "\n")

    # 2. Text in Zeilen aufteilen
    lines = text.splitlines()
    processed_lines = []
    lyrics_started = False # Flag, um zu markieren, wann wir wahrscheinlich bei den echten Lyrics sind

    # 3. Zeilen verarbeiten: Entferne Header und identifiziere Beginn der Lyrics
    for line in lines:
        # Entferne führende/nachfolgende Leerzeichen
        stripped_line = line.strip()

        # Ignoriere Zeilen mit Abschnitt-Headern ([...])
        if re.search(r"\[.*?\]", stripped_line):
            continue

        # Ignoriere leere Zeilen am Anfang, bis wir etwas finden, das wie ein Lyric aussieht
        if not lyrics_started:
            # Ignoriere leere Zeilen
            if not stripped_line:
                continue
            # Ignoriere Zeilen, die typische nicht-lyrische Muster am Anfang enthalten
            if NON_LYRIC_START_PATTERNS.search(stripped_line):
                 # print(f"Skipping potential non-lyric line: {stripped_line}") # Debug-Ausgabe
                 continue

            # Wenn wir hier ankommen, haben wir eine nicht-leere Zeile gefunden,
            # die keinen typischen Header oder non-lyric Startpattern enthält.
            # Wir nehmen an, dies ist der Beginn der Lyrics oder nahe dran.
            lyrics_started = True
            processed_lines.append(stripped_line) # Füge die erste Lyric-Zeile hinzu
        else:
            # Sobald lyrics_started True ist, fügen wir alle nachfolgenden Zeilen hinzu
            # (auch leere Zeilen, um die Strophenstruktur zu erhalten, bis zur finalen Glättung)
            processed_lines.append(stripped_line)


    # 4. Verbleibende Zeilen zu einem String zusammenfügen
    # Wir verwenden "\n" um die ursprünglichen Zeilenumbrüche beizubehalten
    intermediate_text = "\n".join(processed_lines)

    # 5. Satzzeichen entfernen und in Kleinbuchstaben umwandeln
    # Dies sollte auf den gesamten Text angewendet werden, nachdem die Struktur bereinigt ist
    processed_text = re.sub(r"[^\w\s\n]", "", intermediate_text) # Behalte \n hier noch
    processed_text = processed_text.lower()

    # 6. Mehrere Leerzeichen, Newlines und Satzzeichen-Reste durch ein einzelnes Leerzeichen ersetzen und trimmen
    # Das glättet den Text zu einem einzigen Block mit Wörtern, getrennt durch einzelne Leerzeichen.
    processed_text = re.sub(r"\s+", " ", processed_text).strip()


    return processed_text

