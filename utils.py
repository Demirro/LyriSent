import re

# Regex for common non-lyric prefixes and metadata lines.
NON_LYRIC_START_PATTERNS = re.compile(
    r"^\s*\d+\s*Contributors|"  # Starts with numeric contributor count.
    r"^\s*\d+\s*Contributor|"
    r"^\s*Translations|"      # Starts with "Translations".
    r"LyricsTaken from|"      # Contains "LyricsTaken from".
    r"Lyrics"
    r"Read More",             # Contains "Read More".
    re.IGNORECASE             # Case-insensitive matching.
)


def clean_lyrics(text):
    """Clean lyrics text and normalize spacing/casing."""
    if not text:
        return ""

    # Replace HTML line breaks.
    text = text.replace("<br>", "\n")

    # Split text into lines.
    lines = text.splitlines()
    processed_lines = []
    lyrics_started = False  # Tracks when likely lyric lines begin.

    # Remove headers and detect lyric start.
    for line in lines:
        # Trim leading/trailing whitespace.
        stripped_line = line.strip()

        # Skip section headers like [Verse], [Chorus], etc.
        if re.search(r"\[.*?\]", stripped_line):
            continue

        # Skip non-lyric lines until likely lyrics begin.
        if not lyrics_started:
            if not stripped_line:
                continue
            # Skip common non-lyric metadata lines.
            if NON_LYRIC_START_PATTERNS.search(stripped_line):
                 continue

            # First valid line is treated as lyric start.
            lyrics_started = True
            processed_lines.append(stripped_line)
        else:
            # Keep remaining lines as part of lyrics content.
            processed_lines.append(stripped_line)


    # Join remaining lines.
    intermediate_text = "\n".join(processed_lines)

    # Remove punctuation and lowercase text.
    processed_text = re.sub(r"[^\w\s\n]", "", intermediate_text)  # Keep newlines at this step.
    processed_text = processed_text.lower()

    # Collapse whitespace and trim.
    processed_text = re.sub(r"\s+", " ", processed_text).strip()


    return processed_text

