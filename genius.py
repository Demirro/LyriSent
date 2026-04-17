from lyricsgenius import Genius
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GENIUS_TOKEN")


def _genius_client(genius_token: str | None) -> Genius | None:
    tok = (genius_token or "").strip() or (token or "").strip()
    if not tok:
        return None
    g = Genius(tok, timeout=5, retries=3, response_format="plain,html")
    g.response_format = "html"
    return g


def fetch_lyrics(artist_name, track_name, genius_token: str | None = None):
    g = _genius_client(genius_token)
    if g is None:
        print("Genius: no token (set GENIUS_TOKEN or pass genius_token).")
        return None
    try:
        song = g.search_song(track_name, artist_name)
        if song is not None:
            return {
                "artist": artist_name,
                "track": track_name,
                "lyrics": song.lyrics,
            }
        print(f"Lyrics not found for {artist_name} - {track_name}")
        return None
    except Exception as e:
        print(f"Error fetching lyrics for {artist_name} - {track_name}: {e}")
        return None
