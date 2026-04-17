# LyriSent_Bert/app.py — CLI: unseen-song test runner (NB + BERT + OpenAI)
from __future__ import annotations

from unseen_runner import process_songs


def _input(message: str, cast_type=str):
    while True:
        try:
            return cast_type(input(message))
        except ValueError:
            print("Invalid input type. Please try again.")
        except EOFError:
            print("\nInput stream closed. Exiting.")
            raise SystemExit(0)


def _collect_songs() -> list[tuple[str, str]]:
    songs_list: list[tuple[str, str]] = []
    prefab_check = input("Do you want to use the prefab list of songs (10 songs)? Yes (y/Y) or No (n/N)\n")

    if prefab_check.casefold() in {"yes", "y"}:
        songs_list = [
            ("Childish Gambino", "This is America"),
            ("brakence", "deepfacke"),
            ("Peter Fox", "Haus am See"),
            ("Feu! Chatterton", "J'ai tout mon temps"),
            ("Bruno Mars", "Treasure"),
            ("Ed Sheeran", "Shape Of You"),
            ("The Japanese House", "Saw You In A Dream"),
            ("Tom Misch", "Disco Yes"),
            ("Radiohead", "Creep"),
            ("Jacob Collier", "Hideaway"),
        ]
    elif prefab_check.casefold() in {"no", "n"}:
        number_of_songs = _input("How many songs do you want to check: ", int)
        for i in range(number_of_songs):
            print(f"Song no. {i + 1}")
            track_name = input("Track Name: ")
            artist = input("Artist: ")
            songs_list.append((artist, track_name))
            print("")
    else:
        print("Invalid input. Please enter Yes (y/Y) or No (n/N)")
        return []

    return songs_list


def main():
    print("LyriSent unseen-song test runner")
    print("This runs NB + BERT + OpenAI (all configured prompting modes) on songs not seen during training.")
    print("")

    songs = _collect_songs()
    if not songs:
        return

    rows, output_dir, _issues = process_songs(songs, use_openai=True, save=True)
    if output_dir:
        print(f"Saved predictions under: {output_dir}")
    print(f"Processed {len(rows)} songs.")


if __name__ == "__main__":
    main()
