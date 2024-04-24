import re

def clean_lyrics(text):
    text = re.sub(r"<br>", " ", text)  # replacing HTML line breaks with space
    text = re.sub(r"[^\w\s]", "", text)  # removing punctuation
    return text.lower()

