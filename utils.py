import re
import string


def clean_lyrics(text):
    text = re.sub(r"<br>", " ", text)  # replacing HTML line breaks with space
    text = re.sub(r"[^\w\s]", "", text)  # removing punctuation
    return text.lower()

def clean_lyrics_genius(s):
    s = remove_contributors(s)
    s = remove_lyrics_name(s)
    s = remove_embed(s)
    s = remove_song_structure(s)
    #s = remove_punctuation(s)
    return s
def remove_punctuation(s):
    no_punc = str.maketrans('', '', string.punctuation)
    return s.translate(no_punc)

def remove_contributors(s):
    s = re.sub(r'(\d+) contributors', '', s)
    return s

def remove_lyrics_name(s):
    s = re.sub(r'.*?\ Lyrics', '', s)
    return s

def remove_embed(s):
    s = re.sub(r'—?\d+Embed', '', s)
    return s

def remove_song_structure(s):
    s = re.sub(r'\[(.*?)\]', '', s)
    return s