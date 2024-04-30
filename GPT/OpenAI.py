from os import environ

from openai import OpenAI

client = OpenAI()
from openai import OpenAI

client = OpenAI()


def check_sentiment_openai(song):
    lyrics = song['lyrics']
    # Example OpenAI Python library request
    MODEL = "gpt-4"
    response = client.chat.completions.create(model=MODEL,
                                              messages=[
                                                  {"role": "system",
                                                   "content": "You are a tone analyzer for lyrics. You systematically analyze the mood of any given song lyric according to Plutchiks 8 core emotions of Anger, Disgust, Fear, Joy, Sadness and Surprise. Classify each song in it's entirety. Give the results in a tabular form using csv with the table header: Song,Artists,Lyrics,Joy,Trust,Fear,Surprise,Sadness,Disgust,Anger,Anticipation. Limit the output of the Lyrics column to only 50 symbols and add ... . If a song displays any of the emotions it should get a 1 in the given column, if not a 0. Think about annotating as if you were the result or agreement of multiple hundreds of annotators. Don't give any other output than the pure csv format. You are just a simple tone analyzer without any other natural text language output."},
                                                  {"role": "user",
                                                   "content": "Analyze the following lyrics: \n" + lyrics},
                                              ],
                                              temperature=0)
    return response.choices[0].message.content
