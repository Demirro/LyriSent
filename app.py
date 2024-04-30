import csv
import re

import streamlit as st

import nb_class
from GPT.OpenAI import check_sentiment_openai
from genius import fetch_lyrics
from utils import clean_lyrics

# Example new lyrics
new_lyrics = "Now there's too many people that I have done wrong And that I owe my thanks to for sticking along with me Along with me, oh, oh"

# Preprocess the lyrics
preprocessed_lyrics = nb_class.preprocess_lyrics(new_lyrics)

# Predict emotions
emotion_predictions = nb_class.predict_emotions(preprocessed_lyrics)

# Print the emotion predictions
print(emotion_predictions)

csv_file_path = 'data/gpt/songs_data.csv'
try:
    with open(csv_file_path, encoding='utf8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        existing_csv_data = list(reader)
except FileNotFoundError:
    existing_csv_data = []

checked_track = dict()
def is_track_checked(track_to_check):
    global checked_track
    if existing_csv_data:
        for track in existing_csv_data:
            if track and track['track'].casefold() == track_to_check.casefold():
                print(track['track'] + ' was already checked. It was skipped')
                checked_track = track
                return True
def has_gpt_response(track_to_check):
    for track in existing_csv_data:
        if track['track'].casefold() == track_to_check.casefold():
            if track['gpt_response'] != '':
                return True
def _input(message, input_type=str):
    while True:
        try:
            return input_type(input(message))
        except:pass
def main():
    # List of songs (artist, track)
    songs_list = [
    ]
    prefab_check = input('Do you want to use the prefab list of songs (10 songs)? Yes (y/Y) or No (n/N)\n')
    if prefab_check.casefold() in {'yes','y'}:
        songs_list = [
            ('Childish Gambino','This is America'),
            # ('brakence','deepfacke'),
            # ('Peter Fox','Haus am See'),
            # ('Feu! Chatterton', "J'ai tout mon temps"),
            # ('Bruno Mars', 'Treasure'),
            # ('Ed Sheeran', 'Shape Of You'),
            # ('The Japanese House', 'Saw You In A Dream'),
            # ('Tom Misch', 'Disco Yes'),
            # ('Radiohead', 'Creep'),
            # ('Jacob Collier', 'Hideaway'),
        ]
    elif prefab_check.casefold() in {'no','n'}:
        number_of_songs = _input('How many songs do you want to check: ', int)
        for i in range(number_of_songs):
            print('Song no. ' + str(i+1))
            track_name = input('Track Name: ')
            artist = input('Artist: ')
            songs_list.append((artist, track_name))
            print('\n')
        print(songs_list)
    else:
        print('Yes (y/Y) or No (n/N)')
    # Fetch lyrics for each song
    data = []
    song_count = 0
    for artist, track in songs_list:
        if not is_track_checked(track):
            result = fetch_lyrics(artist, track)
            if result:
                lyrics = result['lyrics'] = clean_lyrics(result['lyrics'])
                if not has_gpt_response(track):
                    result['gpt_response'] = check_sentiment_openai(result)
                    result['gpt_response'] = re.sub()
                    print(result['gpt_response'])
                new_row = [result['gpt_response']]
                print(new_row)
                existing_csv_data.extend(new_row)
                print(existing_csv_data)
                song_count += 1

    # # Save the data as CSV
    # csv_file_path = 'Data/Lyrics/songs_data.csv'
    # with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    #     fieldnames = ['artist', 'track', 'lyrics', 'gpt_response']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(data)

    try:
        with open(csv_file_path, mode='w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Song', 'Artist', 'Lyrics', 'Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
            writer.writerow(header)
            writer.writerow(existing_csv_data)
    except Exception as e:
        print(f"An error occurred while writing to the CSV file: {e}")

    print(f"Data has been saved to {csv_file_path}")

if __name__ == "__main__":
    main()