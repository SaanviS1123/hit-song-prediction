import os
from pychorus import find_and_output_chorus
from extract_features import extract_features


def process_song(audio_file, chorus_file):

    print("Finding chorus...")

    chorus_start = find_and_output_chorus(
        audio_file,
        chorus_file
    )

    if chorus_start is None:
        print("No chorus found.")
        return None

    print("Chorus found at:", chorus_start, "seconds")

    print("Extracting features...")

    feature_vector = extract_features(chorus_file)

    print("Number of features:", len(feature_vector))

    return feature_vector


audio_file = "data/raw/Taylor Swift - Blank Space.wav"
chorus_file = "data/chorus/Taylor Swift - Blank Space_chorus.wav"

features = process_song(audio_file, chorus_file)

if features is not None:
    print("Song processed successfully!")