import os
import pandas as pd
from pychorus import find_and_output_chorus
from extract_features import extract_features


SONG_FILE = "data/final_songs.csv"
RAW_DIR = "data/raw"
CHORUS_DIR = "data/chorus"
FEATURE_FILE = "data/features/features.csv"


os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CHORUS_DIR, exist_ok=True)
os.makedirs("data/features", exist_ok=True)


# Load song list
songs = pd.read_csv(SONG_FILE)
results = []


for index, row in songs.iterrows():

    artist = row["artist"]
    song = row["song"]
    label = row["label"]

    print(f"\n[{index + 1}/{len(songs)}] {artist} - {song}")

    # Find matching audio file
    audio_file = None

    for filename in os.listdir(RAW_DIR):

        if not filename.lower().endswith(".wav"):
            continue

        filename_clean = filename.lower().replace(".wav", "")
        song_clean = song.lower()

        if song_clean in filename_clean:
            audio_file = os.path.join(RAW_DIR, filename)
            break

    if audio_file is None:
        print("Audio file not found - skipping.")
        continue

    # Chorus file
    chorus_file = os.path.join(
        CHORUS_DIR,
        f"{artist} - {song}_chorus.wav"
    )

    try:

        # Find chorus
        chorus_start = find_and_output_chorus(
            audio_file,
            chorus_file
        )

        if chorus_start is None:
            print("No chorus found - skipping.")
            continue

        # Extract 518 features
        feature_vector = extract_features(
            chorus_file
        )

        if len(feature_vector) != 518:
            print(
                "Unexpected feature count:",
                len(feature_vector)
            )
            continue

        # Store metadata + features
        results.append(
            [artist, song, label] + feature_vector
        )

        print("Processed successfully.")

    except Exception as e:
        print("Error:", e)


# Create column names
feature_columns = [
    f"feature_{i}"
    for i in range(518)
]

columns = [
    "artist",
    "song",
    "label"
] + feature_columns


# Save results
df = pd.DataFrame(
    results,
    columns=columns
)

df.to_csv(
    FEATURE_FILE,
    index=False
)


print("\n================================")
print("Processing complete!")
print("Songs processed:", len(df))
print("Features per song:", 518)
print("Saved to:", FEATURE_FILE)
print("================================")