import requests
import csv
from datetime import datetime
from collections import Counter

BASE_URL = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/date/"

dates_url = (
    "https://raw.githubusercontent.com/"
    "mhollingshead/billboard-hot-100/main/valid_dates.json"
)

# Get all valid Billboard chart dates
response = requests.get(dates_url)
dates = response.json()

# Find the last available chart date in each quarter
quarter_end_dates = {}

for date in dates:
    d = datetime.strptime(date, "%Y-%m-%d")

    if 2006 <= d.year <= 2020:
        quarter = (d.month - 1) // 3 + 1
        key = (d.year, quarter)

        if key not in quarter_end_dates:
            quarter_end_dates[key] = date
        elif date > quarter_end_dates[key]:
            quarter_end_dates[key] = date

quarter_end_dates = sorted(quarter_end_dates.values())

print("Quarter-end charts:", len(quarter_end_dates))


# Collect all songs appearing on these charts
all_songs = []

for date in quarter_end_dates:

    url = BASE_URL + date + ".json"

    response = requests.get(url)

    if response.status_code != 200:
        print("Could not retrieve:", date)
        continue

    chart = response.json()

    for song in chart["data"]:

        all_songs.append({
            "artist": song["artist"],
            "song": song["song"]
        })


# Remove duplicate artist-song combinations
unique_songs = set()

for song in all_songs:
    unique_songs.add(
        (song["artist"], song["song"])
    )


# Count songs for each artist
artist_counts = Counter(
    artist for artist, song in unique_songs
)

# Select top 7 artists
top_artists = [
    artist
    for artist, count in artist_counts.most_common(7)
]

print("\nTop 7 artists:")

for artist in top_artists:
    print(artist)


# Keep only songs by the top 7 artists
popular_songs = []

for artist, song in unique_songs:

    if artist in top_artists:

        popular_songs.append({
            "artist": artist,
            "song": song,
            "label": 1
        })


# Sort for easier viewing
popular_songs.sort(
    key=lambda x: (x["artist"], x["song"])
)


# Save dataset
output_file = "data/songs.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["artist", "song", "label"]
    )

    writer.writeheader()

    writer.writerows(popular_songs)


print("\nPopular songs saved to:", output_file)
print("Number of popular songs:", len(popular_songs))