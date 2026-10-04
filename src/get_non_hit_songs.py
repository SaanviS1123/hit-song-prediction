import requests
import csv
import time

BASE_URL = "https://musicbrainz.org/ws/2"

ARTISTS = [
    "Taylor Swift",
    "Drake",
    "Jason Aldean",
    "Luke Bryan",
    "Carrie Underwood",
    "Keith Urban",
    "Kenny Chesney"
]

HEADERS = {
    "User-Agent": "HitSongPrediction/1.0 (student-project)"
}


def get_request(url, params):

    for attempt in range(5):

        response = requests.get(
            url,
            params=params,
            headers=HEADERS
        )

        if response.status_code == 200:
            return response

        if response.status_code == 503:
            print("Server busy. Waiting before retry...")
            time.sleep(5 * (attempt + 1))
            continue

        response.raise_for_status()

    print("Request failed after several attempts.")
    return None


def search_artist(artist_name):

    url = f"{BASE_URL}/artist"

    params = {
        "query": f'artist:"{artist_name}"',
        "fmt": "json",
        "limit": 5
    }

    response = get_request(url, params)

    if response is None:
        return None

    data = response.json()

    if not data["artists"]:
        return None

    return data["artists"][0]["id"]


def get_releases(artist_id):

    releases = []
    offset = 0

    while True:

        url = f"{BASE_URL}/release"

        params = {
            "artist": artist_id,
            "status": "official",
            "type": "album",
            "limit": 100,
            "offset": offset,
            "inc": "recordings",
            "fmt": "json"
        }

        response = get_request(url, params)

        if response is None:
            break

        data = response.json()

        batch = data["releases"]

        if not batch:
            break

        releases.extend(batch)

        offset += len(batch)

        if offset >= data["release-count"]:
            break

        time.sleep(2)

    return releases


# --------------------------------------------------
# Load popular songs
# --------------------------------------------------

popular_songs = set()

with open(
    "data/songs.csv",
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        popular_songs.add(
            (
                row["artist"].lower().strip(),
                row["song"].lower().strip()
            )
        )

print("Popular songs loaded:", len(popular_songs))


# --------------------------------------------------
# Collect non-hit songs
# --------------------------------------------------

non_hit_songs = set()


for artist in ARTISTS:

    print("\nProcessing:", artist)

    artist_id = search_artist(artist)

    if artist_id is None:
        print("Could not find artist. Skipping.")
        continue

    print("Artist ID:", artist_id)

    time.sleep(2)

    releases = get_releases(artist_id)

    print("Releases found:", len(releases))

    for release in releases:

        # Ignore releases without a proper title
        if not release.get("title"):
            continue

        for medium in release.get("media", []):

            for track in medium.get("tracks", []):

                recording = track.get("recording", {})

                song_title = recording.get("title")

                if not song_title:
                    continue

                key = (
                    artist.lower().strip(),
                    song_title.lower().strip()
                )

                # Only keep songs that aren't Billboard hits
                if key not in popular_songs:

                    non_hit_songs.add(
                        (artist, song_title)
                    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

output_file = "data/non_hit_candidates.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "artist",
        "song",
        "label"
    ])

    for artist, song in sorted(non_hit_songs):

        writer.writerow([
            artist,
            song,
            0
        ])


print("\n--------------------------------")
print("Non-hit candidates:", len(non_hit_songs))
print("Saved to:", output_file)
print("--------------------------------")