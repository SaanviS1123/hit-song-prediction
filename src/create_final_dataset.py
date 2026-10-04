import pandas as pd

# Load popular songs
popular = pd.read_csv("data/songs.csv")

# Load non-hit candidates
non_hits = pd.read_csv("data/non_hit_candidates.csv")

# Select 216 non-hit songs
non_hits = non_hits.sample(
    n=216,
    random_state=42
)

# Combine both datasets
final_songs = pd.concat(
    [popular, non_hits],
    ignore_index=True
)

# Shuffle the final dataset
final_songs = final_songs.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
final_songs.to_csv(
    "data/final_songs.csv",
    index=False
)

print("Final dataset created!")
print("Total songs:", len(final_songs))
print("Popular songs:", (final_songs["label"] == 1).sum())
print("Non-hit songs:", (final_songs["label"] == 0).sum())
print("\nSaved to: data/final_songs.csv")