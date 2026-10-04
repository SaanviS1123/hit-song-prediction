
import pandas as pd
import numpy as np


DATA_PATH = "data/features/features.csv"


def load_dataset(path=DATA_PATH):

    print("\nLoading dataset...")

    # Read CSV
    df = pd.read_csv(path)

    # Check required columns
    required_columns = ["artist", "song", "label"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Separate labels
    y = pd.to_numeric(df["label"], errors="raise")

    # Separate audio features
    X = df.drop(
        columns=["artist", "song", "label"]
    )

    # Convert features to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Check labels
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")

    # Check dataset
    if X.empty:
        raise ValueError("No feature columns found")

    if X.isnull().any().any():
        raise ValueError("Missing values detected in features")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Infinite values detected")

    print("\nDataset loaded successfully!")

    print("Number of songs:", len(X))
    print("Number of features:", X.shape[1])

    print("\nLabel distribution:")
    print(y.value_counts().sort_index())

    return X, y, df


if __name__ == "__main__":

    X, y, df = load_dataset()

    print("\nFeature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    print("\nFirst 5 feature rows:")
    print(X.head())

    print("\nData loader test completed!")
