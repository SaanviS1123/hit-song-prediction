
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/features/features.csv")

print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist()[:15])

print("\n========== LABEL DISTRIBUTION ==========")

print(df["label"].value_counts())
print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)

# Separate numerical features
X = df.drop(columns=["artist", "song", "label"])

print("\n========== FEATURE INFORMATION ==========")

print("Number of features:", X.shape[1])
print("Numeric columns:", X.select_dtypes(include=np.number).shape[1])
print("Non-numeric columns:", X.select_dtypes(exclude=np.number).shape[1])

# Convert feature columns to numeric for checking
X = X.apply(pd.to_numeric, errors="coerce")

print("\n========== MISSING VALUES ==========")

print("Total missing values:", X.isnull().sum().sum())
print("Columns containing missing values:", 
      (X.isnull().sum() > 0).sum())

print("\n========== INFINITE VALUES ==========")

print("Infinite values:", np.isinf(X.to_numpy()).sum())

print("\n========== DUPLICATES ==========")

print("Duplicate rows:", df.duplicated().sum())
print("Duplicate songs:", df["song"].duplicated().sum())

print("\n========== FEATURE STATISTICS ==========")

print(X.describe().T.head(10))

print("\n========== FEATURE VARIANCE ==========")

print("Zero variance features:",
      (X.var() == 0).sum())

print("Total zero variance features:",
      X.var().isna().sum())

print("\n========== FINAL CHECK ==========")

print("Dataset inspection completed.")
