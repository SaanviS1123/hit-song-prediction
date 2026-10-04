
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.ml.data_loader import load_dataset


def preprocess_data(X):

    print("\n========== PREPROCESSING ==========")

    # STEP 1: Remove constant features

    selector = VarianceThreshold(threshold=0)

    X_selected = selector.fit_transform(X)

    print("Original features:", X.shape[1])
    print("Features after removing constants:", X_selected.shape[1])

    # STEP 2: Standardize features

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_selected)

    print("\nStandardization completed.")

    print("Mean of first feature:", np.mean(X_scaled[:, 0]))
    print("Std of first feature:", np.std(X_scaled[:, 0]))

    # STEP 3: Apply PCA

    pca = PCA(n_components=0.95)

    X_pca = pca.fit_transform(X_scaled)

    print("\nPCA completed.")

    print("Original dimensions:", X_scaled.shape[1])
    print("Reduced dimensions:", X_pca.shape[1])

    print(
        "Explained variance:",
        round(np.sum(pca.explained_variance_ratio_) * 100, 2),
        "%"
    )

    return X_pca, selector, scaler, pca


if __name__ == "__main__":

    X, y, df = load_dataset()

    X_pca, selector, scaler, pca = preprocess_data(X)

    print("\nFinal PCA matrix shape:", X_pca.shape)

    print("\nPreprocessing completed successfully!")
