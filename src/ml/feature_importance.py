import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.ml.data_loader import load_dataset


def analyze_feature_importance():

    print("\n========== FEATURE IMPORTANCE ANALYSIS ==========")

    # Create output directories
    os.makedirs("results/plots", exist_ok=True)

    # Load dataset
    X, y, df = load_dataset()

    # Use the same train-test split as model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Load trained Random Forest pipeline
    model_path = "models/random_forest.pkl"

    if not os.path.exists(model_path):
        print("Random Forest model not found. Train models first.")
        return

    pipeline = joblib.load(model_path)

    print("\nCalculating permutation importance...")

    # Calculate permutation importance on held-out test data
    importance = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    # Store feature importance results
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance.importances_mean,
        "Importance Std": importance.importances_std
    })

    # Sort by importance
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Save all feature importance values
    importance_df.to_csv(
        "results/feature_importance.csv",
        index=False
    )

    # Display top 20 features
    top_features = importance_df.head(20)

    print("\nTop 20 Important Audio Features:")
    print(top_features.to_string(index=False))

    # Plot top 20 features
    plt.figure(figsize=(12, 8))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Mean Decrease in ROC-AUC")
    plt.ylabel("Audio Features")
    plt.title("Top 20 Audio Features - Permutation Importance")

    plt.tight_layout()

    plt.savefig(
        "results/plots/feature_importance.png"
    )

    plt.close()

    print("\nFeature importance analysis completed!")
    print("Results saved to results/feature_importance.csv")
    print("Graph saved to results/plots/feature_importance.png")


if __name__ == "__main__":
    analyze_feature_importance()
