
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.ml.data_loader import load_dataset


def evaluate_models():

    print("\n========== MODEL EVALUATION ==========")

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    # Load dataset
    X, y, df = load_dataset()

    # Remove constant features
    X = X.loc[:, X.nunique() > 1]

    # Same train-test split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=3
        )
    }

    results = []

    for name, model in models.items():

        print("\nEvaluating:", name)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        # Plot confusion matrix
        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Hit", "Hit"],
            yticklabels=["Non-Hit", "Hit"]
        )

        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        filename = name.lower().replace(" ", "_")

        plt.tight_layout()

        plt.savefig(
            f"results/confusion_matrices/{filename}.png"
        )

        plt.close()

        results.append({
            "Model": name,
            "True Negative": cm[0, 0],
            "False Positive": cm[0, 1],
            "False Negative": cm[1, 0],
            "True Positive": cm[1, 1]
        })

    # Save confusion matrix values
    cm_df = pd.DataFrame(results)

    cm_df.to_csv(
        "results/confusion_matrices/confusion_matrix_values.csv",
        index=False
    )

    # Model comparison chart
    comparison = pd.read_csv(
        "results/model_comparison.csv"
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    comparison.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title("Machine Learning Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Machine Learning Models")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        "results/plots/model_comparison.png"
    )

    plt.close()

    print("\nEvaluation completed!")
    print("Confusion matrices saved.")
    print("Model comparison graph saved.")


if __name__ == "__main__":
    evaluate_models()
