
import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.ml.data_loader import load_dataset


def train_models():

    print("\n========== MODEL TRAINING ==========")

    # Load original dataset
    X, y, df = load_dataset()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Define models
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

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # Preprocessing inside pipeline prevents data leakage
        pipeline = Pipeline([
            ("feature_selection", VarianceThreshold(threshold=0)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("classifier", model)
        ])

        # Stratified 5-fold cross-validation
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": make_scorer(
                    precision_score,
                    zero_division=0
                ),
                "recall": make_scorer(
                    recall_score,
                    zero_division=0
                ),
                "f1": make_scorer(
                    f1_score,
                    zero_division=0
                )
            }
        )

        print("\n5-Fold Cross-Validation Results:")

        for metric in ["accuracy", "precision", "recall", "f1"]:

            scores = cv_results[f"test_{metric}"]

            print(
                metric,
                ":",
                round(scores.mean(), 4),
                "+/-",
                round(scores.std(), 4)
            )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        # Save trained model
        import joblib

        filename = name.lower().replace(" ", "_")
        joblib.dump(
            pipeline,
            f"models/{filename}.pkl"
        )

    # Save results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\nModel training completed!")
    print("Results saved to results/model_comparison.csv")
    print("Models saved in models/")


if __name__ == "__main__":
    train_models()
