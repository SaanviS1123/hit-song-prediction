import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split

from src.ml.data_loader import load_dataset


def evaluate_models():

    print("\n========== MODEL EVALUATION ==========")

    # Create output directories
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    # Load original dataset
    X, y, df = load_dataset()

    # Use the same train-test split as model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Paths to saved trained models
    model_files = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "SVM": "models/svm.pkl",
        "Random Forest": "models/random_forest.pkl",
        "KNN": "models/knn.pkl"
    }

    results = []

    for name, model_path in model_files.items():

        print(f"\nEvaluating: {name}")

        # Check whether the trained model exists
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        # Load the exact trained pipeline
        pipeline = joblib.load(model_path)

        # Predict using the held-out test data
        predictions = pipeline.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

        # Generate confusion matrix
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

        # Store results
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "True Negative": cm[0, 0],
            "False Positive": cm[0, 1],
            "False Negative": cm[1, 0],
            "True Positive": cm[1, 1]
        })

    # Save evaluation results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/evaluation_results.csv",
        index=False
    )

    # Plot model comparison
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    results_df.set_index("Model")[metrics].plot(
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
    print("Evaluation results saved to results/evaluation_results.csv")
    print("Confusion matrices saved.")
    print("Model comparison graph saved.")


if __name__ == "__main__":
    evaluate_models()