import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import train_test_split

from src.ml.data_loader import load_dataset


def evaluate_models():

    print("\n========== MODEL EVALUATION ==========")

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    # Load dataset
    X, y, df = load_dataset()

    # Reproduce the same test split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model_files = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "SVM": "models/svm.pkl",
        "Random Forest": "models/random_forest.pkl",
        "KNN": "models/knn.pkl"
    }

    results = []

    # ROC curve
    plt.figure(figsize=(8, 6))

    for name, model_path in model_files.items():

        print(f"\nEvaluating: {name}")

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        # Load saved trained pipeline
        pipeline = joblib.load(model_path)

        predictions = pipeline.predict(X_test)

        # Obtain prediction scores for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            scores = pipeline.decision_function(X_test)

        # Calculate metrics
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

        roc_auc = roc_auc_score(y_test, scores)

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))
        print("ROC-AUC:", round(roc_auc, 4))

        # Confusion matrix
        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        filename = name.lower().replace(" ", "_")

        plt.figure(figsize=(6, 5))

        import seaborn as sns

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
        plt.tight_layout()

        plt.savefig(
            f"results/confusion_matrices/{filename}.png"
        )

        plt.close()

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, scores)

        plt.figure(1)
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})"
        )

        # Store results
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "True Negative": cm[0, 0],
            "False Positive": cm[0, 1],
            "False Negative": cm[1, 0],
            "True Positive": cm[1, 1]
        })

    # Finish ROC curve
    plt.figure(1)
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig("results/plots/roc_curve.png")
    plt.close()

    # Save evaluation results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/evaluation_results.csv",
        index=False
    )

    # Model comparison chart
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
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

    plt.savefig("results/plots/model_comparison.png")
    plt.close()

    print("\nEvaluation completed!")
    print("ROC-AUC results saved.")
    print("ROC curve saved.")
    print("Confusion matrices saved.")
    print("Model comparison graph saved.")


if __name__ == "__main__":
    evaluate_models()