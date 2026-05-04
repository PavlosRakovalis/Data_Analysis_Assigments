"""
Explanatory Report - 4th Assignment - Question 1

This script reproduces the max_depth=2 decision tree analysis and saves
all artifacts used by the explanatory LaTeX report.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


CSV_PATH = "../../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
MAX_DEPTH = 2


def main():
    df = pd.read_csv(CSV_PATH)
    features = [col for col in df.columns if col not in ("quality", "wine_type")]

    X = df[features]
    y = (df["wine_type"] == "white").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_error = 1 - accuracy
    cm = confusion_matrix(y_test, y_pred)

    results = pd.DataFrame(
        [
            {
                "model": "Decision Tree",
                "max_depth": MAX_DEPTH,
                "random_state": RANDOM_STATE,
                "n_total": len(df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": len(features),
                "test_accuracy": accuracy,
                "test_error": test_error,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        ]
    )
    results.to_csv("q1_explanatory_results.csv", index=False)

    class_counts = df["wine_type"].value_counts().rename_axis("wine_type").reset_index(name="count")
    class_counts["percentage"] = class_counts["count"] / len(df)
    class_counts.to_csv("q1_class_distribution.csv", index=False)

    rules = export_text(model, feature_names=features)
    with open("q1_tree_rules.txt", "w", encoding="utf-8") as file:
        file.write(rules)

    plt.figure(figsize=(13, 6))
    plot_tree(
        model,
        feature_names=features,
        class_names=["red", "white"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=9,
    )
    plt.title("Decision Tree for Wine Type Prediction (max_depth=2)")
    plt.tight_layout()
    plt.savefig("q1_decision_tree_plot.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred red", "pred white"])
    ax.set_yticks([0, 1], labels=["true red", "true white"])
    ax.set_title("Confusion Matrix - Decision Tree (max_depth=2)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("q1_confusion_matrix.png", dpi=180)
    plt.close()

    print(results.to_string(index=False))
    print("\nSaved explanatory artifacts for Question 1.")


if __name__ == "__main__":
    main()
