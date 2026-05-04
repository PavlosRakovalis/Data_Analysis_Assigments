"""
Explanatory Report - 4th Assignment - Question 2

This script reproduces the Bagging analysis with n_estimators=200 and saves
all artifacts used by the explanatory LaTeX report.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


CSV_PATH = "../../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
N_ESTIMATORS = 200


def save_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1], labels=["pred red", "pred white"])
    ax.set_yticks([0, 1], labels=["true red", "true white"])
    ax.set_title("Confusion Matrix - Bagging (200 estimators)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("q2_confusion_matrix.png", dpi=180)
    plt.close()


def save_bagging_concept_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")

    boxes = [
        (0.05, 0.62, 0.18, 0.18, "Training set"),
        (0.33, 0.78, 0.18, 0.14, "Bootstrap\nsample 1"),
        (0.33, 0.58, 0.18, 0.14, "Bootstrap\nsample 2"),
        (0.33, 0.38, 0.18, 0.14, "Bootstrap\nsample ..."),
        (0.33, 0.18, 0.18, 0.14, "Bootstrap\nsample 200"),
        (0.62, 0.78, 0.16, 0.14, "Tree 1"),
        (0.62, 0.58, 0.16, 0.14, "Tree 2"),
        (0.62, 0.38, 0.16, 0.14, "Tree ..."),
        (0.62, 0.18, 0.16, 0.14, "Tree 200"),
        (0.83, 0.50, 0.14, 0.18, "Majority\nvote"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor="#e8f4ea", edgecolor="#2f6f3e", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    arrowprops = dict(arrowstyle="->", linewidth=1.4, color="#2f6f3e")
    for y in [0.85, 0.65, 0.45, 0.25]:
        ax.annotate("", xy=(0.33, y), xytext=(0.23, 0.71), arrowprops=arrowprops)
    for y in [0.85, 0.65, 0.45, 0.25]:
        ax.annotate("", xy=(0.62, y), xytext=(0.51, y), arrowprops=arrowprops)
    for y in [0.85, 0.65, 0.45, 0.25]:
        ax.annotate("", xy=(0.83, 0.59), xytext=(0.78, y), arrowprops=arrowprops)

    ax.text(
        0.5,
        0.05,
        "Bagging: many trees are trained on different bootstrap samples, then their votes are combined.",
        ha="center",
        va="center",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("q2_bagging_concept.png", dpi=180)
    plt.close()


def save_feature_importance_plot(feature_importances):
    top = feature_importances.sort_values("importance", ascending=True).tail(8)

    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance"], color="#3d8b4f")
    plt.xlabel("Average feature importance across bagged trees")
    plt.title("Most important predictors in the Bagging model")
    plt.tight_layout()
    plt.savefig("q2_bagging_feature_importance.png", dpi=180)
    plt.close()


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

    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_error = 1 - accuracy
    cm = confusion_matrix(y_test, y_pred)

    results = pd.DataFrame(
        [
            {
                "model": "Bagging",
                "n_estimators": N_ESTIMATORS,
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
    results.to_csv("q2_explanatory_results.csv", index=False)

    q1_comparison = pd.DataFrame(
        [
            {"model": "Decision Tree (max_depth=2)", "test_accuracy": 0.933920704845815, "test_error": 0.06607929515418498},
            {"model": "Bagging (200 estimators)", "test_accuracy": accuracy, "test_error": test_error},
        ]
    )
    q1_comparison.to_csv("q2_vs_q1_comparison.csv", index=False)

    importances = np.zeros(len(features))
    for estimator, feature_idx in zip(model.estimators_, model.estimators_features_):
        importances[feature_idx] += estimator.feature_importances_
    importances /= len(model.estimators_)

    feature_importances = pd.DataFrame({"feature": features, "importance": importances})
    feature_importances.sort_values("importance", ascending=False).to_csv(
        "q2_bagging_feature_importance.csv",
        index=False,
    )

    save_confusion_matrix(cm)
    save_bagging_concept_diagram()
    save_feature_importance_plot(feature_importances)

    print(results.to_string(index=False))
    print("\nSaved explanatory artifacts for Question 2.")


if __name__ == "__main__":
    main()
