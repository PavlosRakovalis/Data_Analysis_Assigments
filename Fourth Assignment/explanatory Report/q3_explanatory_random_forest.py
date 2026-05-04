"""
Explanatory Report - 4th Assignment - Question 3

This script reproduces the Random Forest analysis with n_estimators=200 and
max_features=p/2, then saves all artifacts used by the explanatory LaTeX report.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


CSV_PATH = "../../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
N_ESTIMATORS = 200


def save_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1], labels=["pred red", "pred white"])
    ax.set_yticks([0, 1], labels=["true red", "true white"])
    ax.set_title("Confusion Matrix - Random Forest")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("q3_confusion_matrix.png", dpi=180)
    plt.close()


def save_random_forest_concept_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    boxes = [
        (0.04, 0.62, 0.16, 0.18, "Training set"),
        (0.27, 0.78, 0.17, 0.13, "Bootstrap\nsample 1"),
        (0.27, 0.58, 0.17, 0.13, "Bootstrap\nsample 2"),
        (0.27, 0.38, 0.17, 0.13, "Bootstrap\nsample ..."),
        (0.27, 0.18, 0.17, 0.13, "Bootstrap\nsample 200"),
        (0.52, 0.78, 0.19, 0.13, "Tree 1\nrandom m features"),
        (0.52, 0.58, 0.19, 0.13, "Tree 2\nrandom m features"),
        (0.52, 0.38, 0.19, 0.13, "Tree ...\nrandom m features"),
        (0.52, 0.18, 0.19, 0.13, "Tree 200\nrandom m features"),
        (0.81, 0.50, 0.15, 0.18, "Majority\nvote"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor="#eeeaf8", edgecolor="#5a3f9b", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)

    arrowprops = dict(arrowstyle="->", linewidth=1.4, color="#5a3f9b")
    for y in [0.845, 0.645, 0.445, 0.245]:
        ax.annotate("", xy=(0.27, y), xytext=(0.20, 0.71), arrowprops=arrowprops)
    for y in [0.845, 0.645, 0.445, 0.245]:
        ax.annotate("", xy=(0.52, y), xytext=(0.44, y), arrowprops=arrowprops)
    for y in [0.845, 0.645, 0.445, 0.245]:
        ax.annotate("", xy=(0.81, 0.59), xytext=(0.71, y), arrowprops=arrowprops)

    ax.text(
        0.5,
        0.055,
        "Random Forest = Bagging of trees + random subset of candidate features at each split (m = p/2).",
        ha="center",
        va="center",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("q3_random_forest_concept.png", dpi=180)
    plt.close()


def save_feature_importance_plot(feature_importances):
    top = feature_importances.sort_values("importance", ascending=True).tail(8)

    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance"], color="#6f54b5")
    plt.xlabel("Feature importance")
    plt.title("Most important predictors in the Random Forest model")
    plt.tight_layout()
    plt.savefig("q3_random_forest_feature_importance.png", dpi=180)
    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)
    features = [col for col in df.columns if col not in ("quality", "wine_type")]
    p = len(features)
    max_features = p // 2

    X = df[features]
    y = (df["wine_type"] == "white").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=max_features,
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
                "model": "Random Forest",
                "n_estimators": N_ESTIMATORS,
                "p": p,
                "p_over_2": p / 2,
                "max_features": max_features,
                "random_state": RANDOM_STATE,
                "n_total": len(df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "test_accuracy": accuracy,
                "test_error": test_error,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        ]
    )
    results.to_csv("q3_explanatory_results.csv", index=False)

    comparison = pd.DataFrame(
        [
            {"model": "Decision Tree (max_depth=2)", "test_accuracy": 0.933920704845815, "test_error": 0.06607929515418498},
            {"model": "Bagging (200 estimators)", "test_accuracy": 0.9660163624921334, "test_error": 0.03398363750786659},
            {"model": "Random Forest (200 estimators, m=5)", "test_accuracy": accuracy, "test_error": test_error},
        ]
    )
    comparison.to_csv("q3_comparison_q1_q2_q3.csv", index=False)

    feature_importances = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    )
    feature_importances.sort_values("importance", ascending=False).to_csv(
        "q3_random_forest_feature_importance.csv",
        index=False,
    )

    save_confusion_matrix(cm)
    save_random_forest_concept_diagram()
    save_feature_importance_plot(feature_importances)

    print(results.to_string(index=False))
    print("\nSaved explanatory artifacts for Question 3.")


if __name__ == "__main__":
    main()
