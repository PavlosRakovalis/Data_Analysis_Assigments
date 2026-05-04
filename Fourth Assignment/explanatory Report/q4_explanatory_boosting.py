"""
Explanatory Report - 4th Assignment - Question 4

This script reproduces the Boosting analysis with n_estimators=200,
learning_rate=0.1, and max_depth=1, then saves all artifacts used by the
unified explanatory LaTeX report.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


CSV_PATH = "../../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
N_ESTIMATORS = 200
LEARNING_RATE = 0.1
MAX_DEPTH = 1


def save_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1], labels=["pred red", "pred white"])
    ax.set_yticks([0, 1], labels=["true red", "true white"])
    ax.set_title("Confusion Matrix - Boosting")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("q4_confusion_matrix.png", dpi=180)
    plt.close()


def save_boosting_concept_diagram():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.axis("off")

    boxes = [
        (0.04, 0.56, 0.14, 0.20, "Training\nset"),
        (0.25, 0.70, 0.13, 0.16, "Weak tree 1\nmax_depth=1"),
        (0.43, 0.56, 0.13, 0.16, "Weak tree 2\nlearns errors"),
        (0.61, 0.42, 0.13, 0.16, "Weak tree ...\nlearns errors"),
        (0.79, 0.28, 0.13, 0.16, "Weak tree 200\nlearns errors"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor="#fff0df", edgecolor="#ba6a20", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)

    arrowprops = dict(arrowstyle="->", linewidth=1.5, color="#ba6a20")
    ax.annotate("", xy=(0.25, 0.78), xytext=(0.18, 0.66), arrowprops=arrowprops)
    ax.annotate("", xy=(0.43, 0.64), xytext=(0.38, 0.78), arrowprops=arrowprops)
    ax.annotate("", xy=(0.61, 0.50), xytext=(0.56, 0.64), arrowprops=arrowprops)
    ax.annotate("", xy=(0.79, 0.36), xytext=(0.74, 0.50), arrowprops=arrowprops)

    ax.text(0.32, 0.90, "first model", ha="center", va="center", fontsize=9)
    ax.text(0.51, 0.80, "next model focuses on previous mistakes", ha="center", va="center", fontsize=9)
    ax.text(0.68, 0.66, "sequential improvement", ha="center", va="center", fontsize=9)

    ax.text(
        0.5,
        0.08,
        "Boosting builds trees sequentially. Each new weak tree tries to improve the current ensemble.",
        ha="center",
        va="center",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("q4_boosting_concept.png", dpi=180)
    plt.close()


def save_feature_importance_plot(feature_importances):
    top = feature_importances.sort_values("importance", ascending=True).tail(8)

    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance"], color="#d6842d")
    plt.xlabel("Feature importance")
    plt.title("Most important predictors in the Boosting model")
    plt.tight_layout()
    plt.savefig("q4_boosting_feature_importance.png", dpi=180)
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

    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_error = 1 - accuracy
    cm = confusion_matrix(y_test, y_pred)

    results = pd.DataFrame(
        [
            {
                "model": "Boosting",
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LEARNING_RATE,
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
    results.to_csv("q4_explanatory_results.csv", index=False)

    comparison = pd.DataFrame(
        [
            {"model": "Decision Tree (max_depth=2)", "test_accuracy": 0.933920704845815, "test_error": 0.06607929515418498},
            {"model": "Bagging (200 estimators)", "test_accuracy": 0.9660163624921334, "test_error": 0.03398363750786659},
            {"model": "Random Forest (200 estimators, m=5)", "test_accuracy": 0.9666456891126495, "test_error": 0.033354310887350525},
            {"model": "Boosting (200 estimators)", "test_accuracy": accuracy, "test_error": test_error},
        ]
    )
    comparison.to_csv("q4_comparison_q1_q2_q3_q4.csv", index=False)

    feature_importances = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    )
    feature_importances.sort_values("importance", ascending=False).to_csv(
        "q4_boosting_feature_importance.csv",
        index=False,
    )

    save_confusion_matrix(cm)
    save_boosting_concept_diagram()
    save_feature_importance_plot(feature_importances)

    print(results.to_string(index=False))
    print("\nSaved explanatory artifacts for Question 4.")


if __name__ == "__main__":
    main()
