"""
Explanatory Report - 4th Assignment - Question 5

This script studies the test error of a decision tree as max_depth changes
from 1 to 20, then saves artifacts used by the unified explanatory report.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


CSV_PATH = "../../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
DEPTHS = list(range(1, 21))


def save_depth_plot(results):
    best_row = results.loc[results["test_error"].idxmin()]

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        results["max_depth"],
        results["test_error"],
        marker="o",
        linewidth=2,
        color="#2364aa",
        label="Test error",
    )
    plt.axvline(
        2,
        color="#c23b22",
        linestyle="--",
        linewidth=1.6,
        label="Question 1: max_depth=2",
    )
    plt.axvline(
        int(best_row["max_depth"]),
        color="#2f8f46",
        linestyle=":",
        linewidth=2,
        label=f"Lowest test error: depth={int(best_row['max_depth'])}",
    )
    plt.scatter(
        [best_row["max_depth"]],
        [best_row["test_error"]],
        s=95,
        color="#2f8f46",
        zorder=5,
    )
    plt.xticks(DEPTHS)
    plt.xlabel("Tree max_depth")
    plt.ylabel("Test error (1 - test accuracy)")
    plt.title("Decision Tree test error by max_depth")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("q5_tree_depth_error.png", dpi=180)
    plt.close()


def save_depth_concept_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")

    boxes = [
        (0.05, 0.58, 0.22, 0.22, "Very shallow tree\nlow complexity\npossible underfitting"),
        (0.39, 0.58, 0.22, 0.22, "Moderate depth\nbetter balance\nbest generalization"),
        (0.73, 0.58, 0.22, 0.22, "Very deep tree\nhigh complexity\npossible overfitting"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor="#e7f0fb", edgecolor="#2364aa", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)

    arrowprops = dict(arrowstyle="->", linewidth=1.5, color="#2364aa")
    ax.annotate("", xy=(0.39, 0.69), xytext=(0.27, 0.69), arrowprops=arrowprops)
    ax.annotate("", xy=(0.73, 0.69), xytext=(0.61, 0.69), arrowprops=arrowprops)

    ax.text(0.16, 0.32, "Too simple:\nmisses patterns", ha="center", va="center", fontsize=10)
    ax.text(0.50, 0.32, "Good compromise:\nlow test error", ha="center", va="center", fontsize=10)
    ax.text(0.84, 0.32, "Too complex:\nlearns noise/details", ha="center", va="center", fontsize=10)

    ax.text(
        0.5,
        0.08,
        "Question 5 checks how tree complexity affects test error.",
        ha="center",
        va="center",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig("q5_tree_depth_concept.png", dpi=180)
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

    rows = []
    for depth in DEPTHS:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        rows.append(
            {
                "max_depth": depth,
                "train_accuracy": train_accuracy,
                "train_error": 1 - train_accuracy,
                "test_accuracy": test_accuracy,
                "test_error": 1 - test_accuracy,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv("q5_explanatory_tree_depth_results.csv", index=False)

    best_row = results.loc[results["test_error"].idxmin()]
    summary = pd.DataFrame(
        [
            {
                "best_max_depth": int(best_row["max_depth"]),
                "best_test_accuracy": best_row["test_accuracy"],
                "best_test_error": best_row["test_error"],
                "question_1_depth": 2,
                "question_1_test_accuracy": results.loc[results["max_depth"] == 2, "test_accuracy"].iloc[0],
                "question_1_test_error": results.loc[results["max_depth"] == 2, "test_error"].iloc[0],
            }
        ]
    )
    summary.to_csv("q5_explanatory_tree_depth_summary.csv", index=False)

    save_depth_plot(results)
    save_depth_concept_diagram()

    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBest depth:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved explanatory artifacts for Question 5.")


if __name__ == "__main__":
    main()
