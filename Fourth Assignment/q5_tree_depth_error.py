"""
4th Assignment - Question 5

Plot the test error of a classification tree as a function of max_depth.
The predictors are all variables except quality and wine_type.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


CSV_PATH = "../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
DEPTHS = list(range(1, 21))


def main():
    df = pd.read_csv(CSV_PATH)

    features = [col for col in df.columns if col not in ("quality", "wine_type")]
    X = df[features]
    y = (df["wine_type"] == "white").astype(int)  # red=0, white=1

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
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        rows.append(
            {
                "max_depth": depth,
                "test_accuracy": accuracy,
                "test_error": 1 - accuracy,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv("q5_tree_depth_error.csv", index=False)

    best_row = results.loc[results["test_error"].idxmin()]
    print("Question 5 - Decision Tree test error by max_depth")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\nBest depth by test error: "
        f"{int(best_row['max_depth'])} "
        f"(test accuracy={best_row['test_accuracy']:.4f}, "
        f"test error={best_row['test_error']:.4f})"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        results["max_depth"],
        results["test_error"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    plt.axvline(
        2,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Question 1: max_depth=2",
    )
    plt.axvline(
        int(best_row["max_depth"]),
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label=f"Lowest test error: depth={int(best_row['max_depth'])}",
    )
    plt.xticks(DEPTHS)
    plt.xlabel("Tree max_depth")
    plt.ylabel("Test error (1 - test accuracy)")
    plt.title("Decision Tree test error by tree depth")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("q5_tree_depth_error.png", dpi=160)
    plt.close()

    print("\nSaved: q5_tree_depth_error.csv, q5_tree_depth_error.png")


if __name__ == "__main__":
    main()
