"""
4th Assignment - Question 6

Plot test error for Bagging, Random Forest, and Boosting as a function
of n_estimators. The predictors are all variables except quality and wine_type.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


CSV_PATH = "../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
N_ESTIMATORS_VALUES = [1] + list(range(10, 201, 10))
LEARNING_RATE = 0.1
BOOSTING_MAX_DEPTH = 1


def main():
    df = pd.read_csv(CSV_PATH)

    features = [col for col in df.columns if col not in ("quality", "wine_type")]
    max_features = len(features) // 2
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
    for n_estimators in N_ESTIMATORS_VALUES:
        models = {
            "Bagging": BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                n_estimators=n_estimators,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "Boosting": GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=LEARNING_RATE,
                max_depth=BOOSTING_MAX_DEPTH,
                random_state=RANDOM_STATE,
            ),
        }

        for method, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            rows.append(
                {
                    "method": method,
                    "n_estimators": n_estimators,
                    "test_accuracy": accuracy,
                    "test_error": 1 - accuracy,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv("q6_ensemble_n_estimators_error.csv", index=False)

    print("Question 6 - Ensemble test error by n_estimators")
    pivot = results.pivot(index="n_estimators", columns="method", values="test_error")
    print(pivot.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\nBest result per method:")
    for method in ["Bagging", "Random Forest", "Boosting"]:
        method_results = results[results["method"] == method]
        best_row = method_results.loc[method_results["test_error"].idxmin()]
        print(
            f"{method}: n_estimators={int(best_row['n_estimators'])}, "
            f"test_accuracy={best_row['test_accuracy']:.4f}, "
            f"test_error={best_row['test_error']:.4f}"
        )

    plt.figure(figsize=(9, 5.5))
    colors = {
        "Bagging": "#1f77b4",
        "Random Forest": "#2ca02c",
        "Boosting": "#d62728",
    }
    for method in ["Bagging", "Random Forest", "Boosting"]:
        method_results = results[results["method"] == method]
        plt.plot(
            method_results["n_estimators"],
            method_results["test_error"],
            marker="o",
            linewidth=2,
            label=method,
            color=colors[method],
        )

    plt.xlabel("n_estimators")
    plt.ylabel("Test error (1 - test accuracy)")
    plt.title("Test error by n_estimators")
    plt.xticks(N_ESTIMATORS_VALUES)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("q6_ensemble_n_estimators_error.png", dpi=160)
    plt.close()

    print("\nSaved: q6_ensemble_n_estimators_error.csv, q6_ensemble_n_estimators_error.png")


if __name__ == "__main__":
    main()
