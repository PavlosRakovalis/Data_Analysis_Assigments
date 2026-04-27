"""
4th Assignment - Question 4

Predict wine type (red/white) using Boosting with 200 estimators,
learning_rate=0.1, and max_depth=1.
The predictors are all variables except quality and wine_type.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


CSV_PATH = "../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
N_ESTIMATORS = 200
LEARNING_RATE = 0.1
MAX_DEPTH = 1


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

    print("Question 4 - Boosting")
    print(f"Predictors ({len(features)}): {features}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"n_estimators: {N_ESTIMATORS}")
    print(f"learning_rate: {LEARNING_RATE}")
    print(f"max_depth: {MAX_DEPTH}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test error: {test_error:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=["true_red", "true_white"], columns=["pred_red", "pred_white"]))

    results = pd.DataFrame(
        [
            {
                "model": "Boosting",
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
                "random_state": RANDOM_STATE,
                "test_accuracy": accuracy,
                "test_error": test_error,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        ]
    )
    results.to_csv("q4_boosting_results.csv", index=False)


if __name__ == "__main__":
    main()
