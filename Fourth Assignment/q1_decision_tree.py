"""
4th Assignment - Question 1

Predict wine type (red/white) using a classification tree with max_depth=2.
The predictors are all variables except quality and wine_type.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


CSV_PATH = "../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30


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

    model = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_error = 1 - accuracy
    cm = confusion_matrix(y_test, y_pred)

    print("Question 1 - Classification Tree (max_depth=2)")
    print(f"Predictors ({len(features)}): {features}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test error: {test_error:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=["true_red", "true_white"], columns=["pred_red", "pred_white"]))
    print("\nTree rules:")
    print(export_text(model, feature_names=features))

    results = pd.DataFrame(
        [
            {
                "model": "Decision Tree",
                "max_depth": 2,
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
    results.to_csv("q1_decision_tree_results.csv", index=False)


if __name__ == "__main__":
    main()
