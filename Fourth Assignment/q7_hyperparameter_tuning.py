"""
4th Assignment - Question 7

Hyperparameter tuning for:
1. Classification Tree
2. Bagging
3. Random Forest
4. Boosting

GridSearchCV with 5-fold stratified cross-validation is applied on the
training set. Final performance is evaluated once on the test set.
"""

import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


CSV_PATH = "../Third Assigment/Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30
CV_FOLDS = 5


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, 1 - accuracy, cm


def main():
    df = pd.read_csv(CSV_PATH)

    features = [col for col in df.columns if col not in ("quality", "wine_type")]
    p = len(features)
    X = df[features]
    y = (df["wine_type"] == "white").astype(int)  # red=0, white=1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    untuned_models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features=p // 2,
            random_state=RANDOM_STATE,
        ),
        "Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=1,
            random_state=RANDOM_STATE,
        ),
    }

    tuning_setups = {
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 12, None],
                "min_samples_leaf": [1, 2, 5, 10],
            },
        },
        "Bagging": {
            "estimator": BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_samples": [0.7, 1.0],
                "max_features": [0.7, 1.0],
                "estimator__max_depth": [None, 6],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_features": [3, p // 2, "sqrt"],
                "max_depth": [None, 6, 10],
                "min_samples_leaf": [1, 2],
            },
        },
        "Boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [1, 2],
                "subsample": [0.8, 1.0],
            },
        },
    }

    rows = []
    cm_rows = []

    for method, setup in tuning_setups.items():
        print(f"\n===== {method} =====")

        untuned_accuracy, untuned_error, untuned_cm = evaluate_model(
            untuned_models[method], X_train, X_test, y_train, y_test
        )

        grid_search = GridSearchCV(
            estimator=setup["estimator"],
            param_grid=setup["param_grid"],
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        grid_search.fit(X_train, y_train)

        tuned_model = grid_search.best_estimator_
        y_pred = tuned_model.predict(X_test)
        tuned_cm = confusion_matrix(y_test, y_pred)
        tuned_accuracy = accuracy_score(y_test, y_pred)
        tuned_error = 1 - tuned_accuracy

        print(f"Untuned test accuracy: {untuned_accuracy:.4f}")
        print(f"Best CV accuracy:      {grid_search.best_score_:.4f}")
        print(f"Tuned test accuracy:   {tuned_accuracy:.4f}")
        print(f"Best parameters:       {grid_search.best_params_}")

        rows.append(
            {
                "method": method,
                "untuned_test_accuracy": untuned_accuracy,
                "untuned_test_error": untuned_error,
                "best_cv_accuracy": grid_search.best_score_,
                "tuned_test_accuracy": tuned_accuracy,
                "tuned_test_error": tuned_error,
                "best_params": str(grid_search.best_params_),
            }
        )

        for variant, cm in [("untuned", untuned_cm), ("tuned", tuned_cm)]:
            cm_rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv("q7_hyperparameter_tuning_summary.csv", index=False)
    pd.DataFrame(cm_rows).to_csv("q7_hyperparameter_tuning_confusion_matrices.csv", index=False)

    print("\n===== Summary =====")
    print(
        summary[
            [
                "method",
                "untuned_test_accuracy",
                "best_cv_accuracy",
                "tuned_test_accuracy",
                "best_params",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print(
        "\nSaved: q7_hyperparameter_tuning_summary.csv, "
        "q7_hyperparameter_tuning_confusion_matrices.csv"
    )


if __name__ == "__main__":
    main()
