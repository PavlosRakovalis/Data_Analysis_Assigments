"""
Άσκηση 3 - Ερώτημα Α
Πρόβλεψη του τύπου κρασιού (red/white) με μία προβλεπτική μεταβλητή τη φορά.
Μεταβλητές: alcohol, pH, chlorides
Μέθοδοι: Logistic Regression, Linear Discriminant Analysis,
         Gaussian Naive Bayes (κανονική κατανομή ανά κλάση)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

CSV_PATH = "Data Analysis_2026 3rd Case_Data.csv"
PREDICTORS = ["alcohol", "pH", "chlorides"]
RANDOM_STATE = 6931
TEST_SIZE = 0.30


def evaluate(model, name, predictor, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # white = positive class (1), red = negative class (0)
    sensitivity_white = tp / (tp + fn) if (tp + fn) else 0.0   # recall white
    specificity_red = tn / (tn + fp) if (tn + fp) else 0.0     # recall red
    precision_white = tp / (tp + fp) if (tp + fp) else 0.0
    precision_red = tn / (tn + fn) if (tn + fn) else 0.0
    f1_white = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_red = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    print(f"\n--- {name} | predictor: {predictor} ---")
    if hasattr(model, "coef_"):
        print(f"Coefficient (β1): {model.coef_.ravel()[0]:.6f}")
        print(f"Intercept   (β0): {float(np.ravel(model.intercept_)[0]):.6f}")
    if isinstance(model, GaussianNB):
        for cls, mean, var, prior in zip(
            model.classes_, model.theta_.ravel(),
            model.var_.ravel(), model.class_prior_
        ):
            label = "red" if cls == 0 else "white"
            print(f"Class {label}: prior={prior:.4f}, "
                  f"mean={mean:.4f}, std={np.sqrt(var):.4f}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"AUC:               {auc:.4f}")
    print(f"Sensitivity (white recall): {sensitivity_white:.4f}")
    print(f"Specificity (red recall):   {specificity_red:.4f}")
    print(f"Precision white: {precision_white:.4f} | Precision red: {precision_red:.4f}")
    print(f"F1 white:        {f1_white:.4f} | F1 red:        {f1_red:.4f}")
    print("Confusion Matrix (rows=true [red, white], cols=pred [red, white]):")
    print(pd.DataFrame(cm, index=["true_red", "true_white"],
                       columns=["pred_red", "pred_white"]))

    return {
        "method": name,
        "predictor": predictor,
        "accuracy": round(acc, 4),
        "bal_acc": round(bal_acc, 4),
        "auc": round(auc, 4),
        "sens_white": round(sensitivity_white, 4),
        "spec_red": round(specificity_red, 4),
        "prec_white": round(precision_white, 4),
        "prec_red": round(precision_red, 4),
        "f1_white": round(f1_white, 4),
        "f1_red": round(f1_red, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Σχήμα δεδομένων: {df.shape}")
    print(f"Κατανομή τάξεων:\n{df['wine_type'].value_counts()}\n")

    # Encode: red=0, white=1
    y = (df["wine_type"] == "white").astype(int)

    summary = []
    for predictor in PREDICTORS:
        X = df[[predictor]].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logreg = LogisticRegression(solver="lbfgs", max_iter=1000)
        summary.append(evaluate(logreg, "Logistic Regression",
                                predictor, X_train, X_test, y_train, y_test))

        lda = LinearDiscriminantAnalysis()
        summary.append(evaluate(lda, "LDA",
                                predictor, X_train, X_test, y_train, y_test))

        gnb = GaussianNB()
        summary.append(evaluate(gnb, "Gaussian NB",
                                predictor, X_train, X_test, y_train, y_test))

    print("\n========== Συγκεντρωτικός πίνακας ==========")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("Q1_summary.csv", index=False)
    print("\nΑποθηκεύτηκε: Q1_summary.csv")

    print("\n========== Κατάταξη ανά AUC ==========")
    print(summary_df.sort_values("auc", ascending=False)
          [["predictor", "method", "accuracy", "bal_acc", "auc",
            "sens_white", "spec_red", "f1_red"]].to_string(index=False))


if __name__ == "__main__":
    main()
