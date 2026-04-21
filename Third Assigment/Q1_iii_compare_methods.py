"""
Άσκηση 3 - Ερώτημα Α(iii)
Σύγκριση επίδοσης Logistic Regression / LDA / Gaussian NB
χρησιμοποιώντας ΜΟΝΟ τη μεταβλητή chlorides.
- McNemar test (paired) στο test set
- 10-fold stratified CV για accuracy και AUC
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.contingency_tables import mcnemar

CSV_PATH = "Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30

df = pd.read_csv(CSV_PATH)
X = df[["chlorides"]].values
y = (df["wine_type"] == "white").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

models = {
    "LogReg": LogisticRegression(solver="lbfgs", max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),
    "GNB": GaussianNB(),
}

# ---------- 1) Predictions on test ----------
preds = {}
correct = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    preds[name] = m.predict(X_test)
    correct[name] = (preds[name] == y_test).astype(int)

# ---------- 2) McNemar test for each pair ----------
print("============ McNemar test (paired, test set) ============")
print("H0: οι δύο μέθοδοι έχουν ίδιο ποσοστό λανθασμένων προβλέψεων.\n")
mcnemar_rows = []
for a, b in combinations(models.keys(), 2):
    # b01: A correct, B wrong | b10: A wrong, B correct
    b01 = int(np.sum((correct[a] == 1) & (correct[b] == 0)))
    b10 = int(np.sum((correct[a] == 0) & (correct[b] == 1)))
    table = [[0, b01], [b10, 0]]
    res = mcnemar(table, exact=True)  # exact binomial (κατάλληλο για μικρά b01+b10)
    mcnemar_rows.append({
        "pair": f"{a} vs {b}",
        "A_correct_B_wrong": b01,
        "A_wrong_B_correct": b10,
        "statistic": res.statistic,
        "p_value": res.pvalue,
    })
    print(f"{a:7s} vs {b:7s}: b01={b01:4d}, b10={b10:4d}, "
          f"p-value={res.pvalue:.4g}")

print()
print(pd.DataFrame(mcnemar_rows).to_string(index=False))

# ---------- 3) 10-fold stratified CV ----------
print("\n============ 10-fold Stratified CV ============")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, m in models.items():
    acc_scores = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
    bal_scores = cross_val_score(m, X, y, cv=cv, scoring="balanced_accuracy")
    cv_rows.append({
        "method": name,
        "acc_mean": acc_scores.mean(), "acc_std": acc_scores.std(),
        "bal_mean": bal_scores.mean(), "bal_std": bal_scores.std(),
        "auc_mean": auc_scores.mean(), "auc_std": auc_scores.std(),
    })

cv_df = pd.DataFrame(cv_rows)
print(cv_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

cv_df.to_csv("Q1_iii_cv_results.csv", index=False)
pd.DataFrame(mcnemar_rows).to_csv("Q1_iii_mcnemar.csv", index=False)
print("\nΑποθηκεύτηκαν: Q1_iii_cv_results.csv, Q1_iii_mcnemar.csv")
