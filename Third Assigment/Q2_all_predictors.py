"""
Άσκηση 3 - Ερώτημα Β
Πρόβλεψη του τύπου κρασιού με ΟΛΕΣ τις προβλεπτικές μεταβλητές.
Μέθοδοι:
  a) KNN (το K επιλέγεται με 5-fold CV)
  b) Logistic Regression
  c) LDA
  d) QDA
  e) Gaussian Naive Bayes

Σημείωση για κανονικοποίηση:
  - KNN & LogReg: ΑΠΑΡΑΙΤΗΤΗ τυποποίηση (StandardScaler)
    KNN επειδή στηρίζεται σε ευκλείδειες αποστάσεις,
    LogReg για αριθμητική σταθερότητα/σύγκλιση.
  - LDA, QDA, GNB: αμετάβλητα από γραμμική κλιμάκωση,
    οπότε δεν χρειάζεται (αλλά είναι αβλαβές).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)
from statsmodels.stats.contingency_tables import mcnemar

CSV_PATH = "Data Analysis_2026 3rd Case_Data.csv"
RANDOM_STATE = 6931
TEST_SIZE = 0.30

# ------------------------ Δεδομένα ------------------------
df = pd.read_csv(CSV_PATH)
features = [c for c in df.columns if c not in ("wine_type", "quality")]
X = df[features].values
y = (df["wine_type"] == "white").astype(int).values   # red=0, white=1

print(f"Πλήθος προβλεπτικών μεταβλητών: {len(features)}")
print(f"Features: {features}")
print(f"Κατανομή τάξεων: red={int((y==0).sum())}, white={int((y==1).sum())}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ------------------------ KNN: επιλογή K με 5-fold CV ------------------------
print("====== KNN: επιλογή K με 5-fold stratified CV ======")
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])
k_grid = list(range(1, 31))
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(knn_pipe, {"knn__n_neighbors": k_grid},
                  scoring="accuracy", cv=cv5, n_jobs=-1)
gs.fit(X_train, y_train)
best_k = gs.best_params_["knn__n_neighbors"]
print(f"Βέλτιστο K = {best_k} (CV accuracy = {gs.best_score_:.4f})")

cv_results = pd.DataFrame({
    "K": k_grid,
    "cv_accuracy": gs.cv_results_["mean_test_score"],
    "cv_std": gs.cv_results_["std_test_score"],
})
cv_results.to_csv("Q2_knn_cv.csv", index=False)

plt.figure(figsize=(7, 4))
plt.errorbar(cv_results["K"], cv_results["cv_accuracy"],
             yerr=cv_results["cv_std"], marker="o", capsize=3)
plt.axvline(best_k, color="red", ls="--", label=f"Best K = {best_k}")
plt.xlabel("K (αριθμός γειτόνων)")
plt.ylabel("CV accuracy (5-fold)")
plt.title("KNN — επιλογή K μέσω 5-fold CV")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("Q2_knn_cv.png", dpi=140)
plt.close()

# ------------------------ Ορισμός όλων των μοντέλων ------------------------
models = {
    f"KNN (K={best_k})": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=best_k)),
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=5000)),
    ]),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=1e-3),
    "Gaussian NB": GaussianNB(),
}

# ------------------------ Αξιολόγηση στο test ------------------------
def metrics_row(name, y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "method": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "sens_white": tp / (tp + fn) if (tp + fn) else 0,
        "spec_red": tn / (tn + fp) if (tn + fp) else 0,
        "prec_white": tp / (tp + fp) if (tp + fp) else 0,
        "prec_red": tn / (tn + fn) if (tn + fn) else 0,
        "f1_white": f1_score(y_true, y_pred, pos_label=1),
        "f1_red": f1_score(y_true, y_pred, pos_label=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

results, predictions, probas = [], {}, {}
print("\n====== Αξιολόγηση στο test set ======")
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:, 1]
    predictions[name] = y_pred
    probas[name] = y_proba
    row = metrics_row(name, y_test, y_pred, y_proba)
    results.append(row)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(pd.DataFrame(cm, index=["true_red", "true_white"],
                       columns=["pred_red", "pred_white"]))
    print(f"Accuracy={row['accuracy']:.4f}  BalAcc={row['bal_acc']:.4f}  "
          f"AUC={row['auc']:.4f}  "
          f"Sens(white)={row['sens_white']:.4f}  Spec(red)={row['spec_red']:.4f}")

results_df = pd.DataFrame(results)
print("\n====== Συγκεντρωτικός πίνακας ======")
print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
results_df.to_csv("Q2_summary.csv", index=False)

# ------------------------ ROC curves ------------------------
plt.figure(figsize=(7, 6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    auc = roc_auc_score(y_test, probas[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
plt.xlabel("False Positive Rate (1 − Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC curves — όλες οι μέθοδοι (Q2)")
plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("Q2_roc.png", dpi=140)
plt.close()
print("\nROC γράφημα: Q2_roc.png")

# ------------------------ McNemar test (paired) ------------------------
print("\n====== McNemar test (paired) ======")
mc_rows = []
correct = {n: (predictions[n] == y_test).astype(int) for n in models}
for a, b in combinations(models.keys(), 2):
    b01 = int(np.sum((correct[a] == 1) & (correct[b] == 0)))
    b10 = int(np.sum((correct[a] == 0) & (correct[b] == 1)))
    if b01 + b10 < 25:
        res = mcnemar([[0, b01], [b10, 0]], exact=True)
    else:
        res = mcnemar([[0, b01], [b10, 0]], exact=False, correction=True)
    mc_rows.append({
        "pair": f"{a}  vs  {b}",
        "A_only_correct": b01,
        "B_only_correct": b10,
        "p_value": res.pvalue,
    })

mc_df = pd.DataFrame(mc_rows)
print(mc_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
mc_df.to_csv("Q2_mcnemar.csv", index=False)

print("\nΑποθηκεύτηκαν: Q2_summary.csv, Q2_mcnemar.csv, Q2_knn_cv.csv, "
      "Q2_knn_cv.png, Q2_roc.png")
