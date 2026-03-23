"""
Second Assignment - Data Analysis
Sub-questions 1 & 2: Simple and Multiple Linear Regression

Using the cleaned data from the 1st assignment (AEM 6931) to investigate
whether predictor variables can explain/predict wine quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Load the cleaned data from the first assignment
# ============================================================
DATA_PATH = Path(__file__).resolve().parents[1] / 'First Assignment' / 'cleaned_wine_data_AEM_6931.csv'
OUTPUT_DIR = Path(__file__).resolve().parent
data = pd.read_csv(DATA_PATH)

print("=" * 70)
print("SECOND ASSIGNMENT - LINEAR REGRESSION ANALYSIS")
print("=" * 70)
print(f"\nDataset shape: {data.shape}")
print(f"\nVariables: {list(data.columns)}")

# Response variable: quality
# Predictor variables: all continuous variables (excluding wine_type which is categorical)
response = 'quality'
predictors = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
              'density', 'pH', 'sulphates', 'alcohol']

print(f"\nResponse variable: {response}")
print(f"Predictor variables ({len(predictors)}): {predictors}")

# ============================================================
# SUB-QUESTION 1: Simple Linear Regression Models
# ============================================================
print("\n" + "=" * 70)
print("SUB-QUESTION 1: SIMPLE LINEAR REGRESSION")
print("Fit quality ~ each predictor individually")
print("=" * 70)

# Store results
simple_results = []

for predictor in predictors:
    X = sm.add_constant(data[predictor])
    y = data[response]
    model = sm.OLS(y, X).fit()

    result = {
        'Predictor': predictor,
        'β0 (intercept)': model.params.iloc[0],
        'β1 (slope)': model.params.iloc[1],
        'p-value (β1)': model.pvalues.iloc[1],
        't-statistic': model.tvalues.iloc[1],
        'R²': model.rsquared,
        'R² adj': model.rsquared_adj,
        'F-statistic': model.fvalue,
        'Significant (α=0.05)': 'Yes' if model.pvalues.iloc[1] < 0.05 else 'No'
    }
    simple_results.append(result)

    print(f"\n{'─' * 50}")
    print(f"Model: quality ~ {predictor}")
    print(f"  β0 (intercept) = {model.params.iloc[0]:.4f}")
    print(f"  β1 (slope)     = {model.params.iloc[1]:.6f}")
    print(f"  t-statistic    = {model.tvalues.iloc[1]:.4f}")
    print(f"  p-value        = {model.pvalues.iloc[1]:.6e}")
    print(f"  R²             = {model.rsquared:.4f}")
    print(f"  R² adjusted    = {model.rsquared_adj:.4f}")
    if model.pvalues.iloc[1] < 0.05:
        print(f"  *** STATISTICALLY SIGNIFICANT (p < 0.05) ***")
    else:
        print(f"  Not statistically significant (p >= 0.05)")

# Summary table
results_df = pd.DataFrame(simple_results)
results_df = results_df.sort_values('p-value (β1)')

print("\n\n" + "=" * 70)
print("SUMMARY TABLE - Simple Linear Regression Results")
print("=" * 70)
print(results_df.to_string(index=False))

# Identify significant predictors
significant = results_df[results_df['Significant (α=0.05)'] == 'Yes']
not_significant = results_df[results_df['Significant (α=0.05)'] == 'No']

print(f"\n\nStatistically significant predictors ({len(significant)}):")
for _, row in significant.iterrows():
    print(f"  - {row['Predictor']} (p = {row['p-value (β1)']:.6e}, R² = {row['R²']:.4f})")

if len(not_significant) > 0:
    print(f"\nNot statistically significant predictors ({len(not_significant)}):")
    for _, row in not_significant.iterrows():
        print(f"  - {row['Predictor']} (p = {row['p-value (β1)']:.6e})")

# Save results table
results_df.to_csv(OUTPUT_DIR / 'simple_regression_results.csv', index=False)

# ============================================================
# Scatter plots for significant predictors
# ============================================================
sig_predictors = significant['Predictor'].tolist()
n_sig = len(sig_predictors)

if n_sig > 0:
    # Determine grid size
    ncols = 3
    nrows = int(np.ceil(n_sig / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, predictor in enumerate(sig_predictors):
        ax = axes[i]
        # Add jitter to quality for better visualization
        jitter = np.random.normal(0, 0.1, size=len(data))
        ax.scatter(data[predictor], data[response] + jitter,
                   alpha=0.15, s=10, color='steelblue')

        # Regression line
        X_plot = sm.add_constant(data[predictor])
        model = sm.OLS(data[response], X_plot).fit()
        x_range = np.linspace(data[predictor].min(), data[predictor].max(), 100)
        y_pred = model.params.iloc[0] + model.params.iloc[1] * x_range
        ax.plot(x_range, y_pred, color='red', linewidth=2, label='Regression line')

        # Get p-value and R²
        p_val = model.pvalues.iloc[1]
        r2 = model.rsquared

        ax.set_xlabel(predictor, fontsize=11)
        ax.set_ylabel('Quality', fontsize=11)
        ax.set_title(f'Quality vs {predictor}\nR²={r2:.4f}, p={p_val:.2e}', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_plots_significant.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nScatter plots saved to: scatter_plots_significant.png")

# ============================================================
# SUB-QUESTION 2: Multiple Linear Regression
# ============================================================
print("\n\n" + "=" * 70)
print("SUB-QUESTION 2: MULTIPLE LINEAR REGRESSION")
print("Model: quality ~ all predictors")
print("=" * 70)

X_multi = sm.add_constant(data[predictors])
y = data[response]
multi_model = sm.OLS(y, X_multi).fit()

print("\n" + str(multi_model.summary()))

# Detailed coefficient analysis
print("\n\n" + "─" * 70)
print("COEFFICIENT ANALYSIS - Multiple Regression")
print("─" * 70)
print(f"\n{'Variable':<25} {'Coefficient':>12} {'Std Error':>12} {'t-value':>10} {'p-value':>12} {'Significant':>12}")
print("─" * 85)

alpha = 0.05
rejected_H0 = []
not_rejected_H0 = []

for var in multi_model.params.index:
    coef = multi_model.params[var]
    se = multi_model.bse[var]
    t_val = multi_model.tvalues[var]
    p_val = multi_model.pvalues[var]
    sig = "Yes ***" if p_val < 0.05 else "No"

    print(f"{var:<25} {coef:>12.6f} {se:>12.6f} {t_val:>10.4f} {p_val:>12.6e} {sig:>12}")

    if var != 'const':
        if p_val < alpha:
            rejected_H0.append((var, p_val, coef))
        else:
            not_rejected_H0.append((var, p_val, coef))

print(f"\n\nModel Statistics:")
print(f"  R²           = {multi_model.rsquared:.4f}")
print(f"  R² adjusted  = {multi_model.rsquared_adj:.4f}")
print(f"  F-statistic  = {multi_model.fvalue:.4f}")
print(f"  Prob(F)      = {multi_model.f_pvalue:.6e}")
print(f"  AIC          = {multi_model.aic:.2f}")
print(f"  BIC          = {multi_model.bic:.2f}")

print(f"\n\nVariables where H₀: βⱼ = 0 is REJECTED (p < {alpha}):")
print(f"(These predictors have statistically significant effect on quality)")
for var, p, coef in sorted(rejected_H0, key=lambda x: x[1]):
    direction = "positive" if coef > 0 else "negative"
    print(f"  - {var}: β = {coef:.6f} ({direction} effect), p = {p:.6e}")

print(f"\nVariables where H₀: βⱼ = 0 is NOT rejected (p >= {alpha}):")
print(f"(No statistically significant evidence of effect on quality)")
for var, p, coef in sorted(not_rejected_H0, key=lambda x: x[1]):
    print(f"  - {var}: β = {coef:.6f}, p = {p:.6e}")

# Save multiple regression summary
with open(OUTPUT_DIR / 'multiple_regression_summary.txt', 'w') as f:
    f.write(str(multi_model.summary()))
    f.write("\n\nVariables where H0: βj = 0 is REJECTED (p < 0.05):\n")
    for var, p, coef in sorted(rejected_H0, key=lambda x: x[1]):
        f.write(f"  - {var}: β = {coef:.6f}, p = {p:.6e}\n")
    f.write("\nVariables where H0: βj = 0 is NOT rejected (p >= 0.05):\n")
    for var, p, coef in sorted(not_rejected_H0, key=lambda x: x[1]):
        f.write(f"  - {var}: β = {coef:.6f}, p = {p:.6e}\n")

print("\n\nResults saved to:")
print("  - simple_regression_results.csv")
print("  - scatter_plots_significant.png")
print("  - multiple_regression_summary.txt")
