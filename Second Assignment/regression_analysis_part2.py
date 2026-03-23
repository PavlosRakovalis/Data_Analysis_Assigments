"""
Second Assignment - Data Analysis (Part 2)
Sub-questions 3, 4, 5: Forward Selection, Polynomial Regression, Model Critique
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import combinations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Load the cleaned data
# ============================================================
DATA_PATH = Path(__file__).resolve().parents[1] / 'First Assignment' / 'cleaned_wine_data_AEM_6931.csv'
OUTPUT_DIR = Path(__file__).resolve().parent
data = pd.read_csv(DATA_PATH)

response = 'quality'
predictors = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
              'density', 'pH', 'sulphates', 'alcohol']

y = data[response]

# ============================================================
# SUB-QUESTION 3: Forward Selection
# ============================================================
print("=" * 70)
print("SUB-QUESTION 3: FORWARD SELECTION")
print("Best multiple linear regression model via forward variable selection")
print("=" * 70)

def forward_selection(data, response, predictors, significance_level=0.05):
    """
    Forward selection using p-value criterion.
    At each step, add the variable with the lowest p-value (if < significance_level).
    """
    selected = []
    remaining = list(predictors)
    steps = []

    step = 0
    while remaining:
        step += 1
        pvalues = {}
        for var in remaining:
            candidate = selected + [var]
            X = sm.add_constant(data[candidate])
            model = sm.OLS(data[response], X).fit()
            pvalues[var] = model.pvalues[var]

        best_var = min(pvalues, key=pvalues.get)
        best_pvalue = pvalues[best_var]

        if best_pvalue < significance_level:
            selected.append(best_var)
            remaining.remove(best_var)

            # Fit model with current selected variables
            X = sm.add_constant(data[selected])
            model = sm.OLS(data[response], X).fit()

            step_info = {
                'Step': step,
                'Variable Added': best_var,
                'p-value': best_pvalue,
                'R²': model.rsquared,
                'R² adj': model.rsquared_adj,
                'AIC': model.aic,
                'BIC': model.bic,
                'Variables in model': list(selected)
            }
            steps.append(step_info)

            print(f"\nStep {step}: Add '{best_var}' (p = {best_pvalue:.6e})")
            print(f"  Variables in model: {selected}")
            print(f"  R² = {model.rsquared:.4f}, R² adj = {model.rsquared_adj:.4f}")
            print(f"  AIC = {model.aic:.2f}, BIC = {model.bic:.2f}")
        else:
            print(f"\nStep {step}: No variable meets significance threshold.")
            print(f"  Best candidate: '{best_var}' with p = {best_pvalue:.6e} >= {significance_level}")
            print(f"  STOPPING forward selection.")
            break

    return selected, steps

selected_vars, steps = forward_selection(data, response, predictors, significance_level=0.05)

# Final model summary
print("\n" + "─" * 70)
print("FINAL FORWARD SELECTION MODEL")
print("─" * 70)

X_final = sm.add_constant(data[selected_vars])
final_model = sm.OLS(y, X_final).fit()

print(f"\nVariables included ({len(selected_vars)}):")
for i, var in enumerate(selected_vars, 1):
    coef = final_model.params[var]
    pval = final_model.pvalues[var]
    print(f"  {i}. {var}: β = {coef:.6f}, p = {pval:.6e}")

print(f"\nModel Statistics:")
print(f"  R²          = {final_model.rsquared:.4f}")
print(f"  R² adjusted = {final_model.rsquared_adj:.4f}")
print(f"  AIC         = {final_model.aic:.2f}")
print(f"  BIC         = {final_model.bic:.2f}")
print(f"  F-statistic = {final_model.fvalue:.4f}")
print(f"  Prob(F)     = {final_model.f_pvalue:.6e}")

print("\n" + str(final_model.summary()))

# Forward selection progression plot
steps_df = pd.DataFrame(steps)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(steps_df['Step'], steps_df['R² adj'], 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Step', fontsize=12)
axes[0].set_ylabel('Adjusted R²', fontsize=12)
axes[0].set_title('Forward Selection: Adjusted R² per Step', fontsize=13)
axes[0].grid(True, alpha=0.3)
for i, row in steps_df.iterrows():
    axes[0].annotate(row['Variable Added'], (row['Step'], row['R² adj']),
                     textcoords="offset points", xytext=(5, 10), fontsize=7, rotation=30)

axes[1].plot(steps_df['Step'], steps_df['AIC'], 'rs-', linewidth=2, markersize=8)
axes[1].set_xlabel('Step', fontsize=12)
axes[1].set_ylabel('AIC', fontsize=12)
axes[1].set_title('Forward Selection: AIC per Step', fontsize=13)
axes[1].grid(True, alpha=0.3)

axes[2].plot(steps_df['Step'], steps_df['BIC'], 'g^-', linewidth=2, markersize=8)
axes[2].set_xlabel('Step', fontsize=12)
axes[2].set_ylabel('BIC', fontsize=12)
axes[2].set_title('Forward Selection: BIC per Step', fontsize=13)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'forward_selection_progression.png', dpi=150, bbox_inches='tight')
plt.close()

# Save forward selection results
steps_df.to_csv(OUTPUT_DIR / 'forward_selection_steps.csv', index=False)


# ============================================================
# SUB-QUESTION 4: Non-linear (Polynomial) Regression
# ============================================================
print("\n\n" + "=" * 70)
print("SUB-QUESTION 4: POLYNOMIAL REGRESSION")
print("Y = β0 + β1·X + β2·X² + β3·X³ + ε")
print("For: Residual Sugar, Chlorides, Alcohol")
print("=" * 70)

poly_vars = ['residual sugar', 'chlorides', 'alcohol']

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for idx, var in enumerate(poly_vars):
    print(f"\n{'═' * 60}")
    print(f"Variable: {var}")
    print(f"{'═' * 60}")

    X_val = data[var]

    # ── Linear model (degree 1) ──
    X_lin = sm.add_constant(X_val)
    model_lin = sm.OLS(y, X_lin).fit()

    # ── Quadratic model (degree 2) ──
    X_quad = sm.add_constant(np.column_stack([X_val, X_val**2]))
    X_quad = pd.DataFrame(X_quad, columns=['const', var, f'{var}^2'])
    model_quad = sm.OLS(y, X_quad).fit()

    # ── Cubic model (degree 3) ──
    X_cub = sm.add_constant(np.column_stack([X_val, X_val**2, X_val**3]))
    X_cub = pd.DataFrame(X_cub, columns=['const', var, f'{var}^2', f'{var}^3'])
    model_cub = sm.OLS(y, X_cub).fit()

    # Print comparison
    print(f"\n{'Model':<15} {'R²':>8} {'R² adj':>10} {'AIC':>12} {'BIC':>12}")
    print("─" * 60)
    print(f"{'Linear':<15} {model_lin.rsquared:>8.4f} {model_lin.rsquared_adj:>10.4f} {model_lin.aic:>12.2f} {model_lin.bic:>12.2f}")
    print(f"{'Quadratic':<15} {model_quad.rsquared:>8.4f} {model_quad.rsquared_adj:>10.4f} {model_quad.aic:>12.2f} {model_quad.bic:>12.2f}")
    print(f"{'Cubic':<15} {model_cub.rsquared:>8.4f} {model_cub.rsquared_adj:>10.4f} {model_cub.aic:>12.2f} {model_cub.bic:>12.2f}")

    # Cubic model details
    print(f"\nCubic Model: quality = β0 + β1·{var} + β2·{var}² + β3·{var}³")
    print(f"\n{'Term':<20} {'Coefficient':>12} {'p-value':>14} {'Significant':>12}")
    print("─" * 62)
    for term in model_cub.params.index:
        coef = model_cub.params[term]
        pval = model_cub.pvalues[term]
        sig = "Yes ***" if pval < 0.05 else "No"
        print(f"{term:<20} {coef:>12.6f} {pval:>14.6e} {sig:>12}")

    # Check for non-linearity
    p_quad = model_quad.pvalues.iloc[2]  # p-value of X²
    p_cubic = model_cub.pvalues.iloc[3]  # p-value of X³

    print(f"\nNon-linearity evidence:")
    print(f"  Quadratic term (X²): p = {p_quad:.6e} → {'SIGNIFICANT' if p_quad < 0.05 else 'Not significant'}")
    print(f"  Cubic term (X³):     p = {p_cubic:.6e} → {'SIGNIFICANT' if p_cubic < 0.05 else 'Not significant'}")

    if p_cubic < 0.05:
        print(f"  → Evidence of NON-LINEAR (cubic) relationship")
        best_model_name = "Cubic"
    elif p_quad < 0.05:
        print(f"  → Evidence of NON-LINEAR (quadratic) relationship")
        best_model_name = "Quadratic"
    else:
        print(f"  → No strong evidence of non-linearity; linear model is adequate")
        best_model_name = "Linear"

    # ── Plots ──
    # Scatter with fitted curves
    ax1 = axes[idx, 0]
    x_sort = np.sort(X_val.values)
    jitter = np.random.normal(0, 0.1, size=len(data))
    ax1.scatter(X_val, y + jitter, alpha=0.1, s=8, color='gray', label='Data')

    # Linear fit
    y_lin = model_lin.params.iloc[0] + model_lin.params.iloc[1] * x_sort
    ax1.plot(x_sort, y_lin, 'b-', linewidth=2, label=f'Linear (R²={model_lin.rsquared:.4f})')

    # Quadratic fit
    y_quad = model_quad.params.iloc[0] + model_quad.params.iloc[1] * x_sort + model_quad.params.iloc[2] * x_sort**2
    ax1.plot(x_sort, y_quad, 'r--', linewidth=2, label=f'Quadratic (R²={model_quad.rsquared:.4f})')

    # Cubic fit
    y_cub = (model_cub.params.iloc[0] + model_cub.params.iloc[1] * x_sort +
             model_cub.params.iloc[2] * x_sort**2 + model_cub.params.iloc[3] * x_sort**3)
    ax1.plot(x_sort, y_cub, 'g-.', linewidth=2, label=f'Cubic (R²={model_cub.rsquared:.4f})')

    ax1.set_xlabel(var, fontsize=11)
    ax1.set_ylabel('Quality', fontsize=11)
    ax1.set_title(f'Quality vs {var}\n(Linear, Quadratic, Cubic fits)', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Residual plot for the best polynomial model
    ax2 = axes[idx, 1]
    if best_model_name == "Cubic":
        resid = model_cub.resid
        fitted = model_cub.fittedvalues
        title_resid = "Cubic"
    elif best_model_name == "Quadratic":
        resid = model_quad.resid
        fitted = model_quad.fittedvalues
        title_resid = "Quadratic"
    else:
        resid = model_lin.resid
        fitted = model_lin.fittedvalues
        title_resid = "Linear"

    ax2.scatter(fitted, resid, alpha=0.1, s=8, color='steelblue')
    ax2.axhline(y=0, color='red', linewidth=1.5)
    ax2.set_xlabel('Fitted values', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title(f'Residuals vs Fitted ({title_resid} model)\n{var}', fontsize=11)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'polynomial_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPolynomial regression plots saved to: polynomial_regression.png")


# ============================================================
# SUB-QUESTION 5: Model Appropriateness Critique
# ============================================================
print("\n\n" + "=" * 70)
print("SUB-QUESTION 5: MODEL APPROPRIATENESS CRITIQUE")
print("Is linear regression the right choice for this problem?")
print("=" * 70)

# Fit the full model for diagnostic analysis
X_full = sm.add_constant(data[predictors])
full_model = sm.OLS(y, X_full).fit()

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(full_model.fittedvalues, full_model.resid, alpha=0.15, s=10, color='steelblue')
ax.axhline(y=0, color='red', linewidth=1.5)
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('Residuals vs Fitted Values', fontsize=12)
ax.grid(True, alpha=0.3)

# 2. Q-Q Plot
ax = axes[0, 1]
from scipy import stats
res_standardized = (full_model.resid - full_model.resid.mean()) / full_model.resid.std()
stats.probplot(res_standardized, dist="norm", plot=ax)
ax.set_title('Normal Q-Q Plot of Residuals', fontsize=12)
ax.grid(True, alpha=0.3)

# 3. Histogram of residuals
ax = axes[1, 0]
ax.hist(full_model.resid, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='white')
x_range = np.linspace(full_model.resid.min(), full_model.resid.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, full_model.resid.mean(), full_model.resid.std()),
        'r-', linewidth=2, label='Normal distribution')
ax.set_xlabel('Residuals', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Residuals', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Distribution of response variable (quality)
ax = axes[1, 1]
quality_counts = data['quality'].value_counts().sort_index()
ax.bar(quality_counts.index, quality_counts.values, color='steelblue', alpha=0.7, edgecolor='white')
ax.set_xlabel('Quality Score', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Response Variable (Quality)', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

# Statistical tests
print("\n─── Diagnostic Tests ───")

# Normality of residuals
stat_sw, p_sw = stats.shapiro(full_model.resid[:5000])  # Shapiro-Wilk (limited to 5000)
stat_jb, p_jb = stats.jarque_bera(full_model.resid)[:2]

print(f"\nNormality of residuals:")
print(f"  Shapiro-Wilk test: W = {stat_sw:.4f}, p = {p_sw:.6e}")
print(f"  Jarque-Bera test:  JB = {stat_jb:.4f}, p = {p_jb:.6e}")
if p_jb < 0.05:
    print(f"  → Residuals are NOT normally distributed (p < 0.05)")

# Heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(full_model.resid, X_full)
print(f"\nHomoscedasticity (Breusch-Pagan test):")
print(f"  BP = {bp_stat:.4f}, p = {bp_pval:.6e}")
if bp_pval < 0.05:
    print(f"  → Evidence of HETEROSCEDASTICITY (p < 0.05)")

# Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(full_model.resid)
print(f"\nAutocorrelation (Durbin-Watson test):")
print(f"  DW = {dw:.4f}")
if dw < 1.5:
    print(f"  → Positive autocorrelation suspected")
elif dw > 2.5:
    print(f"  → Negative autocorrelation suspected")
else:
    print(f"  → No strong evidence of autocorrelation")

# Multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(f"\nMulticollinearity (VIF):")
print(f"  {'Variable':<25} {'VIF':>10}")
print(f"  {'─' * 37}")
vif_data = []
for i, var in enumerate(predictors):
    vif = variance_inflation_factor(X_full.values, i + 1)  # +1 because of constant
    vif_data.append((var, vif))
    flag = " *** HIGH" if vif > 10 else (" * moderate" if vif > 5 else "")
    print(f"  {var:<25} {vif:>10.2f}{flag}")

# R² analysis
print(f"\nModel Explanatory Power:")
print(f"  R² = {full_model.rsquared:.4f} → The model explains only {full_model.rsquared*100:.1f}% of variance")
print(f"  R² adjusted = {full_model.rsquared_adj:.4f}")

# Discussion
print(f"\n\n{'=' * 70}")
print("DISCUSSION: Is linear regression appropriate?")
print("=" * 70)

discussion = """
PROBLEMS IDENTIFIED:

1. DISCRETE RESPONSE VARIABLE
   The response variable 'quality' takes integer values (3-9), making it
   an ordinal/discrete variable. Linear regression assumes a continuous
   response. This fundamental violation means:
   - Predicted values may fall outside the valid range
   - The residuals show a characteristic banded pattern
   → ALTERNATIVE: Ordinal logistic regression or multinomial regression

2. LOW EXPLANATORY POWER (R² ≈ {r2:.1%})
   The model explains only about {r2_pct:.1f}% of the variance in quality.
   This means ~{unexplained:.1f}% of quality variation is unexplained,
   suggesting important predictors are missing or relationships are
   more complex than linear.

3. NON-NORMALITY OF RESIDUALS
   The Jarque-Bera and Shapiro-Wilk tests reject normality of residuals.
   This affects the validity of confidence intervals and hypothesis tests.
   The Q-Q plot shows deviations in the tails.

4. HETEROSCEDASTICITY
   The Breusch-Pagan test indicates non-constant variance of residuals.
   This violates a key OLS assumption and affects the reliability of
   standard errors and t-tests.

5. MULTICOLLINEARITY
   Some predictors show high VIF values (especially density, which is
   physically related to alcohol and residual sugar). This makes
   individual coefficient estimates unstable and hard to interpret.

6. NON-LINEAR RELATIONSHIPS
   As shown in Sub-question 4, polynomial terms are statistically
   significant for some variables, suggesting the true relationships
   are not purely linear.

RECOMMENDED ALTERNATIVES:
- Ordinal logistic regression (accounts for the ordinal nature of quality)
- Random forest or gradient boosting (captures non-linear relationships)
- Poisson or negative binomial regression (for count-like response)
- Ridge/LASSO regression (handles multicollinearity)
""".format(
    r2=full_model.rsquared,
    r2_pct=full_model.rsquared * 100,
    unexplained=(1 - full_model.rsquared) * 100
)

print(discussion)

# Save discussion
with open(OUTPUT_DIR / 'model_critique.txt', 'w') as f:
    f.write("SUB-QUESTION 5: Model Appropriateness Critique\n")
    f.write("=" * 70 + "\n")
    f.write(discussion)

print("\nResults saved to:")
print("  - forward_selection_steps.csv")
print("  - forward_selection_progression.png")
print("  - polynomial_regression.png")
print("  - model_diagnostics.png")
print("  - model_critique.txt")
