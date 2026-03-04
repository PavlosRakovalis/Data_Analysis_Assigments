"""
Εργασία στο μάθημα "Ανάλυση Δεδομένων" 2025-26
ΑΕΜ: 6931
Οπτική Παρουσίαση Τελικού Dataset και Περιγραφικά Στατιστικά
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Ρύθμιση style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================
# ΦΟΡΤΩΣΗ ΚΑΘΑΡΙΣΜΕΝΩΝ ΔΕΔΟΜΕΝΩΝ
# ============================================================
AEM = 6931
df = pd.read_csv(f'transformed_wine_data_AEM_{AEM}.csv')

print("=" * 80)
print("ΕΡΓΑΣΙΑ ΑΝΑΛΥΣΗΣ ΔΕΔΟΜΕΝΩΝ 2025-26")
print("ΑΕΜ: 6931")
print("ΟΠΤΙΚΗ ΠΑΡΟΥΣΙΑΣΗ ΤΕΛΙΚΟΥ DATASET")
print("=" * 80)

# Ορισμός μεταβλητών
continuous_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 
                   'residual sugar', 'chlorides', 'free sulfur dioxide',
                   'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
categorical_vars = ['wine_type', 'quality']

print(f"\n📊 Dataset: {df.shape[0]} γραμμές × {df.shape[1]} στήλες")
print(f"📋 Συνεχείς μεταβλητές: {len(continuous_vars)}")
print(f"📋 Κατηγορικές μεταβλητές: {len(categorical_vars)}")

# ============================================================
# 1. ΠΕΡΙΓΡΑΦΙΚΑ ΣΤΑΤΙΣΤΙΚΑ - ΣΥΝΕΧΕΙΣ ΜΕΤΑΒΛΗΤΕΣ
# ============================================================
print("\n" + "=" * 80)
print("ΠΕΡΙΓΡΑΦΙΚΑ ΣΤΑΤΙΣΤΙΚΑ - ΣΥΝΕΧΕΙΣ ΜΕΤΑΒΛΗΤΕΣ")
print("=" * 80)

# Δημιουργία πίνακα περιγραφικών στατιστικών
continuous_stats = pd.DataFrame()

for var in continuous_vars:
    stats = {
        'Μεταβλητή': var,
        'Πλήθος (N)': df[var].count(),
        'Missing': df[var].isna().sum(),
        'Μέση Τιμή': df[var].mean(),
        'Διάμεσος': df[var].median(),
        'Τυπ. Απόκλιση': df[var].std(),
        'Ελάχιστη': df[var].min(),
        'Μέγιστη': df[var].max(),
        'Q1 (25%)': df[var].quantile(0.25),
        'Q3 (75%)': df[var].quantile(0.75),
        'IQR': df[var].quantile(0.75) - df[var].quantile(0.25),
        'Εύρος': df[var].max() - df[var].min(),
        'Ασυμμετρία': df[var].skew(),
        'Κύρτωση': df[var].kurtosis(),
        'Cardinality': df[var].nunique()
    }
    continuous_stats = pd.concat([continuous_stats, pd.DataFrame([stats])], ignore_index=True)

# Εκτύπωση πίνακα
print("\n" + "-" * 80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# Εκτύπωση σε δύο μέρη για καλύτερη αναγνωσιμότητα
print("\n📊 ΒΑΣΙΚΑ ΣΤΑΤΙΣΤΙΚΑ:")
print(continuous_stats[['Μεταβλητή', 'Πλήθος (N)', 'Missing', 'Μέση Τιμή', 
                         'Διάμεσος', 'Τυπ. Απόκλιση', 'Ελάχιστη', 'Μέγιστη']].to_string(index=False))

print("\n📊 ΕΠΙΠΛΕΟΝ ΣΤΑΤΙΣΤΙΚΑ:")
print(continuous_stats[['Μεταβλητή', 'Q1 (25%)', 'Q3 (75%)', 'IQR', 
                         'Εύρος', 'Ασυμμετρία', 'Κύρτωση', 'Cardinality']].to_string(index=False))

# Αποθήκευση σε CSV
continuous_stats.to_csv('descriptive_stats_continuous.csv', index=False, encoding='utf-8-sig')
print("\n💾 Αποθήκευση: descriptive_stats_continuous.csv")

# ============================================================
# 2. ΠΕΡΙΓΡΑΦΙΚΑ ΣΤΑΤΙΣΤΙΚΑ - ΚΑΤΗΓΟΡΙΚΕΣ ΜΕΤΑΒΛΗΤΕΣ
# ============================================================
print("\n" + "=" * 80)
print("ΠΕΡΙΓΡΑΦΙΚΑ ΣΤΑΤΙΣΤΙΚΑ - ΚΑΤΗΓΟΡΙΚΕΣ ΜΕΤΑΒΛΗΤΕΣ")
print("=" * 80)

categorical_stats = pd.DataFrame()

for var in categorical_vars:
    value_counts = df[var].value_counts()
    stats = {
        'Μεταβλητή': var,
        'Πλήθος (N)': df[var].count(),
        'Missing': df[var].isna().sum(),
        'Cardinality': df[var].nunique(),
        'Πιθανότερη Τιμή (Mode)': df[var].mode()[0],
        'Συχνότητα Mode': value_counts.iloc[0],
        'Ποσοστό Mode (%)': (value_counts.iloc[0] / len(df)) * 100,
        'Λιγότερο Συχνή Τιμή': value_counts.index[-1],
        'Συχνότητα Ελάχιστης': value_counts.iloc[-1],
        'Ποσοστό Ελάχιστης (%)': (value_counts.iloc[-1] / len(df)) * 100
    }
    categorical_stats = pd.concat([categorical_stats, pd.DataFrame([stats])], ignore_index=True)

print("\n" + "-" * 80)
print(categorical_stats.to_string(index=False))

# Κατανομές
print("\n📊 ΚΑΤΑΝΟΜΗ WINE_TYPE:")
print(df['wine_type'].value_counts().to_string())
print(f"\nΠοσοστά: {(df['wine_type'].value_counts(normalize=True) * 100).round(2).to_string()}")

print("\n📊 ΚΑΤΑΝΟΜΗ QUALITY:")
print(df['quality'].value_counts().sort_index().to_string())

categorical_stats.to_csv('descriptive_stats_categorical.csv', index=False, encoding='utf-8-sig')
print("\n💾 Αποθήκευση: descriptive_stats_categorical.csv")

# ============================================================
# 3. ΟΠΤΙΚΟΠΟΙΗΣΕΙΣ
# ============================================================
print("\n" + "=" * 80)
print("ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ")
print("=" * 80)

# Χρώματα
colors = {'red': '#c0392b', 'white': '#f39c12', 'all': '#3498db'}

# === 3.1 BOXPLOTS (Τελικό dataset) ===
print("\n📊 1. Boxplots τελικού dataset...")

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
fig.suptitle('Boxplots Συνεχών Μεταβλητών - Τελικό Dataset (ΑΕΜ: 6931)', 
             fontsize=16, fontweight='bold', y=1.02)

for idx, var in enumerate(continuous_vars):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    # Side-by-side boxplots
    df_red = df[df['wine_type'] == 'red'][var]
    df_white = df[df['wine_type'] == 'white'][var]
    
    bp = ax.boxplot([df_red, df_white], patch_artist=True, labels=['Κόκκινο', 'Λευκό'])
    bp['boxes'][0].set_facecolor(colors['red'])
    bp['boxes'][1].set_facecolor(colors['white'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.set_title(var, fontsize=11, fontweight='bold')
    ax.set_ylabel('Τιμή')
    ax.grid(True, alpha=0.3)

axes[2, 3].axis('off')
plt.tight_layout()
plt.savefig('final_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: final_boxplots.png")

# === 3.2 HISTOGRAMS ===
print("\n📊 2. Histograms με κατανομές...")

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
fig.suptitle('Histograms Συνεχών Μεταβλητών - Τελικό Dataset (ΑΕΜ: 6931)', 
             fontsize=16, fontweight='bold', y=1.02)

for idx, var in enumerate(continuous_vars):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    # Histograms με KDE
    ax.hist(df[df['wine_type'] == 'red'][var], bins=30, alpha=0.5, 
            label='Κόκκινο', color=colors['red'], density=True)
    ax.hist(df[df['wine_type'] == 'white'][var], bins=30, alpha=0.5, 
            label='Λευκό', color=colors['white'], density=True)
    
    # KDE lines
    df[df['wine_type'] == 'red'][var].plot(kind='kde', ax=ax, color=colors['red'], linewidth=2)
    df[df['wine_type'] == 'white'][var].plot(kind='kde', ax=ax, color=colors['white'], linewidth=2)
    
    ax.set_title(var, fontsize=11, fontweight='bold')
    ax.set_xlabel('Τιμή')
    ax.set_ylabel('Πυκνότητα')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[2, 3].axis('off')
plt.tight_layout()
plt.savefig('final_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: final_histograms.png")

# === 3.3 CORRELATION HEATMAP ===
print("\n📊 3. Correlation Heatmap...")

# Υπολογισμός correlation matrix
corr_matrix = df[continuous_vars].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Μάσκα για τριγωνικό

heatmap = sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       fmt='.2f', 
                       cmap='RdBu_r',
                       center=0,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                       annot_kws={'size': 9})

plt.title('Correlation Heatmap - Συνεχείς Μεταβλητές (ΑΕΜ: 6931)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: correlation_heatmap.png")

# Εκτύπωση ισχυρών συσχετίσεων
print("\n📊 ΙΣΧΥΡΕΣ ΣΥΣΧΕΤΙΣΕΙΣ (|r| > 0.5):")
print("-" * 50)
strong_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            strong_corr.append({
                'Μεταβλητή 1': corr_matrix.columns[i],
                'Μεταβλητή 2': corr_matrix.columns[j],
                'r': corr_matrix.iloc[i, j]
            })

strong_corr_df = pd.DataFrame(strong_corr).sort_values('r', key=abs, ascending=False)
print(strong_corr_df.to_string(index=False))

# === 3.4 PAIRPLOT (για επιλεγμένες μεταβλητές) ===
print("\n📊 4. Pairplot επιλεγμένων μεταβλητών...")

selected_vars = ['alcohol', 'density', 'volatile acidity', 'quality']
pairplot = sns.pairplot(df, vars=selected_vars, hue='wine_type', 
                         palette={'red': colors['red'], 'white': colors['white']},
                         diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
pairplot.fig.suptitle('Pairplot Επιλεγμένων Μεταβλητών (ΑΕΜ: 6931)', 
                       fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pairplot_selected.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: pairplot_selected.png")

# === 3.5 QUALITY DISTRIBUTION ===
print("\n📊 5. Κατανομή Quality ανά τύπο κρασιού...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart - All
quality_counts = df['quality'].value_counts().sort_index()
axes[0].bar(quality_counts.index, quality_counts.values, color=colors['all'], alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Quality')
axes[0].set_ylabel('Πλήθος')
axes[0].set_title('Κατανομή Quality - Σύνολο', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Bar chart - Red
quality_red = df[df['wine_type'] == 'red']['quality'].value_counts().sort_index()
axes[1].bar(quality_red.index, quality_red.values, color=colors['red'], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Quality')
axes[1].set_ylabel('Πλήθος')
axes[1].set_title('Κατανομή Quality - Κόκκινο Κρασί', fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Bar chart - White
quality_white = df[df['wine_type'] == 'white']['quality'].value_counts().sort_index()
axes[2].bar(quality_white.index, quality_white.values, color=colors['white'], alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Quality')
axes[2].set_ylabel('Πλήθος')
axes[2].set_title('Κατανομή Quality - Λευκό Κρασί', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Κατανομή Βαθμολογίας Ποιότητας (ΑΕΜ: 6931)', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('quality_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: quality_distribution.png")

# === 3.6 VIOLIN PLOTS ===
print("\n📊 6. Violin Plots...")

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
fig.suptitle('Violin Plots Συνεχών Μεταβλητών (ΑΕΜ: 6931)', 
             fontsize=16, fontweight='bold', y=1.02)

for idx, var in enumerate(continuous_vars):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    sns.violinplot(data=df, x='wine_type', y=var, ax=ax, 
                   palette={'red': colors['red'], 'white': colors['white']},
                   inner='box')
    ax.set_title(var, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3)

axes[2, 3].axis('off')
plt.tight_layout()
plt.savefig('violin_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: violin_plots.png")

# === 3.7 SCATTER MATRIX (σημαντικές συσχετίσεις) ===
print("\n📊 7. Scatter plots ισχυρών συσχετίσεων...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Density vs Alcohol (ισχυρή αρνητική)
ax = axes[0, 0]
for wine_type, color in colors.items():
    if wine_type in ['red', 'white']:
        subset = df[df['wine_type'] == wine_type]
        ax.scatter(subset['alcohol'], subset['density'], alpha=0.4, 
                   c=color, label=wine_type.capitalize(), s=15)
ax.set_xlabel('Alcohol')
ax.set_ylabel('Density')
ax.set_title(f'Density vs Alcohol (r={corr_matrix.loc["density", "alcohol"]:.2f})', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Density vs Residual Sugar
ax = axes[0, 1]
for wine_type, color in colors.items():
    if wine_type in ['red', 'white']:
        subset = df[df['wine_type'] == wine_type]
        ax.scatter(subset['residual sugar'], subset['density'], alpha=0.4, 
                   c=color, label=wine_type.capitalize(), s=15)
ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Density')
ax.set_title(f'Density vs Residual Sugar (r={corr_matrix.loc["density", "residual sugar"]:.2f})', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Free vs Total SO2
ax = axes[1, 0]
for wine_type, color in colors.items():
    if wine_type in ['red', 'white']:
        subset = df[df['wine_type'] == wine_type]
        ax.scatter(subset['free sulfur dioxide'], subset['total sulfur dioxide'], alpha=0.4, 
                   c=color, label=wine_type.capitalize(), s=15)
ax.set_xlabel('Free Sulfur Dioxide')
ax.set_ylabel('Total Sulfur Dioxide')
ax.set_title(f'Total vs Free SO₂ (r={corr_matrix.loc["total sulfur dioxide", "free sulfur dioxide"]:.2f})', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Fixed Acidity vs pH
ax = axes[1, 1]
for wine_type, color in colors.items():
    if wine_type in ['red', 'white']:
        subset = df[df['wine_type'] == wine_type]
        ax.scatter(subset['fixed acidity'], subset['pH'], alpha=0.4, 
                   c=color, label=wine_type.capitalize(), s=15)
ax.set_xlabel('Fixed Acidity')
ax.set_ylabel('pH')
ax.set_title(f'pH vs Fixed Acidity (r={corr_matrix.loc["pH", "fixed acidity"]:.2f})', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Scatter Plots Ισχυρών Συσχετίσεων (ΑΕΜ: 6931)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('scatter_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: scatter_correlations.png")

# === 3.8 SUMMARY DASHBOARD ===
print("\n📊 8. Summary Dashboard...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Συνοπτικό Dashboard Τελικού Dataset (ΑΕΜ: 6931)', 
             fontsize=18, fontweight='bold', y=0.98)

# Wine type pie chart
ax1 = fig.add_subplot(3, 4, 1)
wine_counts = df['wine_type'].value_counts()
ax1.pie(wine_counts.values, labels=['Λευκό', 'Κόκκινο'], 
        colors=[colors['white'], colors['red']], autopct='%1.1f%%',
        explode=(0.02, 0.02), startangle=90, textprops={'fontsize': 10})
ax1.set_title('Κατανομή Τύπου Κρασιού', fontweight='bold', fontsize=11)

# Quality distribution
ax2 = fig.add_subplot(3, 4, 2)
quality_all = df['quality'].value_counts().sort_index()
bars = ax2.bar(quality_all.index, quality_all.values, color=colors['all'], 
               alpha=0.7, edgecolor='black')
ax2.set_xlabel('Quality')
ax2.set_ylabel('Πλήθος')
ax2.set_title('Κατανομή Quality', fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.3)

# Mini heatmap
ax3 = fig.add_subplot(3, 4, (3, 4))
top_vars = ['alcohol', 'density', 'volatile acidity', 'residual sugar', 'pH']
mini_corr = df[top_vars].corr()
sns.heatmap(mini_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
            ax=ax3, cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
ax3.set_title('Correlation Matrix (Βασικές Μεταβλητές)', fontweight='bold', fontsize=11)

# Boxplots for key variables
key_vars = ['alcohol', 'volatile acidity', 'residual sugar', 'pH']
for i, var in enumerate(key_vars):
    ax = fig.add_subplot(3, 4, 5 + i)
    bp = ax.boxplot([df[df['wine_type'] == 'red'][var], 
                     df[df['wine_type'] == 'white'][var]], 
                    patch_artist=True, labels=['R', 'W'])
    bp['boxes'][0].set_facecolor(colors['red'])
    bp['boxes'][1].set_facecolor(colors['white'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_title(var, fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)

# Histograms for key variables
for i, var in enumerate(key_vars):
    ax = fig.add_subplot(3, 4, 9 + i)
    ax.hist(df[df['wine_type'] == 'red'][var], bins=20, alpha=0.5, 
            color=colors['red'], label='Red', density=True)
    ax.hist(df[df['wine_type'] == 'white'][var], bins=20, alpha=0.5, 
            color=colors['white'], label='White', density=True)
    ax.set_title(var, fontweight='bold', fontsize=10)
    ax.set_xlabel('Τιμή', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: summary_dashboard.png")

# ============================================================
# ΤΕΛΙΚΗ ΣΥΝΟΨΗ
# ============================================================
print("\n" + "=" * 80)
print("ΤΕΛΙΚΗ ΣΥΝΟΨΗ")
print("=" * 80)

print(f"""
📊 ΓΡΑΦΗΜΑΤΑ ΠΟΥ ΔΗΜΙΟΥΡΓΗΘΗΚΑΝ:
   1. final_boxplots.png - Boxplots συνεχών μεταβλητών
   2. final_histograms.png - Histograms με KDE
   3. correlation_heatmap.png - Correlation matrix heatmap
   4. pairplot_selected.png - Pairplot επιλεγμένων μεταβλητών
   5. quality_distribution.png - Κατανομή quality
   6. violin_plots.png - Violin plots
   7. scatter_correlations.png - Scatter plots ισχυρών συσχετίσεων
   8. summary_dashboard.png - Συνοπτικό dashboard

📊 ΠΙΝΑΚΕΣ ΠΕΡΙΓΡΑΦΙΚΩΝ ΣΤΑΤΙΣΤΙΚΩΝ:
   1. descriptive_stats_continuous.csv - Στατιστικά συνεχών μεταβλητών
   2. descriptive_stats_categorical.csv - Στατιστικά κατηγορικών μεταβλητών

📊 ΒΑΣΙΚΑ ΕΥΡΗΜΑΤΑ:
   • Τελικό dataset: {len(df)} εγγραφές × {len(df.columns)} μεταβλητές
   • Λευκά κρασιά: {len(df[df['wine_type'] == 'white'])} ({len(df[df['wine_type'] == 'white'])/len(df)*100:.1f}%)
   • Κόκκινα κρασιά: {len(df[df['wine_type'] == 'red'])} ({len(df[df['wine_type'] == 'red'])/len(df)*100:.1f}%)
   • Μέση ποιότητα: {df['quality'].mean():.2f}
   • Εύρος ποιότητας: {df['quality'].min():.0f} - {df['quality'].max():.0f}

📊 ΙΣΧΥΡΕΣ ΣΥΣΧΕΤΙΣΕΙΣ (|r| > 0.5):
""")

for _, row in strong_corr_df.iterrows():
    direction = "θετική" if row['r'] > 0 else "αρνητική"
    print(f"   • {row['Μεταβλητή 1']} ↔ {row['Μεταβλητή 2']}: r = {row['r']:.3f} ({direction})")

print("\n" + "=" * 80)
print("ΟΛΟΚΛΗΡΩΣΗ ΟΠΤΙΚΗΣ ΠΑΡΟΥΣΙΑΣΗΣ")
print("=" * 80)
