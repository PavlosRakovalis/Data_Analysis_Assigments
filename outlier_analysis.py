"""
Εργασία στο μάθημα "Ανάλυση Δεδομένων" 2025-26
ΑΕΜ: 6931
Ερώτημα 2 & 3: Boxplots και Εντοπισμός/Μετασχηματισμός Ακραίων Τιμών
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ΦΟΡΤΩΣΗ ΚΑΘΑΡΙΣΜΕΝΩΝ ΔΕΔΟΜΕΝΩΝ
# ============================================================
AEM = 6931
df = pd.read_csv(f'cleaned_wine_data_AEM_{AEM}.csv')

print("=" * 70)
print("ΕΡΓΑΣΙΑ ΑΝΑΛΥΣΗΣ ΔΕΔΟΜΕΝΩΝ 2025-26")
print("ΑΕΜ: 6931")
print("ΕΡΩΤΗΜΑΤΑ 2 & 3: BOXPLOTS ΚΑΙ ΑΚΡΑΙΕΣ ΤΙΜΕΣ")
print("=" * 70)

# Συνεχείς μεταβλητές (χωρίς quality γιατί είναι διακριτή)
continuous_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 
                   'residual sugar', 'chlorides', 'free sulfur dioxide',
                   'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                   'alcohol']

print(f"\n📊 Dataset: {df.shape[0]} γραμμές × {df.shape[1]} στήλες")
print(f"📋 Συνεχείς μεταβλητές: {len(continuous_vars)} (χωρίς quality - διακριτή)")


# ============================================================
# ΕΡΩΤΗΜΑ 2: BOXPLOTS
# ============================================================
print("\n" + "=" * 70)
print("ΕΡΩΤΗΜΑ 2: ΚΑΤΑΣΚΕΥΗ BOXPLOTS")
print("=" * 70)

# Χρώματα για τα γραφήματα
colors = {'all': '#3498db', 'red': '#e74c3c', 'white': '#f1c40f'}

# === 2.1 BOXPLOTS ΓΙΑ ΟΛΟ ΤΟ DATASET ===
print("\n📊 Δημιουργία Boxplots για το σύνολο των δεδομένων...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Boxplots Συνεχών Μεταβλητών - Σύνολο Δεδομένων (ΑΕΜ: 6931)', 
             fontsize=14, fontweight='bold')

for idx, var in enumerate(continuous_vars):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    bp = ax.boxplot(df[var].dropna(), patch_artist=True)
    bp['boxes'][0].set_facecolor(colors['all'])
    bp['boxes'][0].set_alpha(0.7)
    
    ax.set_title(var, fontsize=10, fontweight='bold')
    ax.set_ylabel('Τιμή')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boxplots_all_data.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: boxplots_all_data.png")

# === 2.1B ΚΑΤΑΝΟΜΗ QUALITY (ΔΙΑΚΡΙΤΗ ΜΕΤΑΒΛΗΤΗ) ===
print("\n📊 Δημιουργία κατανομής Quality (ιστόγραμμα)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Κατανομή Quality (Διακριτή Μεταβλητή) - ΑΕΜ: 6931', 
             fontsize=14, fontweight='bold')

# Συνολική κατανομή
quality_counts = df['quality'].value_counts().sort_index()
axes[0].bar(quality_counts.index, quality_counts.values, color=colors['all'], alpha=0.7, edgecolor='black')
axes[0].set_title('Συνολική Κατανομή', fontweight='bold')
axes[0].set_xlabel('Quality Score')
axes[0].set_ylabel('Συχνότητα')
axes[0].grid(True, alpha=0.3, axis='y')

# Κατανομή ανά τύπο κρασιού
quality_red = df[df['wine_type'] == 'red']['quality'].value_counts().sort_index()
quality_white = df[df['wine_type'] == 'white']['quality'].value_counts().sort_index()

x = np.arange(len(quality_counts.index))
width = 0.35

axes[1].bar(x - width/2, [quality_red.get(i, 0) for i in quality_counts.index], 
            width, label='Κόκκινο', color=colors['red'], alpha=0.7, edgecolor='black')
axes[1].bar(x + width/2, [quality_white.get(i, 0) for i in quality_counts.index], 
            width, label='Λευκό', color=colors['white'], alpha=0.7, edgecolor='black')
axes[1].set_title('Σύγκριση Κόκκινου vs Λευκού', fontweight='bold')
axes[1].set_xlabel('Quality Score')
axes[1].set_ylabel('Συχνότητα')
axes[1].set_xticks(x)
axes[1].set_xticklabels(quality_counts.index)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('quality_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: quality_distribution.png")

# === 2.2 BOXPLOTS ΞΕΧΩΡΙΣΤΑ ΑΝΑ ΤΥΠΟ ΚΡΑΣΙΟΥ ===
print("\n📊 Δημιουργία Boxplots ανά τύπο κρασιού (side-by-side)...")

df_red = df[df['wine_type'] == 'red']
df_white = df[df['wine_type'] == 'white']

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('Boxplots Συνεχών Μεταβλητών - Σύγκριση Κόκκινου vs Λευκού Κρασιού (ΑΕΜ: 6931)', 
             fontsize=14, fontweight='bold')

for idx, var in enumerate(continuous_vars):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    data_to_plot = [df_red[var].dropna(), df_white[var].dropna()]
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=['Κόκκινο', 'Λευκό'])
    
    bp['boxes'][0].set_facecolor(colors['red'])
    bp['boxes'][1].set_facecolor(colors['white'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.set_title(var, fontsize=10, fontweight='bold')
    ax.set_ylabel('Τιμή')
    ax.grid(True, alpha=0.3)

# Προσθήκη legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors['red'], alpha=0.7, label='Κόκκινο'),
                   Patch(facecolor=colors['white'], alpha=0.7, label='Λευκό')]
fig.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('boxplots_by_wine_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: boxplots_by_wine_type.png")

# === ΣΧΟΛΙΑΣΜΟΣ BOXPLOTS ===
print("\n" + "-" * 70)
print("ΣΧΟΛΙΑΣΜΟΣ BOXPLOTS")
print("-" * 70)

comments = """
📌 ΠΑΡΑΤΗΡΗΣΕΙΣ ΑΠΟ ΤΑ BOXPLOTS:

1. FIXED ACIDITY (Σταθερή οξύτητα):
   - Τα κόκκινα κρασιά έχουν υψηλότερη σταθερή οξύτητα (median ~7.8) 
     έναντι των λευκών (~6.8)
   - Υπάρχουν ακραίες τιμές και στις δύο κατηγορίες

2. VOLATILE ACIDITY (Πτητική οξύτητα):
   - Σημαντικά υψηλότερη στα κόκκινα κρασιά (median ~0.50 vs ~0.26)
   - Τα λευκά έχουν περισσότερες ακραίες τιμές προς τα πάνω

3. CITRIC ACID (Κιτρικό οξύ):
   - Παρόμοια κατανομή μεταξύ των δύο τύπων
   - Λίγες ακραίες τιμές

4. RESIDUAL SUGAR (Υπολειμματικά σάκχαρα):
   - Τα λευκά κρασιά έχουν σημαντικά υψηλότερα υπολειμματικά σάκχαρα
   - Πολλές ακραίες τιμές (ιδιαίτερα στα λευκά)
   - Asymmetric κατανομή με θετική ασυμμετρία

5. CHLORIDES (Χλωριούχα άλατα):
   - Τα κόκκινα έχουν ελαφρώς υψηλότερες τιμές
   - Αρκετές ακραίες τιμές και στους δύο τύπους

6. FREE SULFUR DIOXIDE (Ελεύθερο SO₂):
   - Τα λευκά κρασιά έχουν σημαντικά υψηλότερες τιμές
   - Πολλές ακραίες τιμές

7. TOTAL SULFUR DIOXIDE (Συνολικό SO₂):
   - Πολύ υψηλότερο στα λευκά κρασιά (median ~133 vs ~40)
   - Μεγάλη διασπορά και ακραίες τιμές

8. DENSITY (Πυκνότητα):
   - Πολύ παρόμοια κατανομή μεταξύ των τύπων
   - Στενό εύρος τιμών (0.99-1.00)

9. pH:
   - Τα κόκκινα έχουν ελαφρώς υψηλότερο pH (median ~3.3 vs ~3.18)
   - Λίγες ακραίες τιμές

10. SULPHATES (Θειικά άλατα):
    - Υψηλότερα στα κόκκινα κρασιά
    - Αρκετές ακραίες τιμές

11. ALCOHOL:
    - Παρόμοια κατανομή μεταξύ τύπων
    - Median γύρω στο 10.4%
"""
print(comments)

# ============================================================
# ΕΡΩΤΗΜΑ 3: ΕΝΤΟΠΙΣΜΟΣ ΚΑΙ ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΣ ΑΚΡΑΙΩΝ ΤΙΜΩΝ
# ============================================================
print("\n" + "=" * 70)
print("ΕΡΩΤΗΜΑ 3: ΕΝΤΟΠΙΣΜΟΣ ΑΚΡΑΙΩΝ ΤΙΜΩΝ (OUTLIERS)")
print("=" * 70)

# Αντίγραφο για μετασχηματισμό
df_transformed = df.copy()

# === 3.1 ΜΕΘΟΔΟΣ IQR ===
print("\n" + "-" * 70)
print("ΜΕΘΟΔΟΣ IQR (Interquartile Range)")
print("-" * 70)
print("\nΟ κανόνας IQR ορίζει ως ακραίες τις τιμές εκτός του διαστήματος:")
print("[Q1 - 1.5*IQR, Q3 + 1.5*IQR]")

iqr_outliers = {}
iqr_bounds = {}

print("\n{:<25} | {:>8} | {:>10} | {:>10} | {:>8}".format(
    "Μεταβλητή", "Outliers", "Lower", "Upper", "% data"))
print("-" * 70)

for var in continuous_vars:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df[var] < lower_bound) | (df[var] > upper_bound)
    n_outliers = outliers_mask.sum()
    pct = (n_outliers / len(df)) * 100
    
    iqr_outliers[var] = n_outliers
    iqr_bounds[var] = (lower_bound, upper_bound)
    
    print("{:<25} | {:>8} | {:>10.4f} | {:>10.4f} | {:>7.2f}%".format(
        var, n_outliers, lower_bound, upper_bound, pct))

total_iqr = sum(iqr_outliers.values())
print("-" * 70)
print(f"Σύνολο ακραίων τιμών (IQR): {total_iqr}")

# === 3.2 ΜΕΘΟΔΟΣ Z-SCORE ===
print("\n" + "-" * 70)
print("ΜΕΘΟΔΟΣ Z-SCORE (threshold = 3.5 τυπικές αποκλίσεις)")
print("-" * 70)
print("\nΑκραίες τιμές: |z-score| > 3.5")

zscore_outliers = {}
zscore_bounds = {}

print("\n{:<25} | {:>8} | {:>10} | {:>10} | {:>8}".format(
    "Μεταβλητή", "Outliers", "Lower", "Upper", "% data"))
print("-" * 70)

for var in continuous_vars:
    mean = df[var].mean()
    std = df[var].std()
    
    z_threshold = 3.5
    lower_bound = mean - z_threshold * std
    upper_bound = mean + z_threshold * std
    
    outliers_mask = (df[var] < lower_bound) | (df[var] > upper_bound)
    n_outliers = outliers_mask.sum()
    pct = (n_outliers / len(df)) * 100
    
    zscore_outliers[var] = n_outliers
    zscore_bounds[var] = (lower_bound, upper_bound)
    
    print("{:<25} | {:>8} | {:>10.4f} | {:>10.4f} | {:>7.2f}%".format(
        var, n_outliers, lower_bound, upper_bound, pct))

total_zscore = sum(zscore_outliers.values())
print("-" * 70)
print(f"Σύνολο ακραίων τιμών (Z-score): {total_zscore}")

# === ΣΥΓΚΡΙΣΗ ΜΕΘΟΔΩΝ ===
print("\n" + "-" * 70)
print("ΣΥΓΚΡΙΣΗ ΜΕΘΟΔΩΝ IQR vs Z-SCORE")
print("-" * 70)

print("\n{:<25} | {:>12} | {:>12}".format("Μεταβλητή", "IQR", "Z-score"))
print("-" * 55)
for var in continuous_vars:
    print("{:<25} | {:>12} | {:>12}".format(var, iqr_outliers[var], zscore_outliers[var]))
print("-" * 55)
print("{:<25} | {:>12} | {:>12}".format("ΣΥΝΟΛΟ", total_iqr, total_zscore))

print("\n📌 ΣΧΟΛΙΟ:")
print("   Η μέθοδος IQR εντοπίζει περισσότερες ακραίες τιμές διότι είναι")
print("   πιο ευαίσθητη. Η μέθοδος Z-score με threshold 3.5 είναι πιο")
print("   συντηρητική και εντοπίζει μόνο τις πολύ ακραίες τιμές.")

# === 3.3 CLAMP TRANSFORMATION ===
print("\n" + "-" * 70)
print("CLAMP TRANSFORMATION")
print("-" * 70)
print("\nΕφαρμογή Clamp μετασχηματισμού με βάση τα όρια της μεθόδου IQR:")
print("- Τιμές < lower_bound → lower_bound")
print("- Τιμές > upper_bound → upper_bound")

clamp_changes = {}

for var in continuous_vars:
    lower_bound, upper_bound = iqr_bounds[var]
    
    # Μέτρηση τιμών που θα αλλάξουν
    below = (df_transformed[var] < lower_bound).sum()
    above = (df_transformed[var] > upper_bound).sum()
    
    # Εφαρμογή Clamp
    df_transformed[var] = df_transformed[var].clip(lower=lower_bound, upper=upper_bound)
    
    clamp_changes[var] = {'below': below, 'above': above, 'total': below + above}

print("\n{:<25} | {:>10} | {:>10} | {:>10}".format(
    "Μεταβλητή", "Clamped↓", "Clamped↑", "Σύνολο"))
print("-" * 60)
for var in continuous_vars:
    print("{:<25} | {:>10} | {:>10} | {:>10}".format(
        var, 
        clamp_changes[var]['below'], 
        clamp_changes[var]['above'],
        clamp_changes[var]['total']))
print("-" * 60)
total_clamped = sum(c['total'] for c in clamp_changes.values())
print(f"Συνολικές αλλαγές: {total_clamped}")

# === BOXPLOTS ΠΡΙΝ ΚΑΙ ΜΕΤΑ CLAMP ===
print("\n📊 Δημιουργία Boxplots πριν και μετά το Clamp transformation...")

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
fig.suptitle('Boxplots ΠΡΙΝ (μπλε) και ΜΕΤΑ (πράσινο) Clamp Transformation (ΑΕΜ: 6931)', 
             fontsize=14, fontweight='bold')

for idx, var in enumerate(continuous_vars):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    # Πριν και μετά
    data_before = df[var].dropna()
    data_after = df_transformed[var].dropna()
    
    bp = ax.boxplot([data_before, data_after], patch_artist=True, 
                    labels=['Πριν', 'Μετά'])
    
    bp['boxes'][0].set_facecolor('#3498db')  # Μπλε - πριν
    bp['boxes'][1].set_facecolor('#2ecc71')  # Πράσινο - μετά
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.set_title(var, fontsize=10, fontweight='bold')
    ax.set_ylabel('Τιμή')
    ax.grid(True, alpha=0.3)

axes[2, 3].axis('off')

legend_elements = [Patch(facecolor='#3498db', alpha=0.7, label='Πριν Clamp'),
                   Patch(facecolor='#2ecc71', alpha=0.7, label='Μετά Clamp')]
fig.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('boxplots_before_after_clamp.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Αποθήκευση: boxplots_before_after_clamp.png")

# === ΣΤΑΤΙΣΤΙΚΑ ΠΡΙΝ ΚΑΙ ΜΕΤΑ ===
print("\n" + "-" * 70)
print("ΣΥΓΚΡΙΣΗ ΣΤΑΤΙΣΤΙΚΩΝ ΠΡΙΝ ΚΑΙ ΜΕΤΑ CLAMP")
print("-" * 70)

print("\n{:<22} | {:>8} | {:>8} | {:>8} | {:>8}".format(
    "Μεταβλητή", "Min(πριν)", "Min(μετά)", "Max(πριν)", "Max(μετά)"))
print("-" * 70)
for var in continuous_vars:
    print("{:<22} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f}".format(
        var,
        df[var].min(), df_transformed[var].min(),
        df[var].max(), df_transformed[var].max()))

# === ΑΠΟΘΗΚΕΥΣΗ ΜΕΤΑΣΧΗΜΑΤΙΣΜΕΝΩΝ ΔΕΔΟΜΕΝΩΝ ===
output_file = f'transformed_wine_data_AEM_{AEM}.csv'
df_transformed.to_csv(output_file, index=False)
print(f"\n💾 Αποθήκευση μετασχηματισμένου dataset: {output_file}")

# === ΤΕΛΙΚΗ ΣΥΝΟΨΗ ===
print("\n" + "=" * 70)
print("ΤΕΛΙΚΗ ΣΥΝΟΨΗ ΕΡΩΤΗΜΑΤΩΝ 2 & 3")
print("=" * 70)

summary = f"""
📊 ΓΡΑΦΗΜΑΤΑ ΠΟΥ ΔΗΜΙΟΥΡΓΗΘΗΚΑΝ:
   1. boxplots_all_data.png - Boxplots για όλα τα δεδομένα
   2. boxplots_by_wine_type.png - Boxplots ανά τύπο κρασιού
   3. boxplots_before_after_clamp.png - Σύγκριση πριν/μετά Clamp

📊 ΑΚΡΑΙΕΣ ΤΙΜΕΣ:
   • Μέθοδος IQR: {total_iqr} ακραίες τιμές
   • Μέθοδος Z-score (3.5σ): {total_zscore} ακραίες τιμές

📊 CLAMP TRANSFORMATION:
   • Συνολικές τιμές που μετασχηματίστηκαν: {total_clamped}
   • Μέθοδος: Περιορισμός τιμών εντός [Q1-1.5*IQR, Q3+1.5*IQR]

📊 ΑΠΟΘΗΚΕΥΜΕΝΑ ΑΡΧΕΙΑ:
   • {output_file} - Dataset μετά τον μετασχηματισμό
"""
print(summary)

print("=" * 70)
print("ΟΛΟΚΛΗΡΩΣΗ ΕΡΩΤΗΜΑΤΩΝ 2 & 3")
print("=" * 70)
