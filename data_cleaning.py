"""
Εργασία στο μάθημα "Ανάλυση Δεδομένων" 2025-26
ΑΕΜ: 6931
Ερώτημα 1: Καθαρισμός Δεδομένων
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ
# ============================================================
print("=" * 70)
print("ΕΡΓΑΣΙΑ ΑΝΑΛΥΣΗΣ ΔΕΔΟΜΕΝΩΝ 2025-26")
print("ΑΕΜ: 6931")
print("=" * 70)

# Φόρτωση του αρχικού dataset
df_original = pd.read_csv('Data Analysis_2026 1st Case_Data.csv')

print(f"\n📊 ΑΡΧΙΚΟ DATASET")
print(f"   Αριθμός γραμμών: {df_original.shape[0]}")
print(f"   Αριθμός στηλών: {df_original.shape[1]}")

# Αφαίρεση extra στηλών (Unnamed columns)
unnamed_cols = [col for col in df_original.columns if 'Unnamed' in col]
if unnamed_cols:
    print(f"\n⚠️  Αφαίρεση περιττών στηλών: {unnamed_cols}")
    df_original = df_original.drop(columns=unnamed_cols)

print(f"\n📋 ΣΤΗΛΕΣ DATASET: {list(df_original.columns)}")

# ============================================================
# ΔΕΙΓΜΑΤΟΛΗΨΙΑ 80% με seed = AEM (6931)
# ============================================================
AEM = 6931
SAMPLE_FRACTION = 0.80

df = df_original.sample(frac=SAMPLE_FRACTION, random_state=AEM)
print(f"\n🎲 ΔΕΙΓΜΑΤΟΛΗΨΙΑ")
print(f"   Seed (ΑΕΜ): {AEM}")
print(f"   Ποσοστό δείγματος: {SAMPLE_FRACTION * 100}%")
print(f"   Μέγεθος δείγματος: {len(df)} γραμμές")

# Reset index για ευκολία
df = df.reset_index(drop=True)

# ============================================================
# ΑΝΑΛΥΣΗ ΠΡΟΒΛΗΜΑΤΩΝ
# ============================================================
print("\n" + "=" * 70)
print("ΑΝΑΛΥΣΗ ΠΡΟΒΛΗΜΑΤΩΝ ΔΕΔΟΜΕΝΩΝ")
print("=" * 70)

# Dictionary για αποθήκευση προβλημάτων
problems_report = {}

# Αναμενόμενες στήλες (αριθμητικές)
numeric_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 
                   'residual sugar', 'chlorides', 'free sulfur dioxide',
                   'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                   'alcohol', 'quality']

categorical_columns = ['wine_type']

# ============================================================
# 1. ΕΝΤΟΠΙΣΜΟΣ ΤΥΠΟΓΡΑΦΙΚΩΝ ΛΑΘΩΝ ΣΤΗ ΣΤΗΛΗ wine_type
# ============================================================
print("\n1️⃣  ΕΛΕΓΧΟΣ ΤΥΠΟΓΡΑΦΙΚΩΝ ΛΑΘΩΝ (wine_type)")
print("-" * 50)

wine_type_values = df['wine_type'].dropna().unique()
print(f"   Μοναδικές τιμές: {wine_type_values}")

# Αναμενόμενες τιμές
expected_wine_types = ['red', 'white']
typos_wine = []

for val in wine_type_values:
    if val not in expected_wine_types:
        typos_wine.append(val)
        count = (df['wine_type'] == val).sum()
        print(f"   ❌ Τυπογραφικό λάθος: '{val}' ({count} εμφανίσεις)")

problems_report['Τυπογραφικά λάθη wine_type'] = len(typos_wine)

# Διόρθωση τυπογραφικών λαθών (βάσει των πραγματικών λαθών που εντοπίστηκαν)
wine_type_corrections = {
    'Redd ': 'red',
    'Redd': 'red',
    'r3d': 'red', 
    'WHITTE': 'white',
    'whit': 'white',
    'whate': 'white',
    '999': np.nan,      # Μη έγκυρη τιμή → NaN
    '9999': np.nan,     # Μη έγκυρη τιμή → NaN
    '-999': np.nan,     # Μη έγκυρη τιμή → NaN
    '?': np.nan,        # Μη έγκυρη τιμή → NaN
}

corrections_made = 0
for wrong, correct in wine_type_corrections.items():
    mask = df['wine_type'] == wrong
    if mask.sum() > 0:
        corrections_made += mask.sum()
        df.loc[mask, 'wine_type'] = correct
        print(f"   ✅ Διόρθωση: '{wrong}' → '{correct}' ({mask.sum()} εγγραφές)")

print(f"\n   Συνολικές διορθώσεις τυπογραφικών: {corrections_made}")

# ============================================================
# 2. ΕΝΤΟΠΙΣΜΟΣ ΜΗ ΕΓΚΥΡΩΝ ΤΙΜΩΝ (OUTLIERS / ΑΔΥΝΑΤΕΣ ΤΙΜΕΣ)
# ============================================================
print("\n2️⃣  ΕΛΕΓΧΟΣ ΜΗ ΕΓΚΥΡΩΝ ΤΙΜΩΝ (ΑΡΙΘΜΗΤΙΚΕΣ ΣΤΗΛΕΣ)")
print("-" * 50)

# Μετατροπή σε αριθμητικές τιμές
invalid_values_count = {}

for col in numeric_columns:
    original_values = df[col].copy()
    # Μετατροπή σε numeric, αντικατάσταση μη αριθμητικών με NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Μέτρηση τιμών που έγιναν NaN λόγω μη έγκυρου format
    newly_nan = df[col].isna() & original_values.notna()
    if newly_nan.sum() > 0:
        invalid_values_count[col] = newly_nan.sum()
        print(f"   ❌ {col}: {newly_nan.sum()} μη αριθμητικές τιμές")

problems_report['Μη αριθμητικές τιμές'] = sum(invalid_values_count.values()) if invalid_values_count else 0

# Έλεγχος για αδύνατες τιμές (π.χ. αρνητικές τιμές, τιμές 999, κλπ)
print("\n   Έλεγχος ακραίων/αδύνατων τιμών:")

impossible_values = {}

# Ορισμός λογικών ορίων για κάθε μεταβλητή
value_limits = {
    'fixed acidity': (0, 20),       # g/L
    'volatile acidity': (0, 5),      # g/L
    'citric acid': (0, 3),           # g/L
    'residual sugar': (0, 100),      # g/L
    'chlorides': (0, 1),             # g/L
    'free sulfur dioxide': (0, 300), # mg/L
    'total sulfur dioxide': (0, 500),# mg/L
    'density': (0.9, 1.1),           # g/cm³
    'pH': (2, 5),                    # pH scale
    'sulphates': (0, 3),             # g/L
    'alcohol': (5, 20),              # % vol
    'quality': (0, 10)               # scale
}

for col, (min_val, max_val) in value_limits.items():
    # Εντοπισμός τιμών εκτός ορίων
    out_of_range = (df[col] < min_val) | (df[col] > max_val)
    out_of_range = out_of_range & df[col].notna()  # Αγνόηση NaN
    
    if out_of_range.sum() > 0:
        impossible_values[col] = out_of_range.sum()
        invalid_examples = df.loc[out_of_range, col].head(5).tolist()
        print(f"   ❌ {col}: {out_of_range.sum()} εκτός ορίων [{min_val}, {max_val}]")
        print(f"      Παραδείγματα: {invalid_examples}")
        
        # Αντικατάσταση αδύνατων τιμών με NaN
        df.loc[out_of_range, col] = np.nan

problems_report['Αδύνατες/ακραίες τιμές'] = sum(impossible_values.values()) if impossible_values else 0

# ============================================================
# 3. ΕΝΤΟΠΙΣΜΟΣ ΔΙΠΛΟΕΓΓΡΑΦΩΝ
# ============================================================
print("\n3️⃣  ΕΛΕΓΧΟΣ ΔΙΠΛΟΕΓΓΡΑΦΩΝ")
print("-" * 50)

# Πλήρεις διπλότυπες γραμμές
duplicates = df.duplicated()
num_duplicates = duplicates.sum()
print(f"   Πλήρεις διπλοεγγραφές: {num_duplicates}")

if num_duplicates > 0:
    print(f"   ✅ Αφαίρεση {num_duplicates} διπλοεγγραφών...")
    df = df.drop_duplicates().reset_index(drop=True)

problems_report['Διπλοεγγραφές'] = num_duplicates

# ============================================================
# 4. ΕΝΤΟΠΙΣΜΟΣ MISSING VALUES
# ============================================================
print("\n4️⃣  ΕΛΕΓΧΟΣ ΤΙΜΩΝ ΠΟΥ ΛΕΙΠΟΥΝ (MISSING VALUES)")
print("-" * 50)

missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

print("\n   Στήλη                      | Missing | Ποσοστό")
print("   " + "-" * 55)
for col in df.columns:
    if missing_values[col] > 0:
        print(f"   {col:<28} | {missing_values[col]:>7} | {missing_percent[col]:>6.2f}%")

total_missing = missing_values.sum()
print(f"\n   Συνολικές missing values: {total_missing}")

problems_report['Missing values (πριν αντιμετώπιση)'] = total_missing

# ============================================================
# 5. ΑΝΤΙΜΕΤΩΠΙΣΗ MISSING VALUES
# ============================================================
print("\n5️⃣  ΑΝΤΙΜΕΤΩΠΙΣΗ MISSING VALUES")
print("-" * 50)

# Στρατηγική: 
# - Για αριθμητικές μεταβλητές: Αντικατάσταση με τη διάμεσο ανά τύπο κρασιού
# - Για wine_type: Αντικατάσταση με την πιο συχνή τιμή

print("\n   Στρατηγική αντιμετώπισης:")
print("   • Αριθμητικές μεταβλητές: Αντικατάσταση με διάμεσο (median) ανά τύπο κρασιού")
print("   • wine_type: Αντικατάσταση με την πλειοψηφική τιμή (mode)")

# Αντιμετώπιση wine_type πρώτα
wine_type_missing = df['wine_type'].isnull().sum()
if wine_type_missing > 0:
    mode_wine = df['wine_type'].mode()[0]
    df['wine_type'] = df['wine_type'].fillna(mode_wine)
    print(f"\n   ✅ wine_type: {wine_type_missing} missing → συμπλήρωση με '{mode_wine}'")

# Αντιμετώπιση αριθμητικών στηλών ανά τύπο κρασιού
for col in numeric_columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        # Υπολογισμός διαμέσου ανά τύπο κρασιού
        for wine_type in ['red', 'white']:
            mask = (df['wine_type'] == wine_type) & (df[col].isnull())
            if mask.sum() > 0:
                median_val = df.loc[df['wine_type'] == wine_type, col].median()
                df.loc[mask, col] = median_val
                print(f"   ✅ {col} ({wine_type}): {mask.sum()} missing → median={median_val:.3f}")

# Τελικός έλεγχος missing values
final_missing = df.isnull().sum().sum()
print(f"\n   Εναπομείναντα missing values: {final_missing}")
problems_report['Missing values (μετά αντιμετώπιση)'] = final_missing

# ============================================================
# ΣΥΝΟΨΗ ΠΡΟΒΛΗΜΑΤΩΝ
# ============================================================
print("\n" + "=" * 70)
print("ΣΥΝΟΨΗ ΠΡΟΒΛΗΜΑΤΩΝ ΚΑΙ ΑΝΤΙΜΕΤΩΠΙΣΗΣ")
print("=" * 70)

print("\n📊 ΠΙΝΑΚΑΣ ΠΡΟΒΛΗΜΑΤΩΝ:")
print("-" * 60)
print(f"{'Τύπος Προβλήματος':<45} | {'Πλήθος':>10}")
print("-" * 60)
for problem, count in problems_report.items():
    print(f"{problem:<45} | {count:>10}")
print("-" * 60)

# ============================================================
# ΤΕΛΙΚΟ DATASET
# ============================================================
print("\n" + "=" * 70)
print("ΤΕΛΙΚΟ ΚΑΘΑΡΙΣΜΕΝΟ DATASET")
print("=" * 70)

print(f"\n📊 Τελικό μέγεθος: {df.shape[0]} γραμμές × {df.shape[1]} στήλες")
print(f"\n📋 Στατιστικά τελικού dataset:")
print(df.describe().round(3).to_string())

print(f"\n📋 Τύποι δεδομένων:")
print(df.dtypes.to_string())

print(f"\n📋 Κατανομή wine_type:")
print(df['wine_type'].value_counts().to_string())

# ============================================================
# ΑΠΟΘΗΚΕΥΣΗ ΚΑΘΑΡΙΣΜΕΝΟΥ DATASET
# ============================================================
output_file = f'cleaned_wine_data_AEM_{AEM}.csv'
df.to_csv(output_file, index=False)
print(f"\n💾 Αποθήκευση καθαρισμένου dataset: {output_file}")

print("\n" + "=" * 70)
print("ΟΛΟΚΛΗΡΩΣΗ ΚΑΘΑΡΙΣΜΟΥ ΔΕΔΟΜΕΝΩΝ")
print("=" * 70)
