"""
======================================================================
ÉTAPE 1 — NETTOYAGE DES DONNÉES
======================================================================

PRINCIPES :
  1. SUPPRESSION des colonnes > 50% manquantes (au lieu de les imputer
     artificiellement). Cela évite la création de données fictives.
  2. Imputation conditionnelle pour HBI (score propre à la maladie de
     Crohn : les autres patients sont mis à 0 car non applicable).
  3. Treatment : les contrôles sains ne reçoivent aucun traitement
     ("Aucun") ; seuls les patients sont imputés par le mode.
  4. Documentation rigoureuse de chaque décision avec justification.
  5. Conservation des outliers cliniques réels (pas suppression aveugle).

Auteur : Marouane Boutayeb — Module Data Mining, ESISA 2025-2026
======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)

# ---------------------------------------------------------------------
# 1. CHARGEMENT DU DATASET
# ---------------------------------------------------------------------
INPUT_FILE = os.path.join(_DIR, "..", "Data_Crohn.xlsx")
OUTPUT_CSV = "crohn_clean.csv"

print("=" * 70)
print("ÉTAPE 1 — NETTOYAGE DES DONNÉES")
print("=" * 70)

df = pd.read_excel(INPUT_FILE)
print(f"\n[1] Dataset original chargé : {df.shape[0]} patients × {df.shape[1]} variables\n")

# Renommage pour faciliter l'usage
rename_map = {
    "Anonymous Patient ID": "PatientID",
    "Age (years)": "Age",
    "BMI": "BMI",
    "Diagnosis": "Diagnosis",
    "Montreal Classification": "Montreal",
    "Ethnicity": "Ethnicity",
    "Treatment": "Treatment",
    "Surgical history": "Surgery",
    "Smoking": "Smoking",
    "Cannabis use (Yes/No)": "Cannabis",
    "CRP (mg/L)": "CRP",
    "WBC (10^9/L)": "WBC",
    "Hemoglobin (g/dL)": "Hb",
    "Platelets (10^9/L)": "Platelets",
    "ALT (U/L)": "ALT",
    "AST (U/L)": "AST",
    "Stool sample available": "Stool",
    "Disease duration (years)": "Duration",
    "HBI (Crohn only)": "HBI",
    "Flare/Remission": "FlareStatus",
    "Perianal lesions (type)": "Perianal",
    "Antibiotics Last Month Before Sampling": "Antibiotics",
    "Comorbidities": "Comorbidities",
    "Non-IBD medication": "NonIBDMed",
    "CD62P Positive Cells (Thrombin Low Dose) ": "CD62P",
    "Sex": "Sex",
}
df = df.rename(columns=rename_map)

# ---------------------------------------------------------------------
# 2. ANALYSE DES VALEURS MANQUANTES
# ---------------------------------------------------------------------
print("[2] ANALYSE DES VALEURS MANQUANTES")
print("-" * 70)

missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100
missing_table = pd.DataFrame({
    "Variable": missing_count.index,
    "Manquants": missing_count.values,
    "Pourcentage (%)": missing_pct.values.round(2)
}).sort_values("Pourcentage (%)", ascending=False)

print(missing_table[missing_table["Manquants"] > 0].to_string(index=False))
missing_table.to_csv("missing_values_analysis.csv", index=False)

# Visualisation des manquants
fig, ax = plt.subplots(figsize=(12, 6))
mvars = missing_table[missing_table["Manquants"] > 0]
colors = ['#e74c3c' if p > 50 else ('#f39c12' if p > 20 else '#27ae60') for p in mvars["Pourcentage (%)"]]
bars = ax.barh(mvars["Variable"], mvars["Pourcentage (%)"], color=colors, edgecolor='black')
ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Seuil 50% (suppression)')
ax.axvline(x=20, color='orange', linestyle='--', linewidth=1.5, label='Seuil 20% (attention)')
ax.set_xlabel("Pourcentage de valeurs manquantes (%)", fontsize=11)
ax.set_title("Analyse des valeurs manquantes par variable",
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right')
for bar, pct in zip(bars, mvars["Pourcentage (%)"]):
    ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%", va='center', fontsize=9)
plt.tight_layout()
plt.savefig("01_missing_values.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✓ Graphique sauvegardé : 01_missing_values.png\n")

# ---------------------------------------------------------------------
# 3. SUPPRESSION DES COLONNES > 50% MANQUANTES
# ---------------------------------------------------------------------
print("[3] SUPPRESSION DES COLONNES À FORT TAUX DE MANQUANTS")
print("-" * 70)

# RÈGLE : au-delà de 50% manquants, l'imputation crée des données artificielles.
# Décision : supprimer ces colonnes. Justification consignée.

DROP_THRESHOLD = 50.0
# Justifications cliniques spécifiques pour chaque colonne supprimée
DROP_JUSTIFICATIONS = {
    "CD62P":        "Biomarqueur plaquettaire (82% manquants) — trop lacunaire pour imputation fiable",
    "Perianal":     "Phénotype périnéal (77% manquants) — absent pour la majorité des patients",
    "Comorbidities":"Comorbidités (59% manquants) — imputation biaiserait les analyses downstream",
    "NonIBDMed":    "Médicaments hors-MICI (55% manquants) — non documentés pour >50% des patients",
}
cols_to_drop = missing_table[missing_table["Pourcentage (%)"] > DROP_THRESHOLD]["Variable"].tolist()

print(f"  Colonnes supprimées (> {DROP_THRESHOLD}% manquants) :")
for col in cols_to_drop:
    pct = missing_table.set_index("Variable").loc[col, "Pourcentage (%)"]
    justif = DROP_JUSTIFICATIONS.get(col, f"imputer {pct:.0f}% créerait un dataset artificiellement homogène")
    print(f"    • {col} ({pct:.1f}%) — {justif}")

df_clean = df.drop(columns=cols_to_drop)
print(f"\n  Dataset après suppression : {df_clean.shape[0]} × {df_clean.shape[1]} ({len(cols_to_drop)} colonnes supprimées)\n")

# ---------------------------------------------------------------------
# 4. IMPUTATION CONDITIONNELLE POUR HBI (KNN-Imputer)
# ---------------------------------------------------------------------
print("[4] IMPUTATION INTELLIGENTE DE HBI (KNN-Imputer k=5)")
print("-" * 70)

# HBI = Harvey-Bradshaw Index : score d'activité de la maladie de Crohn.
# Non applicable aux patients non-Crohn → HBI = 0 (cliniquement justifié).
# Pour les Crohn avec HBI manquant : KNN-Imputer (k=5) basé sur variables
# cliniques proches, plus précis que la médiane globale.

n_crohn = (df_clean["Diagnosis"] == "Crohn's disease").sum()
n_other = len(df_clean) - n_crohn

# Patients non-Crohn : HBI = 0
mask_non_crohn = df_clean["Diagnosis"] != "Crohn's disease"
df_clean.loc[mask_non_crohn, "HBI"] = df_clean.loc[mask_non_crohn, "HBI"].fillna(0)
print(f"  • Patients non-Crohn ({n_other}) : HBI = 0 (cliniquement justifié)")

# Patients Crohn : KNN-Imputer k=5
mask_crohn = df_clean["Diagnosis"] == "Crohn's disease"
KNN_FEATURES = ["Age", "BMI", "CRP", "WBC", "Hb", "Platelets", "ALT", "AST", "Duration"]
n_hbi_missing_crohn = df_clean.loc[mask_crohn, "HBI"].isnull().sum()

crohn_subset = df_clean.loc[mask_crohn, KNN_FEATURES + ["HBI"]].copy()
knn_imp = KNNImputer(n_neighbors=5)
crohn_imputed = pd.DataFrame(
    knn_imp.fit_transform(crohn_subset),
    columns=KNN_FEATURES + ["HBI"],
    index=crohn_subset.index
)
df_clean.loc[mask_crohn, "HBI"] = crohn_imputed["HBI"]

print(f"  • Patients Crohn ({n_crohn}) : {n_hbi_missing_crohn} valeurs HBI imputées par KNN-Imputer (k=5)")
print(f"    Variables utilisées : {KNN_FEATURES}")
print(f"    HBI moyen post-imputation (Crohn) : {df_clean.loc[mask_crohn, 'HBI'].mean():.2f}\n")

# ---------------------------------------------------------------------
# 5. IMPUTATION DE TREATMENT (catégoriel, 14% manquant)
# ---------------------------------------------------------------------
print("[5] IMPUTATION DE TREATMENT")
print("-" * 70)

n_treat_missing = df_clean["Treatment"].isnull().sum()
print(f"  • Treatment : {n_treat_missing} manquants ({n_treat_missing / len(df_clean) * 100:.1f}%)")

# Contrôles sains : pas de maladie, donc pas de traitement de MICI.
# Leur imputer le mode (Anti-TNF) serait cliniquement faux.
mask_healthy = df_clean["Diagnosis"] == "Healthy control"
n_healthy_missing = df_clean.loc[mask_healthy, "Treatment"].isnull().sum()
df_clean.loc[mask_healthy, "Treatment"] = df_clean.loc[mask_healthy, "Treatment"].fillna("Aucun")
print(f"    - Contrôles sains ({n_healthy_missing}) : Treatment = 'Aucun' (pas de traitement de MICI)")

# Patients (Crohn / RCH) : imputation par le mode calculé sur les patients uniquement
mode_treatment = df_clean.loc[~mask_healthy, "Treatment"].mode()[0]
n_patients_missing = df_clean.loc[~mask_healthy, "Treatment"].isnull().sum()
df_clean.loc[~mask_healthy, "Treatment"] = df_clean.loc[~mask_healthy, "Treatment"].fillna(mode_treatment)
print(f"    - Patients ({n_patients_missing}) : imputés par le mode des patients = '{mode_treatment}'")
print(f"  • Distribution post-imputation : {df_clean['Treatment'].value_counts().head(3).to_dict()}\n")

# ---------------------------------------------------------------------
# 5.5 IMPUTATION DES VALEURS MANQUANTES RESTANTES
# ---------------------------------------------------------------------
print("[5.5] IMPUTATION DES VALEURS MANQUANTES RESTANTES")
print("-" * 70)

remaining_missing = df_clean.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if len(remaining_missing) > 0:
    print(f"  Variables encore incomplètes :")
    for col in remaining_missing.index:
        n_miss = remaining_missing[col]
        pct_miss = n_miss / len(df_clean) * 100
        if df_clean[col].dtype in [np.float64, np.int64, float, int]:
            med_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(med_val)
            print(f"    • {col} ({n_miss} manquants, {pct_miss:.1f}%) → médiane = {med_val:.2f}")
        else:
            mod_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mod_val)
            print(f"    • {col} ({n_miss} manquants, {pct_miss:.1f}%) → mode = '{mod_val}'")
else:
    print(f"  Aucune valeur manquante restante à traiter.")
print()

# ---------------------------------------------------------------------
# 6. DÉTECTION ET ANALYSE DES DOUBLONS
# ---------------------------------------------------------------------
print("[6] ANALYSE DES DOUBLONS")
print("-" * 70)
n_dup = df_clean.duplicated().sum()
print(f"  Nombre de doublons exacts : {n_dup}")
if n_dup > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"  Doublons supprimés. Nouveau shape : {df_clean.shape}\n")
else:
    print(f"  Aucun doublon — dataset propre.\n")

# ---------------------------------------------------------------------
# 7. STATISTIQUES DESCRIPTIVES COMPLÈTES
# ---------------------------------------------------------------------
print("[7] STATISTIQUES DESCRIPTIVES")
print("-" * 70)

num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
desc = df_clean[num_cols].describe().T
desc["mediane"] = df_clean[num_cols].median()
desc["skewness"] = df_clean[num_cols].skew().round(3)
desc["kurtosis"] = df_clean[num_cols].kurt().round(3)
desc = desc.round(2)

print(desc.to_string())
desc.to_csv("descriptive_stats.csv")
print(f"\n  ✓ Statistiques descriptives sauvegardées : descriptive_stats.csv\n")

# ---------------------------------------------------------------------
# 8. DÉTECTION DES OUTLIERS PAR IQR
# ---------------------------------------------------------------------
print("[8] DÉTECTION DES OUTLIERS (méthode IQR)")
print("-" * 70)

outlier_summary = []
for col in num_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    pct_out = n_out / len(df_clean) * 100
    outlier_summary.append({
        "Variable": col, "Q1": round(Q1, 2), "Q3": round(Q3, 2),
        "IQR": round(IQR, 2), "Borne_inf": round(lower, 2),
        "Borne_sup": round(upper, 2), "N_outliers": n_out,
        "Pct_outliers": round(pct_out, 2)
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))
outlier_df.to_csv("outliers_analysis.csv", index=False)

# DÉCISION : conserver les outliers car ils représentent des cas cliniques réels
# (CRP très élevé chez patients en flare, HBI sévère pour cas avancés, etc.)
print("\n  DÉCISION : Outliers CONSERVÉS — ce sont des observations cliniques réelles")
print("  (pas des erreurs de saisie). Les supprimer biaiserait le modèle médical.\n")

# ---------------------------------------------------------------------
# 9. VISUALISATIONS
# ---------------------------------------------------------------------
print("[9] GÉNÉRATION DES VISUALISATIONS")
print("-" * 70)

# Boxplots
_ncols = 4
_nrows = (len(num_cols) + _ncols - 1) // _ncols
fig, axes = plt.subplots(_nrows, _ncols, figsize=(18, _nrows * 4), squeeze=False)
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].boxplot(df_clean[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel("Valeur")
    axes[i].grid(True, alpha=0.3)
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Boxplots — Distribution des variables numériques",
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("02_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_boxplots.png")

# Histogrammes
fig, axes = plt.subplots(_nrows, _ncols, figsize=(18, _nrows * 4), squeeze=False)
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df_clean[col].dropna(), bins=30,
                 color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[i].axvline(df_clean[col].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Moyenne={df_clean[col].mean():.1f}')
    axes[i].axvline(df_clean[col].median(), color='green', linestyle='--',
                    linewidth=2, label=f'Médiane={df_clean[col].median():.1f}')
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Histogrammes — Distribution des variables numériques",
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("03_histograms.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_histograms.png")

# Matrice de corrélation
plt.figure(figsize=(11, 9))
corr = df_clean[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, mask=mask, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Matrice de corrélation (Pearson)", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("04_correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_correlation_matrix.png")

# Distribution des variables catégorielles (top 6)
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()[:6]
if cat_cols:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        if i >= 6:
            break
        vc = df_clean[col].value_counts().head(8)
        axes[i].bar(range(len(vc)), vc.values, color='#1abc9c', edgecolor='black')
        axes[i].set_xticks(range(len(vc)))
        axes[i].set_xticklabels(vc.index, rotation=30, ha='right', fontsize=9)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_ylabel("Effectif")
        axes[i].grid(True, alpha=0.3, axis='y')
    fig.suptitle("Distribution des principales variables catégorielles",
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig("05_categorical_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_categorical_distributions.png")

# ---------------------------------------------------------------------
# 10. SAUVEGARDE FINALE
# ---------------------------------------------------------------------
print("\n[10] SAUVEGARDE")
print("-" * 70)
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"  ✓ Dataset nettoyé sauvegardé : {OUTPUT_CSV}")
print(f"  ✓ Shape final : {df_clean.shape[0]} patients × {df_clean.shape[1]} variables")
print(f"  ✓ Aucune valeur manquante restante : {df_clean.isnull().sum().sum() == 0}")

print("\n" + "=" * 70)
print("ÉTAPE 1 TERMINÉE AVEC SUCCÈS")
print("=" * 70)
