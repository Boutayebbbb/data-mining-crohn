"""
======================================================================
ÉTAPE 3 — RÈGLES D'ASSOCIATION SANS CLUSTERING
======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)

print("=" * 70)
print("ÉTAPE 3 — RÈGLES D'ASSOCIATION SANS CLUSTERING")
print("=" * 70)

# ---------------------------------------------------------------------
# 1. CHARGEMENT
# ---------------------------------------------------------------------
df = pd.read_csv(os.path.join(_DIR, "..", "Etape1_Nettoyage", "crohn_clean.csv"))
print(f"\n[1] Dataset chargé : {df.shape[0]} × {df.shape[1]}\n")

# ---------------------------------------------------------------------
# 2. DISCRÉTISATION DES VARIABLES NUMÉRIQUES
# ---------------------------------------------------------------------
print("[2] DISCRÉTISATION DES VARIABLES NUMÉRIQUES")
print("-" * 70)

df_disc = df.copy()

# Age : 4 classes basées sur la médecine
df_disc["Age_cat"] = pd.cut(df["Age"], bins=[0, 25, 40, 60, 100],
                            labels=["Age_Jeune", "Age_Adulte", "Age_Mature", "Age_Senior"])
print(f"  Age : {df_disc['Age_cat'].value_counts().to_dict()}")

# BMI : selon l'OMS
df_disc["BMI_cat"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                            labels=["BMI_Maigreur", "BMI_Normal", "BMI_Surpoids", "BMI_Obesite"])
print(f"  BMI : {df_disc['BMI_cat'].value_counts().to_dict()}")

# CRP : seuil clinique (< 5 = normal, 5-10 = modéré, > 10 = élevé)
df_disc["CRP_cat"] = pd.cut(df["CRP"], bins=[-0.01, 5, 10, 1000],
                            labels=["CRP_Normal", "CRP_Modere", "CRP_Eleve"])
print(f"  CRP : {df_disc['CRP_cat'].value_counts().to_dict()}")

# Hb : seuil clinique d'anémie
df_disc["Hb_cat"] = pd.cut(df["Hb"], bins=[0, 11, 13, 100],
                           labels=["Hb_Anemie", "Hb_Limite", "Hb_Normal"])
print(f"  Hb : {df_disc['Hb_cat'].value_counts().to_dict()}")

# HBI : Harvey-Bradshaw Index (0-4 = rémission, 5-7 = légère, 8-16 = modérée, >16 = sévère)
df_disc["HBI_cat"] = pd.cut(df["HBI"], bins=[-0.01, 4, 7, 16, 100],
                            labels=["HBI_Remission", "HBI_Legere", "HBI_Moderee", "HBI_Severe"])
print(f"  HBI : {df_disc['HBI_cat'].value_counts().to_dict()}")

# Duration : court/moyen/long terme
df_disc["Duration_cat"] = pd.cut(df["Duration"], bins=[-0.01, 2, 10, 100],
                                  labels=["Duration_Recent", "Duration_Moyenne", "Duration_Longue"])
print(f"  Duration : {df_disc['Duration_cat'].value_counts().to_dict()}")

# WBC (globules blancs)
df_disc["WBC_cat"] = pd.cut(df["WBC"], bins=[0, 4, 11, 100],
                            labels=["WBC_Bas", "WBC_Normal", "WBC_Eleve"])
print(f"  WBC : {df_disc['WBC_cat'].value_counts().to_dict()}\n")

# ---------------------------------------------------------------------
# 3. PRÉPARATION DES TRANSACTIONS
# ---------------------------------------------------------------------
print("[3] PRÉPARATION DES TRANSACTIONS")
print("-" * 70)

cat_cols = ["Sex", "Diagnosis", "Treatment", "Surgery", "Smoking",
            "Cannabis", "FlareStatus", "Antibiotics", "Stool",
            "Age_cat", "BMI_cat", "CRP_cat", "Hb_cat", "HBI_cat",
            "Duration_cat", "WBC_cat"]

# Préfixer chaque valeur par le nom de sa colonne pour éviter les ambiguïtés
def make_transaction(row):
    tokens = []
    for c in cat_cols:
        val = row[c]
        if pd.notna(val) and str(val).strip() != "":
            # Si déjà préfixé (cas des _cat), on garde tel quel
            token = str(val)
            if not any(token.startswith(p + "_") for p in ["Age", "BMI", "CRP", "Hb", "HBI", "Duration", "WBC"]):
                token = f"{c}_{val}"
            tokens.append(token)
    return tokens

transactions = df_disc.apply(make_transaction, axis=1).tolist()
print(f"  Nombre de transactions : {len(transactions)}")
print(f"  Exemple de transaction : {transactions[0][:5]} ...\n")

# Encodage one-hot
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
if hasattr(te_array, "toarray"):
    te_array = te_array.toarray()
df_onehot = pd.DataFrame(te_array, columns=te.columns_)
print(f"  Matrice one-hot : {df_onehot.shape[0]} × {df_onehot.shape[1]}\n")

# ---------------------------------------------------------------------
# 4. APRIORI
# ---------------------------------------------------------------------
print("[4] APRIORI — RECHERCHE D'ITEMSETS FRÉQUENTS")
print("-" * 70)

MIN_SUPPORT = 0.10
MIN_CONFIDENCE = 0.6
MIN_LIFT = 1.2

print(f"  min_support    = {MIN_SUPPORT}")
print(f"  min_confidence = {MIN_CONFIDENCE}")
print(f"  min_lift       = {MIN_LIFT}\n")

freq_items = apriori(df_onehot, min_support=MIN_SUPPORT, use_colnames=True, max_len=4)
print(f"  Nombre d'itemsets fréquents : {len(freq_items)}")
freq_items["length"] = freq_items["itemsets"].apply(len)
print(f"  Distribution par taille : {freq_items['length'].value_counts().sort_index().to_dict()}\n")

# ---------------------------------------------------------------------
# 5. GÉNÉRATION DES RÈGLES
# ---------------------------------------------------------------------
print("[5] GÉNÉRATION DES RÈGLES D'ASSOCIATION")
print("-" * 70)

rules = association_rules(freq_items, metric="lift", min_threshold=MIN_LIFT)
rules = rules[rules["confidence"] >= MIN_CONFIDENCE]
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

print(f"  Nombre de règles trouvées : {len(rules)}\n")

# Conversion frozenset -> string pour CSV
rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

# TOP 20 règles
print("  TOP 20 règles (par lift décroissant) :")
print()
top20 = rules.head(20)[["antecedents_str", "consequents_str", "support", "confidence", "lift"]]
top20.columns = ["Antécédent", "Conséquent", "Support", "Confiance", "Lift"]
print(top20.to_string(index=False))

# Sauvegarde
rules_save = rules[["antecedents_str", "consequents_str", "support",
                    "confidence", "lift"]].copy()
rules_save.columns = ["antecedent", "consequent", "support", "confidence", "lift"]
rules_save.to_csv("association_rules_global.csv", index=False)

# ---------------------------------------------------------------------
# 6. VISUALISATIONS
# ---------------------------------------------------------------------
print("\n[6] VISUALISATIONS")
print("-" * 70)

# (a) Scatter Support vs Confiance, taille = Lift
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(rules["support"], rules["confidence"],
                c=rules["lift"], s=rules["lift"]*15,
                cmap='viridis', alpha=0.6, edgecolors='black')
ax.set_xlabel("Support", fontsize=12)
ax.set_ylabel("Confiance", fontsize=12)
ax.set_title("Règles d'association — Support vs Confiance (taille=Lift)",
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
plt.colorbar(sc, label='Lift')
plt.tight_layout()
plt.savefig("01_support_confidence.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_support_confidence.png")

# (b) Top 20 par Lift
fig, ax = plt.subplots(figsize=(13, 8))
top20_plot = rules.head(20).copy()
top20_plot["Règle"] = top20_plot["antecedents_str"] + " ⇒ " + top20_plot["consequents_str"]
top20_plot["Règle"] = top20_plot["Règle"].apply(lambda x: x[:80] + "..." if len(x) > 80 else x)

_cmap = plt.colormaps["viridis"]
ax.barh(range(len(top20_plot)), top20_plot["lift"],
        color=_cmap(top20_plot["lift"] / top20_plot["lift"].max()),
        edgecolor='black')
ax.set_yticks(range(len(top20_plot)))
ax.set_yticklabels(top20_plot["Règle"], fontsize=8)
ax.set_xlabel("Lift", fontsize=11)
ax.set_title("Top 20 règles d'association (par Lift décroissant)",
             fontsize=13, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("02_top20_rules.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_top20_rules.png")

# (c) Distribution Support / Confiance / Lift
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, col, color in zip(axes, ["support", "confidence", "lift"],
                            ["#3498db", "#27ae60", "#e74c3c"]):
    ax.hist(rules[col], bins=30, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(rules[col].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Moyenne={rules[col].mean():.3f}')
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution du {col}", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.suptitle("Distributions des métriques des règles trouvées",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("03_metrics_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_metrics_distribution.png")

# ---------------------------------------------------------------------
# 7. ANALYSE DES RÈGLES MÉDICALEMENT INTÉRESSANTES
# ---------------------------------------------------------------------
print("\n[7] RÈGLES IMPLIQUANT LE STATUT 'FLARE/REMISSION'")
print("-" * 70)

flare_rules = rules[rules["consequents_str"].str.contains("FlareStatus_") |
                    rules["antecedents_str"].str.contains("FlareStatus_")]
print(f"  {len(flare_rules)} règles concernent le statut Flare/Remission")
if len(flare_rules) > 0:
    top_flare = flare_rules.head(10)[["antecedents_str", "consequents_str",
                                       "support", "confidence", "lift"]]
    top_flare.columns = ["Antécédent", "Conséquent", "Sup", "Conf", "Lift"]
    print(top_flare.to_string(index=False))
    flare_rules.to_csv("flare_related_rules.csv", index=False)

print("\n" + "=" * 70)
print("ÉTAPE 3 TERMINÉE AVEC SUCCÈS")
print("=" * 70)
