"""
======================================================================
ÉTAPE 4 — RÈGLES D'ASSOCIATION AVEC CLUSTERING
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
print("ÉTAPE 4 — RÈGLES D'ASSOCIATION AVEC CLUSTERING")
print("=" * 70)

# ---------------------------------------------------------------------
# 1. CHARGEMENT DU DATASET CLUSTERISÉ
# ---------------------------------------------------------------------
df = pd.read_csv(os.path.join(_DIR, "..", "Etape2_Clustering", "crohn_clustered.csv"))
print(f"\n[1] Dataset clusterisé chargé : {df.shape}")
print(f"    Distribution clusters K-Means : {df['Cluster_KMeans'].value_counts().sort_index().to_dict()}\n")

# ---------------------------------------------------------------------
# 2. DISCRÉTISATION (idem étape 3)
# ---------------------------------------------------------------------
df["Age_cat"] = pd.cut(df["Age"], bins=[0, 25, 40, 60, 100],
                       labels=["Age_Jeune", "Age_Adulte", "Age_Mature", "Age_Senior"])
df["BMI_cat"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                       labels=["BMI_Maigreur", "BMI_Normal", "BMI_Surpoids", "BMI_Obesite"])
df["CRP_cat"] = pd.cut(df["CRP"], bins=[-0.01, 5, 10, 1000],
                       labels=["CRP_Normal", "CRP_Modere", "CRP_Eleve"])
df["Hb_cat"] = pd.cut(df["Hb"], bins=[0, 11, 13, 100],
                      labels=["Hb_Anemie", "Hb_Limite", "Hb_Normal"])
df["HBI_cat"] = pd.cut(df["HBI"], bins=[-0.01, 4, 7, 16, 100],
                       labels=["HBI_Remission", "HBI_Legere", "HBI_Moderee", "HBI_Severe"])
df["Duration_cat"] = pd.cut(df["Duration"], bins=[-0.01, 2, 10, 100],
                            labels=["Duration_Recent", "Duration_Moyenne", "Duration_Longue"])
df["WBC_cat"] = pd.cut(df["WBC"], bins=[0, 4, 11, 100],
                       labels=["WBC_Bas", "WBC_Normal", "WBC_Eleve"])

cat_cols = ["Sex", "Diagnosis", "Treatment", "Surgery", "Smoking",
            "Cannabis", "FlareStatus", "Antibiotics", "Stool",
            "Age_cat", "BMI_cat", "CRP_cat", "Hb_cat", "HBI_cat",
            "Duration_cat", "WBC_cat"]

def make_transaction(row):
    tokens = []
    for c in cat_cols:
        val = row[c]
        if pd.notna(val) and str(val).strip() != "":
            token = str(val)
            if not any(token.startswith(p + "_") for p in ["Age", "BMI", "CRP", "Hb", "HBI", "Duration", "WBC"]):
                token = f"{c}_{val}"
            tokens.append(token)
    return tokens

# ---------------------------------------------------------------------
# 3. APRIORI PAR CLUSTER
# ---------------------------------------------------------------------
MIN_SUPPORT = 0.25  # support élevé pour éviter explosion combinatoire
MIN_CONFIDENCE = 0.75
MIN_LIFT = 1.2
MAX_LEN = 4  # limiter la longueur max des itemsets

print(f"[3] APRIORI PAR CLUSTER")
print(f"    min_support = {MIN_SUPPORT}, min_conf = {MIN_CONFIDENCE}, min_lift = {MIN_LIFT}\n")

cluster_rules = {}
cluster_summary = []

for cluster_id in sorted(df["Cluster_KMeans"].unique()):
    df_c = df[df["Cluster_KMeans"] == cluster_id]
    n = len(df_c)
    print(f"  --- CLUSTER {cluster_id} : {n} patients ---")

    transactions = df_c.apply(make_transaction, axis=1).tolist()
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_oh = pd.DataFrame(te_array, columns=te.columns_)

    freq_items = apriori(df_oh, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_LEN)
    rules = association_rules(freq_items, metric="lift", min_threshold=MIN_LIFT)
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE]
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["cluster"] = cluster_id

    cluster_rules[cluster_id] = rules
    cluster_summary.append({
        "Cluster": cluster_id, "N_patients": n,
        "N_itemsets": len(freq_items), "N_rules": len(rules),
        "Lift_max": round(rules["lift"].max(), 3) if len(rules) > 0 else 0,
        "Lift_moyen": round(rules["lift"].mean(), 3) if len(rules) > 0 else 0,
    })

    # Sauvegarder par cluster
    rules_save = rules[["antecedents_str", "consequents_str",
                        "support", "confidence", "lift"]].copy()
    rules_save.columns = ["antecedent", "consequent", "support", "confidence", "lift"]
    rules_save.to_csv(f"rules_cluster_{cluster_id}.csv", index=False)

    print(f"    Itemsets fréquents : {len(freq_items)}")
    print(f"    Règles trouvées    : {len(rules)}")
    if len(rules) > 0:
        print(f"    TOP 3 règles (par Lift) :")
        for _, r in rules.head(3).iterrows():
            ant = r["antecedents_str"][:60] + "..." if len(r["antecedents_str"]) > 60 else r["antecedents_str"]
            cons = r["consequents_str"][:60] + "..." if len(r["consequents_str"]) > 60 else r["consequents_str"]
            print(f"      {ant} ⇒ {cons}")
            print(f"        Sup={r['support']:.3f}, Conf={r['confidence']:.3f}, Lift={r['lift']:.3f}")
    print()

# ---------------------------------------------------------------------
# 4. SYNTHÈSE COMPARATIVE
# ---------------------------------------------------------------------
print("[4] SYNTHÈSE COMPARATIVE DES CLUSTERS")
print("-" * 70)
summary_df = pd.DataFrame(cluster_summary)
print(summary_df.to_string(index=False))
summary_df.to_csv("cluster_rules_summary.csv", index=False)

# ---------------------------------------------------------------------
# 5. VISUALISATIONS
# ---------------------------------------------------------------------
print("\n[5] VISUALISATIONS")
print("-" * 70)

# Comparaison du nombre de règles par cluster
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(summary_df["Cluster"].astype(str), summary_df["N_rules"],
            color=['#3498db', '#e74c3c', '#27ae60'][:len(summary_df)],
            edgecolor='black')
for i, n in enumerate(summary_df["N_rules"]):
    axes[0].text(i, n + max(summary_df["N_rules"]) * 0.02,
                  str(n), ha='center', fontsize=11, fontweight='bold')
axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("Nombre de règles")
axes[0].set_title("Nombre de règles d'association par cluster",
                   fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(summary_df["Cluster"].astype(str), summary_df["Lift_moyen"],
            color=['#3498db', '#e74c3c', '#27ae60'][:len(summary_df)],
            edgecolor='black')
axes[1].axhline(y=1.0, color='red', linestyle='--', label='Lift = 1 (indépendance)')
for i, l in enumerate(summary_df["Lift_moyen"]):
    axes[1].text(i, l + max(summary_df["Lift_moyen"]) * 0.02,
                  f"{l:.2f}", ha='center', fontsize=11, fontweight='bold')
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("Lift moyen")
axes[1].set_title("Lift moyen des règles par cluster",
                   fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle("Comparaison des règles d'association entre clusters",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("01_clusters_rules_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_clusters_rules_comparison.png")

# Top 10 règles par cluster (graphique combiné)
n_clusters = len(cluster_rules)
fig, axes = plt.subplots(1, n_clusters, figsize=(7 * n_clusters, 7))
if n_clusters == 1:
    axes = [axes]

for ax, (cid, rules) in zip(axes, cluster_rules.items()):
    if len(rules) == 0:
        ax.text(0.5, 0.5, "Aucune règle", transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        continue
    top10 = rules.head(10).copy()
    top10["Règle"] = top10["antecedents_str"].str[:30] + " ⇒ " + top10["consequents_str"].str[:30]
    ax.barh(range(len(top10)), top10["lift"],
            color=plt.cm.Set2(cid), edgecolor='black')
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["Règle"], fontsize=8)
    ax.set_xlabel("Lift")
    ax.set_title(f"Cluster {cid} — Top 10 règles", fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

fig.suptitle("Top règles d'association par cluster",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("02_top_rules_per_cluster.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_top_rules_per_cluster.png")

print("\n" + "=" * 70)
print("ÉTAPE 4 TERMINÉE AVEC SUCCÈS")
print("=" * 70)
