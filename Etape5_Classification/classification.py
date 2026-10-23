"""
======================================================================
ÉTAPE 5 — CLASSIFICATION.
======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)

print("=" * 70)
print("ÉTAPE 5 — CLASSIFICATION FLARE vs REMISSION")
print("=" * 70)

# ---------------------------------------------------------------------
# 1. CHARGEMENT
# ---------------------------------------------------------------------
df = pd.read_csv(os.path.join(_DIR, "..", "Etape1_Nettoyage", "crohn_clean.csv"))
print(f"\n[1] Dataset chargé : {df.shape}")
print(f"    Distribution cible 'FlareStatus' : {df['FlareStatus'].value_counts().to_dict()}\n")

# ---------------------------------------------------------------------
# 2. PRÉPARATION DES FEATURES ET DE LA CIBLE
# ---------------------------------------------------------------------
print("[2] PRÉPARATION DES DONNÉES")
print("-" * 70)

# Cible : Flare = 1, Remission = 0
y = (df["FlareStatus"] == "Flare").astype(int)
print(f"  Cible binaire : 1=Flare ({y.sum()}), 0=Remission ({(1-y).sum()})")
print(f"  Ratio classes : {y.mean()*100:.1f}% Flare / {(1-y.mean())*100:.1f}% Remission\n")

# Features : variables cliniques numériques + catégorielles encodées
# On EXCLUT FlareStatus (cible) et PatientID (identifiant)
feature_cols = [c for c in df.columns if c not in ["FlareStatus", "PatientID"]]
X = df[feature_cols].copy()

# Encodage des catégorielles
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"  Variables catégorielles encodées : {len(categorical_cols)}")
for c in categorical_cols:
    X[c] = LabelEncoder().fit_transform(X[c].astype(str))

print(f"  Features finales : {X.shape[1]}\n")

# ---------------------------------------------------------------------
# 3. SPLIT TRAIN/TEST STRATIFIÉ
# ---------------------------------------------------------------------
print("[3] SPLIT TRAIN/TEST STRATIFIÉ (80/20)")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {len(X_train)} ({y_train.sum()} Flare, {(1-y_train).sum()} Remission)")
print(f"  Test  : {len(X_test)} ({y_test.sum()} Flare, {(1-y_test).sum()} Remission)\n")

# Standardisation (essentielle pour KNN et Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 4. OVERSAMPLING (UNIQUEMENT SUR LE TRAIN — pas de data leakage)
# ---------------------------------------------------------------------
print("[4] RÉ-ÉQUILIBRAGE PAR OVERSAMPLING")
print("-" * 70)
print(f"  Avant oversampling — Train : {dict(pd.Series(y_train).value_counts().sort_index())}")

ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)
X_train_unscaled_res, _ = ros.fit_resample(X_train, y_train)

print(f"  Après oversampling — Train : {dict(pd.Series(y_train_res).value_counts().sort_index())}\n")

# ---------------------------------------------------------------------
# 5. ENTRAINEMENT DES 4 MODÈLES
# ---------------------------------------------------------------------
print("[5] ENTRAÎNEMENT DES 4 MODÈLES")
print("-" * 70)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced'),
    "KNN (k=7)":     KNeighborsClassifier(n_neighbors=7, weights='distance'),
    "Logistic Reg.": LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'),
}

results = {}

for name, model in models.items():
    print(f"\n  -- {name} --")
    
    # Pour KNN et Logistic Regression : on utilise les données scaled
    if name in ["KNN (k=7)", "Logistic Reg."]:
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # Decision Tree et Random Forest : pas besoin de scaling
        model.fit(X_train_unscaled_res, y_train_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    AUC       : {auc:.4f}")

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "cm": cm
    }

# ---------------------------------------------------------------------
# 6. VALIDATION CROISÉE 5-FOLD
# ---------------------------------------------------------------------
print("\n[6] VALIDATION CROISÉE 5-FOLD STRATIFIÉE")
print("-" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

# Pour la CV on n'oversample PAS — on utilise class_weight au lieu
for name, model in models.items():
    if name in ["KNN (k=7)", "Logistic Reg."]:
        scores = cross_val_score(model, scaler.fit_transform(X), y, cv=cv,
                                  scoring='roc_auc', n_jobs=-1)
    else:
        scores = cross_val_score(model, X, y, cv=cv,
                                  scoring='roc_auc', n_jobs=-1)
    cv_results.append({
        "Modèle": name,
        "AUC_moyen": round(scores.mean(), 4),
        "AUC_std": round(scores.std(), 4),
        "AUC_min": round(scores.min(), 4),
        "AUC_max": round(scores.max(), 4)
    })
    print(f"  {name:20s}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv("cross_validation_results.csv", index=False)

# ---------------------------------------------------------------------
# 7. TABLEAU COMPARATIF DES MÉTRIQUES
# ---------------------------------------------------------------------
print("\n[7] TABLEAU COMPARATIF DES MÉTRIQUES")
print("-" * 70)

metrics_df = pd.DataFrame({
    "Modèle": list(results.keys()),
    "Accuracy": [round(r["accuracy"], 4) for r in results.values()],
    "Precision": [round(r["precision"], 4) for r in results.values()],
    "Recall": [round(r["recall"], 4) for r in results.values()],
    "F1": [round(r["f1"], 4) for r in results.values()],
    "AUC": [round(r["auc"], 4) for r in results.values()],
})
print(metrics_df.to_string(index=False))
metrics_df.to_csv("model_comparison.csv", index=False)

# Identifier le meilleur modèle
best_model_name = metrics_df.loc[metrics_df["AUC"].idxmax(), "Modèle"]
print(f"\n  ★ MEILLEUR MODÈLE selon AUC : {best_model_name}")

# ---------------------------------------------------------------------
# 8. VISUALISATIONS
# ---------------------------------------------------------------------
print("\n[8] VISUALISATIONS")
print("-" * 70)

# (a) Matrices de confusion
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()
for ax, (name, res) in zip(axes, results.items()):
    cm = res["cm"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Remission', 'Flare'],
                yticklabels=['Remission', 'Flare'],
                cbar=False, linewidths=1, linecolor='white',
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"{name}\nAcc={res['accuracy']:.3f} | F1={res['f1']:.3f} | AUC={res['auc']:.3f}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
fig.suptitle("Matrices de confusion — Comparaison des 4 modèles",
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("01_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_confusion_matrices.png")

# (b) Courbes ROC superposées
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, label=f'{name} (AUC = {res["auc"]:.3f})',
            color=color, linewidth=2.5)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Aléatoire (AUC = 0.500)')
ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs (Recall)', fontsize=12)
ax.set_title('Courbes ROC — Comparaison des 4 modèles',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("02_roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_roc_curves.png")

# (c) Comparaison des métriques
fig, ax = plt.subplots(figsize=(13, 7))
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metric_names))
width = 0.18

for i, (name, res) in enumerate(results.items()):
    vals = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['auc']]
    ax.bar(x + i * width, vals, width, label=name, edgecolor='black')

ax.set_xlabel('Métrique', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparaison des métriques pour les 4 modèles',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(metric_names)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("03_metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_metrics_comparison.png")

# (d) Importance des variables (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({
    "Variable": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(11, 8))
colors_imp = plt.colormaps["viridis"](feat_imp["Importance"] / feat_imp["Importance"].max())
ax.barh(feat_imp["Variable"], feat_imp["Importance"],
        color=colors_imp, edgecolor='black')
ax.set_xlabel("Importance (Mean Decrease in Impurity)", fontsize=11)
ax.set_title("Importance des variables — Random Forest",
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("04_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_feature_importance.png")

feat_imp.to_csv("feature_importance.csv", index=False)
print("\n  Top 5 variables prédictives (Random Forest) :")
top_features = feat_imp.sort_values("Importance", ascending=False).head(5)
for _, r in top_features.iterrows():
    print(f"    {r['Variable']:30s} : {r['Importance']:.4f}")

# (e) Validation croisée — bar plot avec error bars
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(cv_df["Modèle"], cv_df["AUC_moyen"], yerr=cv_df["AUC_std"],
       capsize=10, color=['#3498db', '#e74c3c', '#27ae60', '#9b59b6'],
       edgecolor='black', error_kw={"linewidth": 2})
for i, (m, s) in enumerate(zip(cv_df["AUC_moyen"], cv_df["AUC_std"])):
    ax.text(i, m + s + 0.02, f"{m:.3f}±{s:.3f}",
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel("AUC (validation croisée 5-fold)", fontsize=12)
ax.set_title("Validation croisée 5-fold stratifiée — AUC moyen ± écart-type",
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim([0, 1.1])
ax.axhline(y=0.5, color='red', linestyle='--', label='Classifieur aléatoire')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("05_cross_validation.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 05_cross_validation.png")

# =====================================================================
# ANALYSE COMPLÉMENTAIRE — MODÈLES SANS HBI
# =====================================================================
print("\n" + "=" * 70)
print("ANALYSE COMPLÉMENTAIRE — MODÈLES SANS HBI")
print("=" * 70)
print("  Objectif : mesurer l'apport réel de HBI dans la prédiction de FlareStatus.")
print("  Si les modèles sans HBI sont presque aussi bons, HBI est redondant.\n")

# --- Préparation des données sans HBI ---
feature_cols_no_hbi = [c for c in feature_cols if c != "HBI"]
X_no_hbi = df[feature_cols_no_hbi].copy()
for c in X_no_hbi.select_dtypes(include=["object"]).columns:
    X_no_hbi[c] = LabelEncoder().fit_transform(X_no_hbi[c].astype(str))
print(f"[A] Features SANS HBI : {X_no_hbi.shape[1]} variables\n")

# --- Split stratifié ---
X_train_nh, X_test_nh, y_train_nh, y_test_nh = train_test_split(
    X_no_hbi, y, test_size=0.2, random_state=42, stratify=y
)

# --- Standardisation ---
scaler_nh = StandardScaler()
X_train_nh_scaled = scaler_nh.fit_transform(X_train_nh)
X_test_nh_scaled  = scaler_nh.transform(X_test_nh)

# --- Oversampling ---
ros_nh = RandomOverSampler(random_state=42)
X_train_nh_res, y_train_nh_res         = ros_nh.fit_resample(X_train_nh_scaled, y_train_nh)
X_train_nh_unscaled_res, _             = ros_nh.fit_resample(X_train_nh, y_train_nh)

# --- Entraînement des 4 modèles sans HBI ---
print("[B] ENTRAÎNEMENT DES 4 MODÈLES SANS HBI")
print("-" * 70)

models_no_hbi = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced'),
    "KNN (k=7)":     KNeighborsClassifier(n_neighbors=7, weights='distance'),
    "Logistic Reg.": LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'),
}

results_no_hbi = {}
for name, model in models_no_hbi.items():
    if name in ["KNN (k=7)", "Logistic Reg."]:
        model.fit(X_train_nh_res, y_train_nh_res)
        y_pred_nh   = model.predict(X_test_nh_scaled)
        y_proba_nh  = model.predict_proba(X_test_nh_scaled)[:, 1]
    else:
        model.fit(X_train_nh_unscaled_res, y_train_nh_res)
        y_pred_nh   = model.predict(X_test_nh)
        y_proba_nh  = model.predict_proba(X_test_nh)[:, 1]

    results_no_hbi[name] = {
        "model":     model,
        "y_pred":    y_pred_nh,
        "y_proba":   y_proba_nh,
        "accuracy":  accuracy_score(y_test_nh, y_pred_nh),
        "precision": precision_score(y_test_nh, y_pred_nh),
        "recall":    recall_score(y_test_nh, y_pred_nh),
        "f1":        f1_score(y_test_nh, y_pred_nh),
        "auc":       roc_auc_score(y_test_nh, y_proba_nh),
        "cm":        confusion_matrix(y_test_nh, y_pred_nh),
    }
    print(f"  {name:20s}: AUC={results_no_hbi[name]['auc']:.4f}  F1={results_no_hbi[name]['f1']:.4f}")

# --- Tableau comparatif AVEC vs SANS HBI ---
print("\n[C] TABLEAU COMPARATIF : AVEC HBI vs SANS HBI")
print("-" * 70)

comp_rows = []
for name in results:
    comp_rows.append({
        "Modèle":       name,
        "AUC_avecHBI":  round(results[name]["auc"], 4),
        "AUC_sansHBI":  round(results_no_hbi[name]["auc"], 4),
        "Delta_AUC":    round(results[name]["auc"] - results_no_hbi[name]["auc"], 4),
        "F1_avecHBI":   round(results[name]["f1"], 4),
        "F1_sansHBI":   round(results_no_hbi[name]["f1"], 4),
        "Delta_F1":     round(results[name]["f1"] - results_no_hbi[name]["f1"], 4),
    })

comp_df = pd.DataFrame(comp_rows)
print(comp_df.to_string(index=False))
comp_df.to_csv("comparison_avec_sans_HBI.csv", index=False)
print("\n  ✓ comparison_avec_sans_HBI.csv")

print("\n  INTERPRÉTATION :")
for _, row in comp_df.iterrows():
    if abs(row["Delta_AUC"]) < 0.02:
        verdict = "HBI apporte peu — redondant avec les autres variables"
    elif row["Delta_AUC"] > 0:
        verdict = "HBI est utile — améliore significativement la prédiction"
    else:
        verdict = "HBI nuit — bruit ou surapprentissage"
    print(f"    {row['Modèle']:20s} ΔAUC={row['Delta_AUC']:+.4f} — {verdict}")

# --- Courbes ROC sans HBI ---
fig, ax = plt.subplots(figsize=(10, 8))
colors_roc = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6']
for (name, res), color in zip(results_no_hbi.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test_nh, res["y_proba"])
    ax.plot(fpr, tpr, label=f'{name} (AUC = {res["auc"]:.3f})',
            color=color, linewidth=2.5)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Aléatoire (AUC = 0.500)')
ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs (Recall)', fontsize=12)
ax.set_title('Courbes ROC — Modèles SANS HBI',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("06_roc_curves_sans_hbi.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✓ 06_roc_curves_sans_hbi.png")

# --- Importance des variables RF sans HBI ---
rf_no_hbi      = results_no_hbi["Random Forest"]["model"]
feat_imp_nh    = pd.DataFrame({
    "Variable":   feature_cols_no_hbi,
    "Importance": rf_no_hbi.feature_importances_
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(11, 8))
_cmap_nh = plt.colormaps["viridis"]
ax.barh(feat_imp_nh["Variable"],
        feat_imp_nh["Importance"],
        color=_cmap_nh(feat_imp_nh["Importance"] / feat_imp_nh["Importance"].max()),
        edgecolor='black')
ax.set_xlabel("Importance (Mean Decrease in Impurity)", fontsize=11)
ax.set_title("Importance des variables SANS HBI — Random Forest",
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("07_feature_importance_sans_hbi.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 07_feature_importance_sans_hbi.png")

feat_imp_nh.sort_values("Importance", ascending=False).to_csv(
    "feature_importance_sans_hbi.csv", index=False)
print("  ✓ feature_importance_sans_hbi.csv")

print("\n" + "=" * 70)
print("ÉTAPE 5 TERMINÉE AVEC SUCCÈS")
print("=" * 70)
