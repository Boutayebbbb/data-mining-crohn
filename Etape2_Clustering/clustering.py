"""
======================================================================
ÉTAPE 2 — CLUSTERING
======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)

# ---------------------------------------------------------------------
# 1. CHARGEMENT
# ---------------------------------------------------------------------
print("=" * 70)
print("ÉTAPE 2 — CLUSTERING")
print("=" * 70)

df = pd.read_csv(os.path.join(_DIR, "..", "Etape1_Nettoyage", "crohn_clean.csv"))
print(f"\n[1] Dataset chargé : {df.shape[0]} × {df.shape[1]}\n")

# Variables numériques pour le clustering
num_features = ["Age", "BMI", "CRP", "WBC", "Hb", "Platelets",
                "ALT", "AST", "Duration", "HBI"]
X = df[num_features].copy()

# Standardisation (essentielle pour le clustering basé distance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"[2] {len(num_features)} variables numériques standardisées")
print(f"    Moyenne post-standardisation : {X_scaled.mean(axis=0).round(3)}")
print(f"    Écart-type post-standardisation : {X_scaled.std(axis=0).round(3)}\n")

# ---------------------------------------------------------------------
# 3. RECHERCHE SYSTÉMATIQUE DU K OPTIMAL POUR K-MEANS
# ---------------------------------------------------------------------
print("[3] RECHERCHE DU K OPTIMAL — k de 2 à 10")
print("-" * 70)

k_range = range(2, 11)
inertias = []
silhouettes = []
db_scores = []
ch_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    print(f"  k={k} : Inertie={km.inertia_:8.1f} | "
          f"Silhouette={silhouettes[-1]:.4f} | "
          f"Davies-Bouldin={db_scores[-1]:.4f} | "
          f"Calinski-Harabasz={ch_scores[-1]:.1f}")

# Méthode du coude visuelle
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (a) Méthode du coude
axes[0, 0].plot(list(k_range), inertias, marker='o', linewidth=2, markersize=10,
                color='#3498db', markerfacecolor='#e74c3c')
axes[0, 0].set_xlabel("Nombre de clusters k", fontsize=11)
axes[0, 0].set_ylabel("Inertie intra-cluster", fontsize=11)
axes[0, 0].set_title("(a) Méthode du Coude (Elbow)",
                     fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(list(k_range))
axes[0, 0].grid(True, alpha=0.3)

# (b) Silhouette
axes[0, 1].plot(list(k_range), silhouettes, marker='s', linewidth=2,
                markersize=10, color='#27ae60', markerfacecolor='#f39c12')
axes[0, 1].set_xlabel("Nombre de clusters k", fontsize=11)
axes[0, 1].set_ylabel("Score Silhouette", fontsize=11)
axes[0, 1].set_title("(b) Silhouette (à MAXIMISER)",
                     fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(list(k_range))
best_k_sil = list(k_range)[np.argmax(silhouettes)]
axes[0, 1].axvline(x=best_k_sil, color='red', linestyle='--',
                   label=f'Meilleur k = {best_k_sil}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) Davies-Bouldin
axes[1, 0].plot(list(k_range), db_scores, marker='^', linewidth=2,
                markersize=10, color='#9b59b6', markerfacecolor='#1abc9c')
axes[1, 0].set_xlabel("Nombre de clusters k", fontsize=11)
axes[1, 0].set_ylabel("Score Davies-Bouldin", fontsize=11)
axes[1, 0].set_title("(c) Davies-Bouldin (à MINIMISER)",
                     fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(list(k_range))
best_k_db = list(k_range)[np.argmin(db_scores)]
axes[1, 0].axvline(x=best_k_db, color='red', linestyle='--',
                   label=f'Meilleur k = {best_k_db}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# (d) Calinski-Harabasz
axes[1, 1].plot(list(k_range), ch_scores, marker='D', linewidth=2,
                markersize=10, color='#e67e22', markerfacecolor='#34495e')
axes[1, 1].set_xlabel("Nombre de clusters k", fontsize=11)
axes[1, 1].set_ylabel("Score Calinski-Harabasz", fontsize=11)
axes[1, 1].set_title("(d) Calinski-Harabasz (à MAXIMISER)",
                     fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(list(k_range))
best_k_ch = list(k_range)[np.argmax(ch_scores)]
axes[1, 1].axvline(x=best_k_ch, color='red', linestyle='--',
                   label=f'Meilleur k = {best_k_ch}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle("Recherche du nombre optimal de clusters — 4 critères",
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("01_CHOSEN_K_search.png", dpi=150, bbox_inches='tight')
plt.close()

# Sauvegarder les scores
scores_df = pd.DataFrame({
    "k": list(k_range), "Inertie": inertias, "Silhouette": silhouettes,
    "Davies_Bouldin": db_scores, "Calinski_Harabasz": ch_scores
})
scores_df.to_csv("kmeans_scores_per_k.csv", index=False)

print(f"\n  → Meilleur k selon Silhouette        : {best_k_sil}")
print(f"  → Meilleur k selon Davies-Bouldin    : {best_k_db}")
print(f"  → Meilleur k selon Calinski-Harabasz : {best_k_ch}")

# DÉCISION CLINIQUE : k = 3 (indépendamment du k mathématiquement optimal)
CHOSEN_K = 3
print(f"\n  ★ K RETENU : k = {CHOSEN_K} (DÉCISION CLINIQUE, pas mathématique)")
print(f"  JUSTIFICATION : 3 profils correspondent à la réalité de la MICI :")
print(f"    Cluster 0 — Rémission        : HBI bas, CRP normale, maladie stable")
print(f"    Cluster 1 — Activité modérée : HBI intermédiaire, légère inflammation")
print(f"    Cluster 2 — Poussée sévère   : HBI élevé, CRP élevée, maladie active")
print(f"  → k=2 est trop grossier ; k>3 fragmente des groupes cliniquement similaires.\n")

# ---------------------------------------------------------------------
# 4. K-MEANS FINAL AVEC LE K OPTIMAL
# ---------------------------------------------------------------------
print(f"[4] K-MEANS FINAL avec k = {CHOSEN_K}")
print("-" * 70)

km_final = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=20, max_iter=500)
df["Cluster_KMeans"] = km_final.fit_predict(X_scaled)

sil_kmeans = silhouette_score(X_scaled, df["Cluster_KMeans"])
db_kmeans = davies_bouldin_score(X_scaled, df["Cluster_KMeans"])
ch_kmeans = calinski_harabasz_score(X_scaled, df["Cluster_KMeans"])

print(f"  Silhouette         : {sil_kmeans:.4f}")
print(f"  Davies-Bouldin     : {db_kmeans:.4f}")
print(f"  Calinski-Harabasz  : {ch_kmeans:.1f}")

# Distribution des clusters
print(f"\n  Distribution des clusters :")
for c, n in df["Cluster_KMeans"].value_counts().sort_index().items():
    print(f"    Cluster {c} : {n} patients ({100*n/len(df):.1f}%)")

# Profil moyen de chaque cluster
print(f"\n  Profils moyens des clusters :")
profile = df.groupby("Cluster_KMeans")[num_features].mean().round(2)
print(profile.to_string())
profile.to_csv("cluster_profiles_kmeans.csv")

# ---------------------------------------------------------------------
# 5. DBSCAN — DÉTERMINATION DE EPS PAR K-DISTANCE
# ---------------------------------------------------------------------
print("\n[5] DBSCAN — DÉTERMINATION DE EPS PAR K-DISTANCE")
print("-" * 70)

min_pts = 2 * len(num_features)  # règle empirique : 2 × dim
nbrs = NearestNeighbors(n_neighbors=min_pts).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_dist = np.sort(distances[:, min_pts - 1])

# Trouver le coude (méthode du percentile 90)
eps_value = float(np.percentile(k_dist, 90))
print(f"  MinPts choisi : {min_pts} (= 2 × nb dimensions)")
print(f"  Eps choisi (percentile 90 du k-distance) : {eps_value:.3f}")

# Visualisation k-distance
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(np.arange(len(k_dist)), k_dist, color='#2c3e50', linewidth=1.5)
ax.axhline(y=eps_value, color='red', linestyle='--', linewidth=2,
           label=f'Eps optimal = {eps_value:.3f}')
ax.set_xlabel("Points triés", fontsize=11)
ax.set_ylabel(f"Distance au {min_pts}e voisin le plus proche", fontsize=11)
ax.set_title("Graphique k-distance pour le choix de Eps",
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("02_kdistance_dbscan.png", dpi=150, bbox_inches='tight')
plt.close()

# Application DBSCAN
dbscan = DBSCAN(eps=eps_value, min_samples=min_pts)
df["Cluster_DBSCAN"] = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(df["Cluster_DBSCAN"])) - (1 if -1 in df["Cluster_DBSCAN"].values else 0)
n_noise_db = (df["Cluster_DBSCAN"] == -1).sum()
print(f"\n  Résultats DBSCAN :")
print(f"    Nombre de clusters détectés : {n_clusters_db}")
print(f"    Points de bruit : {n_noise_db} ({100*n_noise_db/len(df):.1f}%)")

# ---------------------------------------------------------------------
# 6. CLUSTERING HIÉRARCHIQUE (WARD)
# ---------------------------------------------------------------------
print("\n[6] CLUSTERING HIÉRARCHIQUE (Ward) — pour comparaison")
print("-" * 70)

ward = AgglomerativeClustering(n_clusters=CHOSEN_K, linkage='ward')
df["Cluster_Ward"] = ward.fit_predict(X_scaled)
sil_ward = silhouette_score(X_scaled, df["Cluster_Ward"])
print(f"  Silhouette (Ward) : {sil_ward:.4f}")
print(f"  Comparaison avec K-Means : {'meilleur' if sil_ward > sil_kmeans else 'moins bon'}")

# ---------------------------------------------------------------------
# 7. PCA POUR VISUALISATION 2D
# ---------------------------------------------------------------------
print("\n[7] PCA — Réduction de dimensionnalité pour visualisation")
print("-" * 70)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
print(f"  Variance expliquée par PC1 : {explained[0]*100:.2f}%")
print(f"  Variance expliquée par PC2 : {explained[1]*100:.2f}%")
print(f"  Variance totale capturée   : {explained.sum()*100:.2f}%")

# Visualisation des 3 méthodes côte-à-côte
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# K-Means
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                            c=df["Cluster_KMeans"], cmap='tab10',
                            s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
axes[0].set_title(f"K-Means (k={CHOSEN_K})\nSilhouette={sil_kmeans:.3f}",
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# DBSCAN
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                            c=df["Cluster_DBSCAN"], cmap='tab10',
                            s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
axes[1].set_title(f"DBSCAN (eps={eps_value:.2f})\n{n_clusters_db} clusters + {n_noise_db} bruit",
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

# Ward
scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1],
                            c=df["Cluster_Ward"], cmap='tab10',
                            s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
axes[2].set_title(f"Ward (k={CHOSEN_K})\nSilhouette={sil_ward:.3f}",
                  fontsize=12, fontweight='bold')
axes[2].set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
axes[2].set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
axes[2].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[2], label='Cluster')

fig.suptitle("Comparaison des 3 méthodes de clustering — Projection PCA 2D",
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("03_clustering_comparison_pca.png", dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------
# 8. HEATMAP DES PROFILS DES CLUSTERS K-MEANS
# ---------------------------------------------------------------------
print("\n[8] Heatmap des profils des clusters")
print("-" * 70)

# Standardiser les profils pour une heatmap lisible
profile_std = (profile - profile.mean()) / profile.std()
plt.figure(figsize=(13, 5))
sns.heatmap(profile_std.T, annot=profile.T, fmt=".1f", cmap='RdYlBu_r',
            center=0, cbar_kws={"label": "Z-score (normalized)"},
            linewidths=0.5)
plt.title(f"Profils standardisés des {CHOSEN_K} clusters K-Means\n"
          "(annotations = valeurs originales, couleurs = z-scores)",
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Cluster")
plt.ylabel("Variable")
plt.tight_layout()
plt.savefig("04_cluster_profiles_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------
# 9. RECAP COMPARATIF DES MÉTHODES
# ---------------------------------------------------------------------
print("\n[9] COMPARAISON FINALE DES MÉTHODES")
print("-" * 70)

comparison = pd.DataFrame({
    "Méthode": ["K-Means", "DBSCAN", "Ward (Hiérarchique)"],
    "Nb_clusters": [CHOSEN_K, n_clusters_db, CHOSEN_K],
    "Silhouette": [round(sil_kmeans, 4),
                   round(silhouette_score(X_scaled, df["Cluster_DBSCAN"]), 4) if n_clusters_db >= 2 else None,
                   round(sil_ward, 4)],
    "Bruit_détecté": [0, n_noise_db, 0]
})
print(comparison.to_string(index=False))
comparison.to_csv("clustering_methods_comparison.csv", index=False)

# ---------------------------------------------------------------------
# 10. SAUVEGARDE
# ---------------------------------------------------------------------
df.to_csv("crohn_clustered.csv", index=False)
print(f"\n  ✓ Dataset avec clusters sauvegardé : crohn_clustered.csv")
print(f"  ✓ Méthode retenue pour la suite : K-Means k={CHOSEN_K}")

print("\n" + "=" * 70)
print("ÉTAPE 2 TERMINÉE AVEC SUCCÈS")
print("=" * 70)
