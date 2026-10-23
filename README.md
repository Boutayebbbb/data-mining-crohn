# Data Mining sur des données cliniques de la maladie de Crohn

Pipeline complet de Data Mining en 5 étapes sur une cohorte de **1 570 patients** (maladie de Crohn,
rectocolite hémorragique et contrôles sains) : nettoyage, clustering, règles d'association et
classification supervisée **Flare vs Remission**.

Le point central du projet est la **détection d'une fuite d'information** : le modèle atteint une AUC de
0,9996, mais cette performance repose presque entièrement sur le score clinique HBI. Une analyse
complémentaire sans HBI mesure la vraie difficulté du problème.

## Résultats principaux

| Étape | Résultat |
| --- | --- |
| 1. Nettoyage | 4 colonnes supprimées (> 50 % de manquants), imputation conditionnelle cohérente avec le diagnostic, 1 570 × 22 sans valeur manquante |
| 2. Clustering | K-Means k = 3 (recherche k = 2 à 10 sur 4 critères), comparé à DBSCAN et Ward |
| 3. Règles globales | 4 716 itemsets fréquents, 4 992 règles (support ≥ 10 %, confiance ≥ 60 %, lift ≥ 1,2) |
| 4. Règles par cluster | 89 règles fortes (lift moyen 2,09) dans le cluster sévère, 3 158 règles faibles (lift moyen 1,36) dans le cluster en rémission |
| 5. Classification | Random Forest : AUC 0,9996 avec HBI, **0,6564 sans HBI** |

### Classification : avec et sans HBI

| Modèle | AUC avec HBI | AUC sans HBI | F1 avec HBI | F1 sans HBI |
| --- | --- | --- | --- | --- |
| **Random Forest** | **0,9996** | **0,6564** | 0,9923 | 0,6031 |
| Logistic Regression | 0,9962 | 0,6122 | 0,9569 | 0,6064 |
| Decision Tree | 0,9831 | 0,5962 | 0,9808 | 0,5977 |
| KNN (k = 7) | 0,8534 | 0,6031 | 0,7234 | 0,5893 |

Avec HBI, cette variable représente **72,6 %** de l'importance du Random Forest : le HBI est un score
d'activité de la maladie, donc très proche de la cible. Sans HBI, les prédicteurs les plus utiles sont
Montreal Classification, Platelets, CRP et Duration, et l'AUC tombe à 0,66 : c'est la vraie difficulté
de prédire une poussée à partir des seuls marqueurs biologiques.

## Méthodologie

- **Nettoyage** : suppression des colonnes trop lacunaires plutôt qu'imputation artificielle ;
  HBI fixé à 0 pour les non-Crohn (score non applicable) ; traitement « Aucun » pour les contrôles sains ;
  outliers conservés car cliniquement réels.
- **Clustering** : standardisation, recherche du k optimal (inertie, Silhouette, Davies-Bouldin,
  Calinski-Harabasz), choix de k = 3 justifié cliniquement, validation par Ward et DBSCAN.
- **Règles d'association** : discrétisation selon des seuils cliniques, Apriori global puis par cluster.
- **Classification** : 4 modèles, split stratifié 80/20, sur-échantillonnage sur le train uniquement,
  validation croisée 5-fold, importance des variables, analyse avec et sans HBI.

Les limites de l'étude sont discutées dans le rapport (section 7.3).

## Structure du projet

```
├── Etape1_Nettoyage/                     nettoyage.py + statistiques et graphiques
├── Etape2_Clustering/                    clustering.py + recherche de k et profils des clusters
├── Etape3_Association_Sans_Clustering/   association_globale.py + règles globales
├── Etape4_Association_Avec_Clustering/   association_clustering.py + règles par cluster
├── Etape5_Classification/                classification.py + métriques, ROC, importance des variables
├── run_all.py                            exécute les 5 étapes dans l'ordre
└── docs/                                 rapport complet (PDF et Word)
```

## Exécution

```bash
pip install -r requirements.txt
python run_all.py
```

> **Données :** le fichier `Data_Crohn.xlsx` a été fourni dans le cadre du cours et contient des
> données cliniques. Il n'est **pas publié** dans ce dépôt, ni les fichiers intermédiaires contenant des
> données individuelles. Pour reproduire l'étude, placez le fichier à la racine du projet.
> Les résultats agrégés (statistiques, règles, métriques) et les graphiques sont inclus.

## Technologies

Python · pandas · NumPy · scikit-learn · mlxtend (Apriori) · imbalanced-learn · Matplotlib · Seaborn

## Contexte

Projet réalisé dans le cadre du module Data Mining (ESISA Fès, 2025-2026).

**Auteur :** Marouane Boutayeb
