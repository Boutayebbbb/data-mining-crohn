"""Exécute les 5 étapes du pipeline dans l'ordre (chaque script écrit ses résultats dans son dossier)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ETAPES = [
    ("Etape1_Nettoyage", "nettoyage.py"),
    ("Etape2_Clustering", "clustering.py"),
    ("Etape3_Association_Sans_Clustering", "association_globale.py"),
    ("Etape4_Association_Avec_Clustering", "association_clustering.py"),
    ("Etape5_Classification", "classification.py"),
]

if not os.path.exists(os.path.join(ROOT, "Data_Crohn.xlsx")):
    sys.exit("Fichier Data_Crohn.xlsx introuvable à la racine du projet (voir README).")

for dossier, script in ETAPES:
    print(f"\n>>> {dossier}/{script}")
    subprocess.run([sys.executable, script], cwd=os.path.join(ROOT, dossier), check=True)
print("\nPipeline terminé.")
