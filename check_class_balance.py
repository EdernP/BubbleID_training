import os
import json
import glob
from collections import Counter

# --- Configuration (copiée depuis train_kfold.py) ---
DATASET_PATH = "dataset"

# Label Mapping
MAPPING = {
    "detached": "detached",
    "occluseDetached": "detached",
    "occlusedDetached": "detached",
    "occlusedAttached": "occludedAttached",
    "unknown": "occludedAttached",
    "attachedSide": "attached",
    "attached": "attached"
}
FINAL_LABELS = sorted(list(set(MAPPING.values())))
# ----------------------------------------------------

def check_class_balance(dataset_path: str):
    """
    Analyse les fichiers d'annotation LabelMe pour vérifier la distribution des classes.

    Args:
        dataset_path: Chemin vers le dossier contenant les fichiers .json de LabelMe.
    """
    if not os.path.isdir(dataset_path):
        print(f"Erreur : Le dossier '{dataset_path}' n'a pas été trouvé.")
        return

    labelme_files = glob.glob(os.path.join(dataset_path, "*.json"))
    if not labelme_files:
        print(f"Aucun fichier .json trouvé dans '{dataset_path}'.")
        return

    instance_counts = Counter()

    print(f"Analyse de {len(labelme_files)} fichiers d'annotation...")

    for file_path in labelme_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                label_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Avertissement : Impossible de lire le fichier JSON malformé : {file_path}")
                continue

        for shape in label_data.get("shapes", []):
            raw_label = shape.get("label")
            if raw_label:
                mapped_label = MAPPING.get(raw_label)
                if mapped_label:
                    instance_counts[mapped_label] += 1

    print("\n--- Distribution des Instances par Classe ---")
    if not instance_counts:
        print("Aucune instance de classe valide n'a été trouvée.")
        return
        
    total_instances = sum(instance_counts.values())
    print(f"Nombre total d'instances (annotations) : {total_instances}\n")

    for label in FINAL_LABELS:
        count = instance_counts.get(label, 0)
        percentage = (count / total_instances) * 100 if total_instances > 0 else 0
        print(f"- {label:<20}: {count:>6} instances ({percentage:.2f}%)")
    
    print("-------------------------------------------")

    detached_percentage = (instance_counts.get("detached", 0) / total_instances) * 100 if total_instances > 0 else 0
    if detached_percentage < 10:
         print("\n💡 Analyse : La classe 'detached' est significativement sous-représentée (< 10%).")
         print("   Ceci est très probablement la cause principale de la faible performance (AP) pour cette classe.")

if __name__ == "__main__":
    check_class_balance(DATASET_PATH)