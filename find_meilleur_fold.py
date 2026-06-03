import os
import json
import logging
import sys

# --- Configuration ---
OUTPUT_DIR = "kfold_run"
METRIC_TO_CHECK = "bbox/AP"
# ---------------------

def find_best_fold(output_dir: str, metric: str):
    """
    Analyse les fichiers metrics.json dans les sous-dossiers de k-fold
    pour trouver le fold ayant obtenu le meilleur score.
    """
    logger = logging.getLogger("detectron2")
    fold_scores = {}

    # Parcourir tous les dossiers "fold_X"
    for item in os.listdir(output_dir):
        fold_path = os.path.join(output_dir, item)
        if os.path.isdir(fold_path) and item.startswith("fold_"):
            metrics_path = os.path.join(fold_path, "metrics.json")
            
            if not os.path.exists(metrics_path):
                logger.warning(f"Fichier metrics.json non trouvé pour {item}. On ignore.")
                continue

            best_fold_score = -1.0
            try:
                with open(metrics_path, 'r') as f:
                    for line in f:
                        metrics_data = json.loads(line)
                        if metric in metrics_data:
                            # La métrique peut être une valeur simple ou un tuple (valeur, itération)
                            current_score = metrics_data[metric]
                            if isinstance(current_score, (list, tuple)):
                                current_score = current_score[0]
                            
                            if current_score > best_fold_score:
                                best_fold_score = current_score
            except Exception as e:
                logger.error(f"Erreur en lisant {metrics_path}: {e}")
                continue
            
            fold_scores[item] = best_fold_score

    if not fold_scores:
        print("Aucun score trouvé. Avez-vous bien lancé l'entraînement ?")
        return

    # Trouver le meilleur fold parmi tous
    best_fold_name = max(fold_scores, key=fold_scores.get)
    best_overall_score = fold_scores[best_fold_name]

    print("--- Analyse des performances des Folds ---")
    for fold, score in sorted(fold_scores.items()):
        print(f"{fold}: Meilleur {metric} = {score:.4f}")
    
    print("\n-------------------------------------------")
    print(f"🏆 Le modèle champion provient de : {best_fold_name}")
    print(f"   Avec un score de : {best_overall_score:.4f}")
    print("-------------------------------------------")


if __name__ == "__main__":
    # Assurez-vous que le logger est configuré pour voir les messages
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    target_dir = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_DIR
    find_best_fold(target_dir, METRIC_TO_CHECK)
