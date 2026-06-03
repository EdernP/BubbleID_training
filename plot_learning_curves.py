
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_learning_curves(metrics_path: str):
    """
    Lit un fichier metrics.json de Detectron2 et trace les courbes d'apprentissage.

    Args:
        metrics_path: Chemin vers le fichier metrics.json.
    """
    # Vérifier si le fichier existe
    if not os.path.exists(metrics_path):
        print(f"Erreur : Le fichier '{metrics_path}' n'a pas été trouvé.")
        return

    # Charger les métriques
    metrics_data = []
    with open(metrics_path, 'r') as f:
        for line in f:
            try:
                metrics_data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Avertissement : Impossible de lire une ligne dans {metrics_path}")

    if not metrics_data:
        print("Aucune donnée de métrique trouvée.")
        return

    # Séparer les métriques d'entraînement et d'évaluation
    train_metrics = [m for m in metrics_data if 'total_loss' in m]
    eval_metrics = [m for m in metrics_data if 'bbox/AP' in m]

    # Créer un DataFrame pandas pour une manipulation facile
    df = pd.DataFrame(train_metrics)

    # Créer la figure et les axes pour les graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Courbes d'apprentissage pour {os.path.basename(metrics_path)}", fontsize=16)

    # --- Graphique 1 : Perte d'entraînement vs. Performance de validation (AP) ---
    ax1.plot(df['iteration'], df['total_loss'], label='Total Loss (Entraînement)', color='tab:blue')
    # Calculer une moyenne mobile pour lisser la courbe
    if len(df) > 10:
        ax1.plot(df['iteration'], df['total_loss'].rolling(window=10).mean(), label='Moyenne mobile (10 it.)', linestyle='--', color='tab:cyan')
    ax1.set_ylabel('Perte (Loss)', color='tab:blue')
    ax1.set_title("Perte d'Entraînement vs. AP de Validation")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Axe Y secondaire pour l'Average Precision (AP)
    ax1_ap = ax1.twinx()
    if eval_metrics:
        eval_df = pd.DataFrame(eval_metrics)
        # La métrique AP peut être un float ou un tuple (valeur, itération)
        eval_df['bbox/AP_val'] = eval_df['bbox/AP'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
        ax1_ap.plot(eval_df['iteration'], eval_df['bbox/AP_val'], label='bbox/AP (Validation)', color='tab:red', marker='o', linestyle=':')
        ax1_ap.set_ylabel('bbox/AP', color='tab:red')
        ax1_ap.tick_params(axis='y', labelcolor='tab:red')

    # Combiner les légendes des deux axes Y
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_ap.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # --- Graphique 2 : Détailler les pertes ---
    loss_components = [key for key in df.columns if key.startswith('loss_')]
    for loss_name in loss_components:
        ax2.plot(df['iteration'], df[loss_name], label=loss_name)
    
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Perte (Loss)')
    ax2.set_title('Composantes de la Perte')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()
    
    # Afficher les métriques d'évaluation finales si elles existent
    if eval_metrics:
        final_eval = eval_metrics[-1]
        print("\n--- Métriques d'Évaluation Finales ---")
        for key, value in final_eval.items():
            if 'AP' in key:
                print(f"{key}: {value:.4f}")
        print("------------------------------------")


    # Afficher le graphique
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace les courbes d'apprentissage à partir d'un fichier metrics.json de Detectron2."
    )
    parser.add_argument(
        "metrics_file",
        type=str,
        help="Chemin vers le fichier metrics.json à analyser.",
    )
    args = parser.parse_args()
    
    plot_learning_curves(args.metrics_file)
