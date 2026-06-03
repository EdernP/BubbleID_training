import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- Configuration des chemins ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")

def analyze_bubble_sizes(dataset_path):
    labelme_files = sorted(glob.glob(os.path.join(dataset_path, "*.json")))
    
    widths = []
    heights = []
    
    # Pour le calcul des ancres (redimensionnées à 1024px)
    scaled_sizes = []
    aspect_ratios = []
    
    for file_path in labelme_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            
        for shape in label_data["shapes"]:
            points = np.asarray(shape["points"])
            
            # Ignorer les annotations mal formées (moins de 2 points)
            if len(points) < 2:
                continue
                
            # Calcul des limites (bounding box)
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
                
            w = xmax - xmin
            h = ymax - ymin
            
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
                
                # Simulation du redimensionnement de l'image (A.LongestMaxSize(max_size=1024))
                img_w = label_data.get("imageWidth", 1024)
                img_h = label_data.get("imageHeight", 1024)
                scale = 1024.0 / max(img_w, img_h)
                
                w_scaled = w * scale
                h_scaled = h * scale
                
                # Detectron2 définit la "taille" de l'ancre comme la racine carrée de l'aire
                scaled_sizes.append(np.sqrt(w_scaled * h_scaled))
                # Detectron2 définit le ratio d'aspect comme hauteur / largeur
                aspect_ratios.append(h_scaled / w_scaled)
                
    if not widths or not heights:
        print(f"Aucune annotation trouvée dans {dataset_path}.")
        return

    widths = np.array(widths)
    heights = np.array(heights)
    
    print(f"--- Analyse de {len(widths)} bulles ---")
    print(f"Largeur (Width)  : Moy= {np.mean(widths):.1f}px | Médiane= {np.median(widths):.1f}px | Min= {np.min(widths):.1f}px | Max= {np.max(widths):.1f}px")
    print(f"Hauteur (Height) : Moy= {np.mean(heights):.1f}px | Médiane= {np.median(heights):.1f}px | Min= {np.min(heights):.1f}px | Max= {np.max(heights):.1f}px")
    
    print("\n--- Recommandations pour CoarseDropout ---")
    print("Note : Il est souvent préférable d'utiliser le 75ème ou 90ème percentile plutôt que le 'Max' absolu (qui peut être une bulle géante ou une erreur).")
    print(f"Pour masquer ~90% des bulles : max_width={int(np.percentile(widths, 90))}, max_height={int(np.percentile(heights, 90))}")
    print(f"Pour masquer ~75% des bulles : max_width={int(np.percentile(widths, 75))}, max_height={int(np.percentile(heights, 75))}")
    print(f"Valeur médiane absolue        : max_width={int(np.median(widths))}, max_height={int(np.median(heights))}")

    # --- Calcul des Ancres par K-Means ---
    print("\n--- Recommandations pour les Ancres Detectron2 (K-Means) ---")
    print("Note : Calculé sur les bulles simulées après redimensionnement du réseau à 1024px.")
    
    scaled_sizes_arr = np.array(scaled_sizes).reshape(-1, 1)
    aspect_ratios_arr = np.array(aspect_ratios).reshape(-1, 1)
    
    # Trouver les 5 tailles optimales
    kmeans_sizes = KMeans(n_clusters=5, random_state=42, n_init=10).fit(scaled_sizes_arr)
    optimal_sizes = np.sort(kmeans_sizes.cluster_centers_.flatten())
    optimal_sizes_int = [int(round(s)) for s in optimal_sizes]
    
    # Trouver les 3 ratios optimaux
    kmeans_ar = KMeans(n_clusters=3, random_state=42, n_init=10).fit(aspect_ratios_arr)
    optimal_ar = np.sort(kmeans_ar.cluster_centers_.flatten())
    optimal_ar_round = [round(ar, 2) for ar in optimal_ar]
    
    print("\nÀ COPIER DANS VOTRE CONFIGURATION (train_kfold_optimized.py) :")
    print(f"cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[{'], ['.join(map(str, optimal_sizes_int))}]]")
    print(f"cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[{', '.join(map(str, optimal_ar_round))}]]")

    # Affichage du graphique de distribution
    plt.figure(figsize=(10, 5))
    plt.hist([widths, heights], bins=50, label=['Largeurs', 'Hauteurs'], alpha=0.7)
    plt.title('Distribution des dimensions des bulles (Pixels)')
    plt.xlabel('Taille (pixels)')
    plt.ylabel('Nombre de bulles')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    analyze_bubble_sizes(DATASET_PATH)