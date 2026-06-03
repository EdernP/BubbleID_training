import os
import cv2
import numpy as np
import glob
import argparse
import random

def create_background_image(dataset_path: str, output_path: str, num_images: int = 50) -> None:
    """
    Crée une image de fond en calculant la médiane d'un échantillon d'images.

    Args:
        dataset_path (str): Chemin vers le dossier contenant les images du dataset.
        output_path (str): Chemin où sauvegarder l'image de fond (ex: 'background.png').
        num_images (int): Nombre d'images à utiliser pour calculer la médiane.

    """
    image_files = glob.glob(os.path.join(dataset_path, "*.png")) + \
                  glob.glob(os.path.join(dataset_path, "*.jpg")) + \
                  glob.glob(os.path.join(dataset_path, "*.jpeg"))

    if not image_files:
        print(f"Erreur : Aucune image trouvée dans {dataset_path}")
        return

    # Sélectionner un échantillon aléatoire d'images
    sample_files = random.sample(image_files, min(num_images, len(image_files)))
    
    print(f"Échantillonnage de {len(sample_files)} images pour créer le fond...")

    # Lire et empiler les images, en ignorant celles qui ne peuvent pas être lues
    images = []
    for file in sample_files:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
        else:
            print(f"Avertissement : Impossible de lire l'image {file}. Elle sera ignorée.")

    if not images:
        print("Erreur : Aucune image valide n'a pu être chargée. Impossible de créer le fond.")
        return
    
    print(f"Utilisation de {len(images)} images valides pour le calcul.")
    # Vérifier que toutes les images ont la même taille
    first_shape = images[0].shape
    if not all(img.shape == first_shape for img in images):
        print("Erreur : Toutes les images ne font pas la même taille. Impossible de créer le fond.")
        return

    # Calculer la médiane pixel par pixel
    median_frame = np.median(images, axis=0).astype(np.uint8)

    # Sauvegarder l'image de fond
    cv2.imwrite(output_path, median_frame)
    print(f"Image de fond sauvegardée dans : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crée une image de fond à partir d'un dataset.")
    parser.add_argument("--dataset", required=True, help="Chemin vers le dossier du jeu de données (contenant les images).")
    parser.add_argument("--output", default="background.png", help="Chemin du fichier de sortie pour l'image de fond.")
    parser.add_argument("--num_images", type=int, default=50, help="Nombre d'images à utiliser pour la médiane.")
    args = parser.parse_args()

    create_background_image(args.dataset, args.output, args.num_images)