import os
import json
import glob
import numpy as np
import torch
import detectron2
import cv2
import random
import yaml
import copy
import logging
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import StratifiedKFold

# Import detectron2 utilities
from detectron2.utils.env import seed_all_rng
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, hooks
from detectron2.engine.hooks import HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader, detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# --- HYPERPARAMETERS ---
# Modify these values to tune your training run.

# --- Nom du test en cours (à modifier avant chaque lancement) ---
CURRENT_TEST_NAME = "La_totale" # Ex: "baseline", "CLAHE", "GaussNoise", etc.

# --- Paths Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, f"Test_augmentation_{CURRENT_TEST_NAME}")

N_SPLITS = 4
RANDOM_STATE = 42
# Configuration par défaut pour des images en 800x800
MAX_ITER = 7000      # Passer à 14000 pour BATCH_SIZE = 2 (1024x1024)
LR = 0.0005          # Passer à 0.00025 pour BATCH_SIZE = 2 (1024x1024)
BATCH_SIZE = 4       # Passer à 2 pour des images en 1024x1024
EVAL_PERIOD = 200    # Passer à 400 pour 1024x1024
CHECKPOINT_PERIOD = 200

# --- Early Stopping Configuration ---
EARLY_STOPPING_PATIENCE = 6  # 6 * 200 = 1200 itérations de marge (Passer à 8 pour 1024x1024)
EARLY_STOPPING_METRIC = "bbox/AP" # Métrique à surveiller pour l'arrêt précoce.

# --- Focal Loss Configuration ---
USE_FOCAL_LOSS = False # Mettre à False pour utiliser la CrossEntropy standard pour la classification des boîtes
USE_BOUNDARY_LOSS = False # Mettre à False pour utiliser la Mask Loss standard (BCE classique)

# --- Configuration Section ---

# Label Mapping (from labelme2cocoMy.py)
MAPPING = {
    "detached": "detached",
    "occluseDetached": "detached",
    "occlusedDetached": "detached",
    "occlusedAttached": "occludedAttached",
    "unknown": "occludedAttached",
    "attachedSide": "attached",
    "attached": "attached"
}
CATEGORY_IDS = {
    "detached": 0,
    "occludedAttached": 1,
    "attached": 2
}
# On force l'ordre des labels à correspondre exactement aux valeurs de CATEGORY_IDS (0, 1, 2)
FINAL_LABELS = [k for k, v in sorted(CATEGORY_IDS.items(), key=lambda item: item[1])]

# --- Early Stopping Hook ---
class EarlyStoppingHook(HookBase):
    def __init__(self, patience, metric, goal="max"):
        self._patience = patience
        self._metric = metric
        self._goal = goal
        
        self._patience_counter = 0
        self._best_metric = -float('inf') if self._goal == "max" else float('inf')
        self._logger = logging.getLogger("detectron2")

    def after_step(self):
        # Ce hook dépend des résultats de l'EvalHook stockés
        latest_metrics = self.trainer.storage.latest()
        
        # La métrique n'est présente que lorsque l'évaluation a eu lieu
        if self._metric not in latest_metrics:
            return

        # Récupérer la valeur ET l'itération où elle a été enregistrée
        current_metric, metric_iter = latest_metrics[self._metric]

        # CORRECTION CRUCIALE : N'agir que si la métrique a été enregistrée à l'itération ACTUELLE.
        # Cela empêche le hook de ré-évaluer la même ancienne métrique à chaque pas, ce qui
        # déclenchait l'arrêt prématurément.
        if metric_iter != self.trainer.iter:
            return

        improved = False
        if self._goal == "max":
            if current_metric > self._best_metric:
                self._best_metric = current_metric
                improved = True
        else: # min
            if current_metric < self._best_metric:
                self._best_metric = current_metric
                improved = True
        
        if improved:
            self._patience_counter = 0
            # Le BestCheckpointer de detectron2 s'occupe de sauvegarder
        else:
            self._patience_counter += 1
        
        if self._patience_counter >= self._patience:
            self._logger.info(f"Arrêt précoce déclenché après {self._patience} évaluations sans amélioration.")
            self._logger.info(f"Meilleure métrique obtenue : {self._best_metric:.4f}")
            raise StopIteration # Stoppe la boucle d'entraînement proprement


# --- Helper Functions ---

def get_all_labelme_files(dataset_path):
    """Gathers all labelme json files from the dataset directory."""
    return sorted(glob.glob(os.path.join(dataset_path, "*.json")))

def create_detectron2_dataset_from_labelme(labelme_files):
    """
    Creates a Detectron2 formatted list of dicts in memory from a list of labelme files.
    It also returns a list of categories per image for stratification and the category mapping.
    """
    logger = logging.getLogger("detectron2")
    dataset_dicts = []
    image_categories = []

    for i, file_path in enumerate(labelme_files):
        with open(file_path) as f:
            label_data = json.load(f)

        base_path = os.path.splitext(file_path)[0]
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            potential_path = base_path + ext
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        if image_path is None:
            logger.warning(f"Image for {file_path} not found. Skipping.")
            continue

        record = {
            "file_name": image_path,
            "image_id": i,
            "height": label_data["imageHeight"],
            "width": label_data["imageWidth"],
            "annotations": []
        }

        img_cats = set()
        for shape in label_data["shapes"]:
            raw_label = shape["label"]
            label = MAPPING.get(raw_label)
            if label is None:
                continue

            points = np.asarray(shape["points"])
            
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            elif len(points) < 3:
                continue

            img_cats.add(CATEGORY_IDS[label])

            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin

            ann = {
                "bbox": [float(xmin), float(ymin), float(width), float(height)],
                "bbox_mode": detectron2.structures.BoxMode.XYWH_ABS,
                "category_id": CATEGORY_IDS[label],
                "segmentation": [points.flatten().tolist()],
                "iscrowd": 0
            }
            record["annotations"].append(ann)

        dataset_dicts.append(record)

        # Résolution du non-déterminisme des sets en Python : on trie avant de choisir
        primary_category = sorted(list(img_cats))[0] if img_cats else -1
        image_categories.append(primary_category)

    categories = [{"id": CATEGORY_IDS[label], "name": label} for label in FINAL_LABELS]

    return dataset_dicts, np.array(image_categories), categories


# --- Affichage des courbes d'apprentissage ---
def plot_best_learning_curves(metrics_path, save_path=None):
    """
    Lit un fichier metrics.json de Detectron2 et trace les courbes d'apprentissage
    (Loss d'entraînement et AP de validation) sur un seul graphique.
    """
    logger = logging.getLogger("detectron2")
    if not os.path.exists(metrics_path):
        logger.error(f"Le fichier '{metrics_path}' n'a pas été trouvé pour le tracé.")
        return

    metrics_data = []
    with open(metrics_path, 'r') as f:
        for line in f:
            try:
                metrics_data.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not metrics_data:
        logger.warning("Aucune donnée de métrique trouvée.")
        return

    train_metrics = [m for m in metrics_data if 'total_loss' in m]
    eval_metrics = [m for m in metrics_data if 'bbox/AP' in m]

    df = pd.DataFrame(train_metrics)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fold_name = os.path.basename(os.path.dirname(metrics_path))
    fig.suptitle(f"Courbes d'apprentissage ({fold_name})", fontsize=16)

    # --- Graphique : Perte d'entraînement ---
    if not df.empty and 'iteration' in df.columns and 'total_loss' in df.columns:
        ax1.plot(df['iteration'], df['total_loss'], label='Total Loss (Entraînement)', color='tab:blue', alpha=0.4)
        if len(df) > 10:
            ax1.plot(df['iteration'], df['total_loss'].rolling(window=10).mean(), label='Moyenne mobile (10 it.)', linestyle='-', color='tab:blue', linewidth=2)
    ax1.set_xlabel('Itération')
    ax1.set_ylabel('Perte (Loss)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # --- Axe Y secondaire pour l'Average Precision (AP) ---
    ax1_ap = ax1.twinx()
    if eval_metrics:
        eval_df = pd.DataFrame(eval_metrics)
        if 'iteration' in eval_df.columns and 'bbox/AP' in eval_df.columns:
            eval_df['bbox/AP_val'] = eval_df['bbox/AP'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
            ax1_ap.plot(eval_df['iteration'], eval_df['bbox/AP_val'], label='bbox/AP (Validation)', color='tab:red', marker='o', linestyle='-', linewidth=2)
            
    ax1_ap.set_ylabel('bbox/AP', color='tab:red')
    ax1_ap.tick_params(axis='y', labelcolor='tab:red')

    # --- Finalisation et Sauvegarde ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_ap.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique des courbes sauvegardé dans : {save_path}")
    plt.show()


# --- Global variable to cache the background image ---
# _background_image = None # Désactivé car la soustraction de fond réduisait la précision.

def custom_mapper_with_albumentations(dataset_dict):
    """
    A custom data mapper that uses Albumentations for data augmentation.
    This version handles non-rigid transformations (like GridDistortion) by
    converting segmentations to masks, transforming the masks, and then
    reconstructing the annotations.
    La soustraction de fond a été désactivée car elle semblait réduire la précision.
    """
    # global _background_image # Désactivé

    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR").copy()

    # --- Background Subtraction (Désactivé car réduisait la précision) ---
    # if _background_image is None:
    #     background_path = os.path.join(DATASET_PATH, "background.png")
    #     if os.path.exists(background_path):
    #         _background_image = cv2.imread(background_path, cv2.IMREAD_COLOR)
    #         logging.getLogger("detectron2").info(f"Loaded background image from {background_path}")
    # if _background_image is not None:
    #     bg_resized = cv2.resize(_background_image, (image.shape[1], image.shape[0]))
    #     image = cv2.absdiff(image, bg_resized)

    H, W, _ = image.shape

    # 1. Convert polygon segmentations to binary masks
    masks = []
    category_ids = []
    for ann in dataset_dict.get("annotations", []):
        mask = np.zeros((H, W), dtype=np.uint8)
        
        if not isinstance(ann["segmentation"], list) or not ann["segmentation"]:
            continue
            
        poly = np.array(ann["segmentation"][0]).reshape(-1, 2)
        
        # fillPoly requires integer coordinates
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        masks.append(mask)
        category_ids.append(ann["category_id"])

    # --- NOUVEAU : Copy-Paste Intra-image pour multiplier les petites bulles ---
    # Probabilité d'appliquer l'augmentation (ex: 30% du temps)
    if random.random() < 0.3 and len(masks) > 0:
        # On cherche les petites bulles (ex: aire inférieure à 1000 pixels)
        small_indices = [idx for idx, m in enumerate(masks) if 0 < np.sum(m > 0) < 1000]
        
        if small_indices:
            # Créer une carte globale des objets pour éviter de coller sur une bulle existante
            global_mask = np.any(np.array(masks) > 0, axis=0)
            
            # On choisit aléatoirement une petite bulle à dupliquer (1 à 3 fois)
            for _ in range(random.randint(1, 3)):
                src_idx = random.choice(small_indices)
                src_mask = masks[src_idx]
                src_cat = category_ids[src_idx]
                
                # Trouver la bounding box de la bulle source
                rows = np.any(src_mask, axis=1)
                cols = np.any(src_mask, axis=0)
                if not rows.any() or not cols.any(): continue
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                bh, bw = y_max - y_min + 1, x_max - x_min + 1
                
                # Vérifier qu'on a la place de la coller ailleurs
                if H - bh > 0 and W - bw > 0:
                    # Essayer de trouver une zone vide (jusqu'à 10 tentatives)
                    for _attempt in range(10):
                        dy = random.randint(0, H - bh - 1)
                        dx = random.randint(0, W - bw - 1)
                        
                        # Extraire le patch du masque de la source
                        roi_mask = src_mask[y_min:y_max+1, x_min:x_max+1]
                        
                        # Si la zone d'atterrissage ne croise aucune bulle existante
                        if not np.any(global_mask[dy:dy+bh, dx:dx+bw][roi_mask > 0]):
                            roi_img = image[y_min:y_max+1, x_min:x_max+1]
                            
                            # Coller l'image
                            image[dy:dy+bh, dx:dx+bw] = np.where(roi_mask[..., None] > 0, roi_img, image[dy:dy+bh, dx:dx+bw])
                            
                            # Ajouter le nouveau masque
                            new_mask = np.zeros((H, W), dtype=np.uint8)
                            new_mask[dy:dy+bh, dx:dx+bw] = roi_mask
                            masks.append(new_mask)
                            category_ids.append(src_cat)
                            
                            # Mettre à jour la carte globale pour les prochains collages
                            global_mask[dy:dy+bh, dx:dx+bw] = np.logical_or(global_mask[dy:dy+bh, dx:dx+bw], roi_mask > 0)
                            break # Succès, on passe à la copie suivante

    # 2. Define Albumentations pipeline for images and masks
    # Base transforms (toujours appliqués pour maintenir la géométrie requise)
    base_transforms = [
        # --- Pour 1024x1024 ---
        #A.LongestMaxSize(max_size=1024),
        #A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=0),
        # --- Pour 800x800 ---
        A.LongestMaxSize(max_size=800),
        A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=0),
    ]

    # TEST DE LA PIPELINE COMBINÉE ULTIME
    test_transform = [
        # 1. Augmentations Géométriques (Cumulables)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3, border_mode=cv2.BORDER_CONSTANT), # Déformation fluide
        
        # 2. Groupe Lumière/Contraste (Choisit une seule méthode parmi les 3)
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.RandomGamma(p=1.0),
        ], p=0.5),
        
        # 3. Accentuation des contours (Indépendant)
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2), 
        
        # 4. Groupe Destructeur (Choisit une seule méthode de dégradation)
        A.OneOf([
            A.MotionBlur(p=1.0),
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        ], p=0.2),
        
        # 5. Occlusions (Indépendant)
        # A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2), #valeurs standard
        A.CoarseDropout(max_holes=8, max_height=88, max_width=123, fill_value=0, p=0.2), #800
        #A.CoarseDropout(max_holes=8, max_height=113, max_width=158, fill_value=0, p=0.2), #1024
    ]
    transform = A.Compose(base_transforms + test_transform)

    try:
        # 3. Apply the transformation
        # Conversion BGR -> RGB pour Albumentations (crucial pour CLAHE et modifications de teintes)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=image_rgb, masks=masks)
        image_transformed_rgb = transformed['image']
        
        # Conversion RGB -> BGR pour que Detectron2 retrouve le format attendu
        image_transformed = cv2.cvtColor(image_transformed_rgb, cv2.COLOR_RGB2BGR)
        masks_transformed = transformed['masks']
    except (ValueError, IndexError) as e:
        # This can happen if augmentations result in empty masks/images
        logging.getLogger("detectron2").warning(f"Skipping an image due to augmentation error: {e}")
        return None

    # 4. Reconstruct annotations from transformed masks (Lossless BitMask Approach)
    gt_masks_list = []
    gt_boxes_list = []
    gt_classes_list = []

    for i, mask_t in enumerate(masks_transformed):
        binary = (mask_t > 127)
        if not binary.any():
            continue

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        w = float(x_max - x_min + 1)
        h = float(y_max - y_min + 1)
        if w <= 1 or h <= 1:
            continue

        gt_masks_list.append(binary)
        gt_boxes_list.append([float(x_min), float(y_min), float(x_max + 1), float(y_max + 1)])
        gt_classes_list.append(category_ids[i])

    if not gt_masks_list:
        return None

    # 5. Finalize the dataset dictionary for Detectron2
    H_out, W_out = image_transformed.shape[:2]
    instances = Instances((H_out, W_out))
    instances.gt_masks = BitMasks(torch.stack([torch.from_numpy(m.astype(np.bool_)) for m in gt_masks_list]))
    instances.gt_boxes = Boxes(torch.tensor(gt_boxes_list, dtype=torch.float32))
    instances.gt_classes = torch.tensor(gt_classes_list, dtype=torch.int64)

    dataset_dict.pop("annotations", None)
    dataset_dict["image"] = torch.as_tensor(image_transformed.transpose(2, 0, 1).astype("float32"))
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    
    return dataset_dict


# ── Boundary-Aware Mask Loss ──────────────────────────────────────────────────
import torch.nn.functional as F
from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss, mask_rcnn_inference
import detectron2.modeling.roi_heads.mask_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from fvcore.nn import sigmoid_focal_loss_jit

def boundary_weighted_mask_loss(pred_mask_logits, instances, vis_period: int = 0):
    """
    Remplace la perte standard mask_rcnn_loss par une version qui augmente le poids
    des pixels de contour via une carte de poids.
    """
    BOUNDARY_WEIGHT = 3.0   # Les erreurs sur les bords comptent 3 fois plus
    BOUNDARY_DILATION = 3   # Épaisseur du bord considéré (pixels)

    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks   = pred_mask_logits.size(0)
    mask_side_len     = pred_mask_logits.size(2)

    gt_classes, gt_masks_list = [], []
    for inst_per_image in instances:
        if len(inst_per_image) == 0:
            continue
        gt_classes.append(inst_per_image.gt_classes)
        gt_masks_list.append(
            inst_per_image.gt_masks.crop_and_resize(
                inst_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device, dtype=torch.float32)
        )

    if not gt_classes:
        return pred_mask_logits.sum() * 0.0

    gt_classes = torch.cat(gt_classes, dim=0)
    gt_masks   = torch.cat(gt_masks_list, dim=0)  # (N, H, W) ∈ {0,1}

    if cls_agnostic_mask:
        pred_masks = pred_mask_logits[:, 0]
    else:
        indices    = torch.arange(total_num_masks, device=gt_classes.device)
        pred_masks = pred_mask_logits[indices, gt_classes]  # (N, H, W)

    # ── Construit la carte des poids des contours ────────────────────────
    kernel_size = 2 * BOUNDARY_DILATION + 1
    gt_masks_4d  = gt_masks.unsqueeze(1)           # (N,1,H,W)
    eroded       = -F.max_pool2d(
        -gt_masks_4d,
        kernel_size=kernel_size,
        stride=1,
        padding=BOUNDARY_DILATION
    ).squeeze(1)                                    # (N,H,W)

    boundary_map = (gt_masks - eroded).clamp(0, 1)  # 1 sur les contours
    weight_map   = 1.0 + (BOUNDARY_WEIGHT - 1.0) * boundary_map  # (N,H,W)

    # ── Calcule l'erreur pondérée ────────────────────────────────────────
    loss = F.binary_cross_entropy_with_logits(
        pred_masks, gt_masks, reduction="none"
    )                                               # (N,H,W)
    loss = (loss * weight_map).mean()

    return loss

from detectron2.modeling.roi_heads.mask_head import BaseMaskRCNNHead, ROI_MASK_HEAD_REGISTRY as MASK_HEAD_REGISTRY

@MASK_HEAD_REGISTRY.register()
class BoundaryAwareMaskHead(detectron2.modeling.roi_heads.mask_head.MaskRCNNConvUpsampleHead):
    def forward(self, x, instances):
        x = self.layers(x)
        if self.training:
            return {"loss_mask": boundary_weighted_mask_loss(x, instances)}
        else:
            mask_rcnn_inference(x, instances)
            return instances

# --- Focal Loss pour la tête de Classification (Box Head) ---
class FocalFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # Hyperparamètres classiques de la Focal Loss
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0

        # Poids par classe (à adapter selon vos fréquences). 
        # Ordre : [detached (0), occludedAttached (1), attached (2)]
        # Ici, on donne un poids 3x plus fort à 'detached'.
        self.class_weights = [3.0, 1.0, 1.0]

    def losses(self, predictions, proposals):
        """ Remplace la Cross-Entropy standard par la Sigmoid Focal Loss """
        scores, box_deltas = predictions
        
        # Récupère le dictionnaire de pertes standard (gère la perte de régression bbox de manière robuste)
        losses_dict = super().losses(predictions, proposals)
        
        if not len(proposals):
            return losses_dict

        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        num_classes = scores.shape[1] - 1

        # Création du target one-hot (sans la classe background)
        gt_classes_one_hot = F.one_hot(gt_classes, num_classes=num_classes + 1)[:, :-1].float()
        pred_class_logits_fg = scores[:, :-1]

        # Calcul de la Focal Loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits_fg,
            gt_classes_one_hot,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="none", # "none" pour appliquer les poids avant de faire la somme
        )
        
        # Application des poids par classe
        device = loss_cls.device
        weight_tensor = torch.tensor(self.class_weights, device=device).view(1, -1)
        loss_cls = loss_cls * weight_tensor
        
        # Normalisation par le nombre de vrais objets (foreground)
        num_fg = max(1.0, gt_classes_one_hot.sum().item())
        loss_cls = loss_cls.sum() / num_fg

        losses_dict["loss_cls"] = loss_cls
        return losses_dict

    def predict_probs(self, predictions, proposals):
        """ Puisqu'on entraîne avec Sigmoid, on doit inférer avec Sigmoid (et non Softmax) """
        scores = torch.sigmoid(predictions[0][:, :-1])
        bg_scores = 1.0 - scores.max(dim=1, keepdim=True)[0] # Score du fond
        probs = torch.cat([scores, bg_scores], dim=1)
        return probs.split([len(p) for p in proposals], dim=0)

# Tête ROI pour la Boundary Loss SEULEMENT (utilise la classification standard)
@ROI_HEADS_REGISTRY.register()
class BoundaryAwareROIHeads(StandardROIHeads):
    pass

# Tête ROI qui combine Boundary Loss ET Focal Loss pour la classification
@ROI_HEADS_REGISTRY.register()
class BoundaryAndFocalROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        # Surcharge du Box Predictor par notre version Focal Loss
        ret["box_predictor"] = FocalFastRCNNOutputLayers(cfg, ret["box_head"].output_shape)
        return ret
# ─────────────────────────────────────────────────────────────────────────────

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper_with_albumentations)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        # Surcharge pour ajouter l'évaluation périodique et l'arrêt précoce
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = self.cfg.DATALOADER.NUM_WORKERS 
        
        ret = super().build_hooks()
        
        # Le BestCheckpointer exécute déjà une évaluation, il n'est donc pas
        # nécessaire d'ajouter un EvalHook séparé. Les résultats de l'évaluation
        # du BestCheckpointer seront disponibles pour les autres hooks.

        # Hook pour sauvegarder le meilleur modèle (qui inclut l'évaluation)
        ret.append(hooks.BestCheckpointer(
            cfg.TEST.EVAL_PERIOD, self.checkpointer, EARLY_STOPPING_METRIC, "max"
        ))

        # Hook pour l'arrêt précoce
        ret.append(EarlyStoppingHook(
            patience=EARLY_STOPPING_PATIENCE,
            metric=EARLY_STOPPING_METRIC
        ))
        
        return ret


# --- Main Training Logic ---

def main():
    setup_logger()
    logger = logging.getLogger("detectron2")

    # --- Reproductibilité Stricte ---
    logger.info("Setting global seed for reproducibility...")
    seed_all_rng(RANDOM_STATE)
    
    # Forcer PyTorch à être déterministe sur le GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Prepare unified dataset
    logger.info("Preparing unified dataset from labelme files...")
    all_labelme_files = get_all_labelme_files(DATASET_PATH)
    all_dataset_dicts, image_categories, categories = create_detectron2_dataset_from_labelme(all_labelme_files)
    
    # --- Correction de la Stratification ---
    from collections import Counter
    logger.info("--- Analyse de la distribution pour la stratification ---")
    category_counts = Counter(image_categories)
    id_to_name = {c["id"]: c["name"] for c in categories}
    id_to_name[-1] = "NO_CATEGORY"

    stratify_categories = np.copy(image_categories)
    rare_category_group_id = -2
    was_modified = False

    for cat_id, count in sorted(category_counts.items()):
        cat_name = id_to_name.get(cat_id, f"ID_{cat_id}_INCONNU")
        logger.info(f"Catégorie '{cat_name}': {count} images")
        if count > 0 and count < N_SPLITS:
            was_modified = True
            stratify_categories[image_categories == cat_id] = rare_category_group_id
            logger.warning(f"  -> La catégorie '{cat_name}' a seulement {count} image(s). Regroupée.")

    if was_modified:
        logger.info("Stratification ajustée pour les classes rares.")
    # ---------------------------------------

    # 2. K-Fold Cross-validation loop
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_final_metrics = {}
    
    if -1 in image_categories:
        logger.warning("Some images have no categories. Stratification might be suboptimal.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_dataset_dicts)), stratify_categories)):
        # Condition pour ne lancer que le fold 4, comme demandé.
        # Parfait pour tester rapidement et itérativement des augmentations.
        if (fold + 1) != 4:
            continue

        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        logger.info(f"--- Starting Fold {fold + 1}/{N_SPLITS} ---")

        # Create train/val datasets for this fold
        train_dicts = [all_dataset_dicts[i] for i in train_idx]
        val_dicts = [all_dataset_dicts[i] for i in val_idx]
        
        train_dataset_name = f"bubbleid_train_fold_{fold + 1}"
        val_dataset_name = f"bubbleid_val_fold_{fold + 1}"
        
        # Nettoyer les catalogues au cas où le script est exécuté plusieurs fois
        for d in [train_dataset_name, val_dataset_name]:
            if d in DatasetCatalog.list():
                DatasetCatalog.remove(d)
            if d in MetadataCatalog.list():
                MetadataCatalog.remove(d)
            
        DatasetCatalog.register(train_dataset_name, lambda d=train_dicts: d)
        MetadataCatalog.get(train_dataset_name).set(thing_classes=[c['name'] for c in categories])

        DatasetCatalog.register(val_dataset_name, lambda d=val_dicts: d)
        MetadataCatalog.get(val_dataset_name).set(thing_classes=[c['name'] for c in categories])

        # 3. Configure and train
        cfg = get_cfg()
        
        # --- Choix du Backbone (Squelette du réseau) ---
        # Décommentez UNE SEULE des deux lignes ci-dessous pour choisir le modèle :
        MODEL_YAML = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # ResNet-50 (Modèle classique, plus rapide)
        #MODEL_YAML = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" # ResNeXt-101 (Modèle profond, plus précis mais plus lent)
        
        cfg.merge_from_file(model_zoo.get_config_file(MODEL_YAML))
        cfg.OUTPUT_DIR = fold_output_dir
        
        cfg.DATASETS.TRAIN = (train_dataset_name,)
        cfg.DATASETS.TEST = (val_dataset_name,)
        cfg.DATALOADER.NUM_WORKERS = 8 # Augmenté pour accélérer les augmentations Albumentations sur le processeur
        
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_YAML)
        
        cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
        cfg.SOLVER.BASE_LR = LR
        cfg.SOLVER.MAX_ITER = MAX_ITER
        
        # Une descente du Learning Rate en "escalier" plus douce
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.SOLVER.STEPS = (5000, 6000) # BATCH_SIZE = 4
        #cfg.SOLVER.STEPS = (10000, 12000) # BATCH_SIZE = 2
        cfg.SOLVER.GAMMA = 0.333 # Divise RÉELLEMENT le LR par 3 à chaque palier (freinage doux)
        
        # --- Configuration explicite du Warmup ---
        cfg.SOLVER.WARMUP_ITERS = 1000          # 2000 si BATCH_SIZE = 2
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000   # 1.0 / 2000 si BATCH_SIZE = 2
        cfg.SOLVER.WARMUP_METHOD = "linear"     # Montée linéaire (linéaire, constante ou step)
        cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
        cfg.SOLVER.AMP.ENABLED = True
        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(FINAL_LABELS)
        
        # --- OPTIMISATION DE LA PRÉCISION DES MASQUES (AP/segm) ---
        # Augmente la résolution interne des masques pour des contours plus fins.
        # Le réseau dessine les masques sur une grille interne avant de les redimensionner.
        # Une grille plus grande (28x28) produit des contours plus lisses et précis,
        # surtout pour les petits objets ronds.
        # Décommentez la ligne que vous souhaitez tester.
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14 # Valeur par défaut, plus rapide.
        #cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28 # Recommandé pour améliorer AP/segm, légèrement plus lent.

        # Activation de la perte sensible aux contours (Boundary Loss)
        if USE_BOUNDARY_LOSS:
            logger.info("Utilisation de la Boundary Loss pour les masques.")
            cfg.MODEL.ROI_MASK_HEAD.NAME = "BoundaryAwareMaskHead"
        else:
            logger.info("Utilisation de la Mask Loss standard.")
        
        # Activation optionnelle de la Focal Loss pour la classification
        if USE_FOCAL_LOSS:
            logger.info("Utilisation de la Focal Loss pour la classification des boîtes.")
            cfg.MODEL.ROI_HEADS.NAME = "BoundaryAndFocalROIHeads"
        else:
            logger.info("Utilisation de la Cross Entropy standard pour la classification des boîtes.")
            cfg.MODEL.ROI_HEADS.NAME = "BoundaryAwareROIHeads"

        # OPTIMISATIONS : Ratios d'ancres pour les amas de bulles et tailles d'ancres plus petites
        #cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] #classique
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.51, 0.89, 1.21]] #obtenu avec analyse_taille
        
        # CORRECTION : Une liste par niveau de FPN (5 niveaux) pour capturer les micro-bulles
        # --- Configuration pour 1024x1024 ---
        #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[31], [83], [132], [219], [330]]
        
        # --- Configuration pour 800x800 ---
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[24], [65], [103], [171], [258]] # obtenu avec analyse_taille

        cfg.TEST.EVAL_PERIOD = EVAL_PERIOD

        # --- Test-Time Augmentation (TTA) ---
        # Active l'augmentation au moment du test (évaluation) pour améliorer la robustesse.
        # Le modèle prédit sur l'image originale et sa version retournée horizontalement,
        # puis les résultats sont fusionnés.
        # Cela ralentit l'évaluation mais peut augmenter l'AP de 1-2 points.
        cfg.TEST.AUG.ENABLED = True
        cfg.TEST.AUG.FLIP = True

        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
            f.write(cfg.dump())

        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)
        
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Training stopped with an exception: {e}", exc_info=True)

        best_fold_ap = -1.0
        try:
            with open(os.path.join(fold_output_dir, "metrics.json")) as f:
                metrics_lines = f.readlines()
            
            for line in metrics_lines:
                metrics = json.loads(line)
                if EARLY_STOPPING_METRIC in metrics and isinstance(metrics[EARLY_STOPPING_METRIC], (float, int)):
                    current_ap = metrics[EARLY_STOPPING_METRIC]
                    if current_ap > best_fold_ap:
                        best_fold_ap = current_ap
                        
        except FileNotFoundError:
            logger.warning(f"Le fichier metrics.json n'a pas été trouvé pour le fold {fold + 1}. "
                           f"Score AP considéré comme -1.")
        
        all_final_metrics[f"fold_{fold + 1}"] = best_fold_ap
        logger.info(f"Fold {fold+1} Best AP: {best_fold_ap:.4f}")

    # 4. Final Report
    logger.info("--- K-Fold Cross-Validation Finished ---")
    
    best_fold_name = ""
    best_fold_ap = -1
    for fold_name, ap in all_final_metrics.items():
        logger.info(f"Final AP for {fold_name}: {ap:.4f}")
        if ap > best_fold_ap:
            best_fold_ap = ap
            best_fold_name = fold_name
            
    if best_fold_name:
        logger.info(f"\nBest fold was {best_fold_name} with AP: {best_fold_ap:.4f}")
        # Copier le meilleur modèle du meilleur "fold" dans le dossier principal
        BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model_overall")
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        
        best_model_src_path = os.path.join(OUTPUT_DIR, best_fold_name, "model_best.pth")
        
        if os.path.exists(best_model_src_path):
            best_model_dst_path = os.path.join(BEST_MODEL_DIR, "model_final.pth")
            config_dst_path = os.path.join(BEST_MODEL_DIR, "config.yaml")

            with open(best_model_src_path, "rb") as f_src, open(best_model_dst_path, "wb") as f_dst:
                f_dst.write(f_src.read())
                
            config_src_path = os.path.join(OUTPUT_DIR, best_fold_name, "config.yaml")
            if os.path.exists(config_src_path):
                with open(config_src_path, "r") as f_src, open(config_dst_path, "w") as f_dst:
                    f_dst.write(f_src.read())

            # --- Sauvegarde des métriques du meilleur modèle dans un fichier texte ---
            metrics_src_path = os.path.join(OUTPUT_DIR, best_fold_name, "metrics.json")
            metrics_txt_path = os.path.join(BEST_MODEL_DIR, "evaluation_metrics.txt")
            
            if os.path.exists(metrics_src_path):
                best_metrics_dict = {}
                with open(metrics_src_path, "r") as f_src:
                    for line in f_src:
                        try:
                            m = json.loads(line)
                            if EARLY_STOPPING_METRIC in m and m[EARLY_STOPPING_METRIC] == best_fold_ap:
                                best_metrics_dict = m
                        except json.JSONDecodeError:
                            pass
                
                with open(metrics_txt_path, "w", encoding="utf-8") as f_txt:
                    f_txt.write(f"=== RAPPORT DU MEILLEUR MODÈLE ===\n")
                    f_txt.write(f"Fold d'origine : {best_fold_name}\n")
                    f_txt.write(f"Score principal ({EARLY_STOPPING_METRIC}) : {best_fold_ap:.4f}\n\n")
                    
                    if best_metrics_dict:
                        f_txt.write(f"--- Détail des métriques (Itération {best_metrics_dict.get('iteration', 'Inconnue')}) ---\n")
                        for k, v in sorted(best_metrics_dict.items()):
                            f_txt.write(f"{k} : {v}\n")
                    else:
                        f_txt.write("Détails supplémentaires non trouvés dans metrics.json.\n")
                
                logger.info(f"Best evaluation metrics saved to {metrics_txt_path}")

            logger.info(f"Best overall model copied to {BEST_MODEL_DIR}")
            
            # --- Affichage et sauvegarde du graphique ---
            if os.path.exists(metrics_src_path):
                plot_save_path = os.path.join(BEST_MODEL_DIR, "learning_curves.png")
                logger.info("Génération du graphique des courbes d'apprentissage...")
                plot_best_learning_curves(metrics_src_path, plot_save_path)
        else:
            logger.warning(f"Could not find best model file at {best_model_src_path}")


if __name__ == "__main__":
    main()