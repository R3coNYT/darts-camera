"""
Train + export a YOLO11n-pose model to detect 5 calibration keypoints on a dartboard.

Usage
-----
    python scripts/train_yolo.py

Prerequisites
-------------
    pip install ultralytics

Dataset structure (create this before running)
-----------------------------------------------
    datasets/dartboard/
        images/
            train/   ← vos photos JPG/PNG
            val/     ← ~20 % des photos pour validation
        labels/
            train/   ← fichiers .txt au format YOLO pose (générés par Roboflow / CVAT)
            val/

Format d'un fichier label (.txt)
---------------------------------
One line per dartboard in the image :
    <class_id> <cx> <cy> <w> <h>  <kp0_x> <kp0_y> <kp0_vis>  ...  <kp4_x> <kp4_y> <kp4_vis>
All values are normalised (0..1).  class_id = 0.

Keypoints order
---------------
    0 : centre (bull)
    1 : haut  du double extérieur (12h)
    2 : droite du double extérieur (3h)
    3 : bas   du double extérieur (6h)
    4 : gauche du double extérieur (9h)

Tip : use Roboflow (free tier) to annotate and export directly in "YOLOv8 Pose" format.

Résultat
--------
The trained weights are saved to runs/pose/dartboard/weights/best.pt.
Copy that file to models/dartboard-pose.pt so the detector picks it up automatically.
"""

import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_YAML    = Path(__file__).parent / "dartboard-pose.yaml"
MODEL_BASE   = "yolo11n-pose.pt"   # nano pose, ~7 MB, fastest
OUTPUT_NAME  = "dartboard"
EPOCHS       = 100
IMG_SIZE     = 640
BATCH        = 8                   # réduire à 4 si mémoire insuffisante
DEVICE       = ""                  # "" = auto (GPU si dispo, sinon CPU)
PATIENCE     = 20                  # early stopping si pas d'amélioration

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    from ultralytics import YOLO  # noqa: PLC0415

    print(f"[train_yolo] Modèle de base : {MODEL_BASE}")
    print(f"[train_yolo] Dataset        : {DATA_YAML}")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        name=OUTPUT_NAME,
        project="runs/pose",
        # Augmentations utiles pour varier la luminosité / perspective
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=10,       # rotation légère
        perspective=0.001,
        flipud=0.0,       # ne pas retourner verticalement (haut ≠ bas)
        fliplr=0.5,       # retournement horizontal ok si flip_idx=[] → désactiver si asymétrique
        mosaic=0.5,
    )

    print("\n[train_yolo] Entraînement terminé.")
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[train_yolo] Meilleurs poids : {best}")
    return best


# ---------------------------------------------------------------------------
# Export to ONNX (optionnel, pour déploiement sans PyTorch)
# ---------------------------------------------------------------------------

def export_onnx(weights_path: Path):
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights_path))
    model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)
    onnx_path = weights_path.with_suffix(".onnx")
    print(f"[train_yolo] Exporté ONNX : {onnx_path}")


# ---------------------------------------------------------------------------
# Copy best.pt → models/dartboard-pose.pt
# ---------------------------------------------------------------------------

def install_model(weights_path: Path):
    dest = Path(__file__).parent.parent / "models" / "dartboard-pose.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(weights_path, dest)
    print(f"[train_yolo] Modèle installé : {dest}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    best = train()
    install_model(best)
    # Décommenter pour exporter en ONNX aussi :
    # export_onnx(best)
    print("\nDone. Lance ton app et déclenche une calibration — le modèle YOLO sera utilisé.")
