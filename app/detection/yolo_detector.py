"""
app/detection/yolo_detector.py

Wrapper autour de YOLOv8 (ultralytics) pour la détection d'EPI et de
comportements dangereux frame par frame.

Classes du modèle (9 au total) :
    0: Hardhat        4: NO-Safety Vest
    1: Mask            5: Person
    2: NO-Hardhat      6: Safety Vest
    3: NO-Mask         7: Harness / NO-Harness / Phone / Machine
    (l'ordre exact dépend de votre data.yaml — vérifiez model.names au chargement)
"""

from pathlib import Path
from ultralytics import YOLO


class YoloDetector:
    """
    Encapsule un modèle YOLOv8 fine-tuné pour la détection d'EPI.

    Utilisation :
        detector = YoloDetector("models/best.pt")
        detections = detector.detect(frame)
        # detections est une liste de dicts :
        # [{"class_name": "Person", "class_id": 5, "confidence": 0.91,
        #   "bbox": (x1, y1, x2, y2)}, ...]
    """

    def __init__(self, model_path: str = "models/best.pt", confidence_threshold: float = 0.5, device: str = "cpu"):
        """
        Args:
            model_path: chemin vers le fichier .pt du modèle fine-tuné.
            confidence_threshold: seuil minimal de confiance pour garder une détection.
            device: "cpu" ou "cuda" (ou "0" pour le premier GPU) selon la machine.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}. "
                "Vérifiez que best.pt est bien dans le dossier models/."
            )

        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.class_names = self.model.names  # dict {id: nom_de_classe}

        print(f"[YoloDetector] Modèle chargé : {model_path}")
        print(f"[YoloDetector] Classes disponibles : {self.class_names}")

    def detect(self, frame) -> list[dict]:
        """
        Détecte les objets/personnes/EPI sur une seule frame (image OpenCV, format BGR numpy array).

        Returns:
            Liste de dictionnaires, un par détection retenue.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_name": self.class_names[class_id],
                    "class_id": class_id,
                    "confidence": round(confidence, 3),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        return detections

    def get_class_names(self) -> dict:
        """Retourne le dictionnaire {id: nom} des classes du modèle."""
        return self.class_names
