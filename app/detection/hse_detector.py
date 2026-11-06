"""
app/detection/hse_detector.py

Détection combinée pour le POC HSE, sur une frame à la fois :

1. Modèle fine-tuné (models/best.pt, YOLOv8s) + ByteTrack : personnes, EPI
   (Hardhat / NO-Hardhat, Safety Vest / NO-Safety Vest, Harness / NO-Harness)
   et engins (Machine), avec un identifiant de suivi temporaire.
2. Détecteur spécialisé téléphone : modèle YOLOv8n pré-entraîné COCO (classe
   "cell phone"), appliqué sur le recadrage de chaque personne suivie.

Pourquoi un second modèle pour le téléphone ? Mesuré sur les vidéos de test :
la classe "Phone" de best.pt n'est jamais détectée sur just_phone.mp4, même à
une confiance de 0,25, alors que le modèle COCO la détecte (0,56 à 0,74) sans
faux positif sur test1.mp4. Le recadrage par personne agrandit le téléphone
(objet petit) et limite les détections à des téléphones tenus par quelqu'un.

Les poids COCO sont publics (aucune donnée du site OCP), ce qui respecte la
contrainte de confidentialité du cahier des charges.

Les .pt sont désérialisés avec pickle : leur empreinte SHA-256 est vérifiée
avant chargement (config/model_integrity.json, app/security/model_integrity.py).
"""

from pathlib import Path

from ultralytics import YOLO

from app.config import resolve
from app.security.model_integrity import verify_model

# Classes attendues dans best.pt, et comportement associé
EXPECTED_PPE_CLASSES = {
    "Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest",
    "Harness", "NO-Harness", "Phone", "Machine",
}
COCO_PHONE_CLASS = "cell phone"


class HSEDetector:
    def __init__(self, rules: dict):
        models = rules["models"]
        inference = rules["inference"]
        self.device = inference.get("device", "cpu")
        self.imgsz = inference.get("imgsz", 960)
        self.confidence = inference.get("confidence", 0.35)
        # Le tracker reçoit aussi les détections faibles (2e étage de ByteTrack,
        # utile pour garder une personne partiellement masquée) ; seules les
        # personnes sont conservées sous le seuil principal.
        self.track_confidence = inference.get("track_confidence", 0.15)
        self.tracker_config = str(resolve(inference.get("tracker", "config/bytetrack_hse.yaml")))
        self.phone_imgsz = inference.get("phone_imgsz", 416)
        self.phone_confidence = inference.get("phone_confidence", 0.3)
        self.phone_max_persons = inference.get("phone_max_persons", 6)
        self.phone_min_person_height_px = inference.get("phone_min_person_height_px", 80)

        ppe_path = resolve(models["ppe"])
        if not ppe_path.is_file():
            raise FileNotFoundError(f"Modèle EPI introuvable : {ppe_path}")
        verify_model(ppe_path)  # refus si poids inconnus ou modifiés (ModelIntegrityError)
        self.model_paths = [ppe_path]
        self.ppe_model = YOLO(str(ppe_path))
        self.ppe_classes = set(self.ppe_model.names.values())
        missing = EXPECTED_PPE_CLASSES - self.ppe_classes
        if missing:
            print(f"[HSEDetector] Classes absentes du modèle EPI : {sorted(missing)}")

        self.phone_model = None
        self.phone_class_ids = []
        phone_path = models.get("phone")
        if phone_path and resolve(phone_path).is_file():
            verify_model(phone_path)
            self.model_paths.append(resolve(phone_path))
            self.phone_model = YOLO(str(resolve(phone_path)))
            self.phone_class_ids = [
                class_id for class_id, name in self.phone_model.names.items() if name == COCO_PHONE_CLASS
            ]
            if not self.phone_class_ids:
                print(f"[HSEDetector] '{COCO_PHONE_CLASS}' absent de {phone_path} : détecteur téléphone désactivé.")
                self.phone_model = None
        else:
            print("[HSEDetector] Détecteur téléphone spécialisé absent : seule la classe 'Phone' de best.pt est utilisée.")

    def verify_models(self):
        """
        Revérifie l'empreinte des poids sur disque (environ 0,2 s). Les modèles
        restent chargés en mémoire d'une analyse à l'autre : ce contrôle, fait à
        chaque démarrage d'analyse, détecte un fichier remplacé depuis leur
        chargement (ModelIntegrityError).
        """
        for path in self.model_paths:
            verify_model(path)

    def capabilities(self) -> dict:
        """Décrit, pour l'interface, ce que les poids chargés permettent réellement de détecter."""
        return {
            "casque": "Hardhat" in self.ppe_classes and "NO-Hardhat" in self.ppe_classes,
            "gilet": "Safety Vest" in self.ppe_classes and "NO-Safety Vest" in self.ppe_classes,
            "harnais": "Harness" in self.ppe_classes and "NO-Harness" in self.ppe_classes,
            "telephone": self.phone_model is not None or "Phone" in self.ppe_classes,
            "telephone_source": "YOLOv8n COCO (recadrage personne)" if self.phone_model else "best.pt (classe Phone, peu fiable)",
            "engin": "Machine" in self.ppe_classes,
            "classes_modele": sorted(self.ppe_classes),
        }

    def warmup(self):
        """
        La première inférence est lente (initialisation de PyTorch, fusion des
        couches). On la fait avant de lancer la lecture pour que le flux
        temps réel ne perde pas ses premières secondes.
        """
        import numpy as np

        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.ppe_model.predict(blank, imgsz=self.imgsz, device=self.device, verbose=False)
        if self.phone_model is not None:
            self.phone_model.predict(blank[: self.phone_imgsz, : self.phone_imgsz], imgsz=self.phone_imgsz,
                                     device=self.device, verbose=False)

    def reset(self):
        """Oublie les identifiants de suivi (nouvelle source vidéo)."""
        predictor = getattr(self.ppe_model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def detect(self, frame) -> list[dict]:
        detections = self._detect_tracked(frame)
        if self.phone_model is not None:
            persons = [d for d in detections if d["class_name"] == "Person"]
            detections.extend(self._detect_phones(frame, persons))
        return detections

    def _detect_tracked(self, frame) -> list[dict]:
        result = self.ppe_model.track(
            source=frame,
            conf=self.track_confidence,
            imgsz=self.imgsz,
            device=self.device,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        detections = []
        for track_id, class_id, confidence, xyxy in zip(
            ids, boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            class_name = self.ppe_model.names[class_id]
            if confidence < self.confidence and not (class_name == "Person" and track_id is not None):
                continue
            detections.append({
                "track_id": track_id,  # ID temporaire, non biométrique
                "class_name": class_name,
                "confidence": round(float(confidence), 3),
                "bbox": tuple(int(v) for v in xyxy),
                "source": "best.pt",
            })
        return detections

    def _detect_phones(self, frame, persons: list[dict]) -> list[dict]:
        height, width = frame.shape[:2]
        candidates = sorted(
            (p for p in persons if p["bbox"][3] - p["bbox"][1] >= self.phone_min_person_height_px),
            key=lambda p: (p["bbox"][3] - p["bbox"][1]) * (p["bbox"][2] - p["bbox"][0]),
            reverse=True,
        )[: self.phone_max_persons]
        crops, offsets = [], []
        for person in candidates:
            x1, y1, x2, y2 = person["bbox"]
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.05)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
            if cx2 - cx1 < 16 or cy2 - cy1 < 16:
                continue
            crops.append(frame[cy1:cy2, cx1:cx2])
            offsets.append((cx1, cy1))
        if not crops:
            return []

        results = self.phone_model.predict(
            source=crops,
            conf=self.phone_confidence,
            imgsz=self.phone_imgsz,
            classes=self.phone_class_ids,
            device=self.device,
            verbose=False,
        )
        phones = []
        # Les recadrages de personnes voisines se recouvrent : un même
        # téléphone peut être détecté plusieurs fois.
        for (ox, oy), result in zip(offsets, results):
            for confidence, xyxy in zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
                x1, y1, x2, y2 = xyxy
                phones.append({
                    "track_id": None,
                    "class_name": "Phone",
                    "confidence": round(float(confidence), 3),
                    "bbox": (int(x1) + ox, int(y1) + oy, int(x2) + ox, int(y2) + oy),
                    "source": "coco",
                })
        kept = []
        for phone in sorted(phones, key=lambda p: p["confidence"], reverse=True):
            if all(_iou(phone["bbox"], other["bbox"]) < 0.4 for other in kept):
                kept.append(phone)
        return kept


def _iou(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
