"""
app/tracking/bytetrack_tracker.py

Suivi temporel des personnes détectées, via ByteTrack (natif dans ultralytics).

IMPORTANT (conformité cahier des charges) :
L'identifiant attribué (track_id) est un ID TEMPORAIRE de suivi vidéo,
valable uniquement le temps de la séquence traitée. Ce n'est PAS un
identifiant biométrique : il ne permet pas de reconnaître une personne
d'une vidéo à l'autre, et n'est associé à aucune donnée d'identité.
Il sert uniquement à :
    - éviter de compter/alerter plusieurs fois la même personne
    - calculer une durée d'exposition à une zone à risque
"""

from pathlib import Path
from ultralytics import YOLO


class ByteTrackTracker:
    """
    Utilise model.track() (ByteTrack natif d'ultralytics) pour suivre
    chaque personne détectée avec un ID temporaire stable dans le temps.

    Utilisation :
        tracker = ByteTrackTracker("models/best.pt")
        tracked_objects = tracker.track(frame)
        # [{"track_id": 3, "class_name": "Person", "confidence": 0.88,
        #   "bbox": (x1, y1, x2, y2)}, ...]
    """

    def __init__(self, model_path: str = "models/best.pt", confidence_threshold: float = 0.5, device: str = "cpu", imgsz: int = 960):
        """
        Args:
            model_path: chemin vers le fichier .pt du modèle fine-tuné.
            confidence_threshold: seuil minimal de confiance pour garder une détection.
            device: "cpu" ou "cuda" (ou "0" pour le premier GPU) selon la machine.
            imgsz: résolution d'inférence. Le défaut d'ultralytics est 640, ce qui
                fait rater les personnes/objets petits ou éloignés dans l'image.
                Augmenter à 960 ou 1280 améliore nettement leur détection, au prix
                d'un traitement un peu plus lent.
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
        self.imgsz = imgsz
        self.class_names = self.model.names

        print(f"[ByteTrackTracker] Modèle chargé : {model_path} (imgsz={imgsz})")

    def track(self, frame, persist: bool = True) -> list[dict]:
        """
        Suit les objets sur une frame (à appeler séquentiellement sur les
        frames successives d'une même vidéo pour que le tracking soit cohérent).

        Args:
            frame: image OpenCV (BGR numpy array).
            persist: True pour garder la mémoire des ID entre les appels
                     successifs (indispensable pour un flux vidéo continu).

        Returns:
            Liste de dictionnaires, un par objet suivi.
        """
        results = self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            imgsz=self.imgsz,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        tracked_objects = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                # boxes.id peut être None si aucun objet n'a encore reçu d'ID
                # (ex: toute première frame, ou aucune détection)
                continue

            for box in boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                tracked_objects.append({
                    "track_id": track_id,  # ID temporaire, non biométrique
                    "class_name": self.class_names[class_id],
                    "class_id": class_id,
                    "confidence": round(confidence, 3),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        return tracked_objects

    def reset(self):
        """
        Réinitialise l'état de tracking (à appeler entre deux vidéos
        différentes pour ne pas mélanger les ID d'une séquence à l'autre).
        """
        # Recharger le modèle réinitialise proprement l'état interne du tracker
        self.model = YOLO(self.model.ckpt_path) if hasattr(self.model, "ckpt_path") else self.model
        print("[ByteTrackTracker] État de tracking réinitialisé.")
