"""
app/tracking/box_smoother.py

Lisse les boîtes de détection dans le temps, par track_id, pour réduire les
tremblements/sauts visuels d'une frame à l'autre (dus au fait que YOLO
redétecte chaque frame indépendamment, sans aucune notion de continuité).

Principe : moyenne mobile exponentielle (EMA) sur la position de chaque
boîte, calculée séparément pour chaque objet suivi (identifié par son
track_id + sa classe).
"""


class BoxSmoother:
    """
    Utilisation :
        smoother = BoxSmoother(alpha=0.5, max_missed_frames=15)
        tracked_objects = tracker.track(frame, persist=True)
        tracked_objects = smoother.smooth(tracked_objects, frame_idx)
        # les bbox de tracked_objects sont maintenant lissées
    """

    def __init__(self, alpha: float = 0.5, max_missed_frames: int = 15):
        """
        Args:
            alpha: poids donné à la NOUVELLE détection dans la moyenne
                (entre 0 et 1). Plus alpha est bas, plus le lissage est fort
                (boîtes très stables mais qui réagissent plus lentement aux
                vrais mouvements). 0.5 est un bon compromis pour commencer.
            max_missed_frames: nombre de frames sans nouvelle détection
                avant d'oublier l'historique d'un track_id (évite de garder
                en mémoire des personnes qui ont quitté le champ depuis
                longtemps).
        """
        self.alpha = alpha
        self.max_missed_frames = max_missed_frames
        # Clé : (class_name, track_id) -> {"bbox": (x1,y1,x2,y2), "last_frame": int}
        self._history: dict[tuple, dict] = {}

    def smooth(self, tracked_objects: list[dict], frame_idx: int) -> list[dict]:
        """
        Remplace la bbox de chaque objet par sa version lissée.

        Args:
            tracked_objects: sortie de ByteTrackTracker.track() (doit contenir track_id).
            frame_idx: numéro de la frame courante (pour gérer l'expiration).

        Returns:
            La même liste, avec les bbox remplacées par leur version lissée.
        """
        smoothed = []

        for obj in tracked_objects:
            track_id = obj.get("track_id")

            if track_id is None:
                # Pas d'ID de suivi (rare) -> on ne peut pas lisser dans le temps, on garde tel quel
                smoothed.append(obj)
                continue

            key = (obj["class_name"], track_id)
            new_bbox = obj["bbox"]

            if key in self._history:
                old_bbox = self._history[key]["bbox"]
                blended_bbox = tuple(
                    int(self.alpha * new_val + (1 - self.alpha) * old_val)
                    for new_val, old_val in zip(new_bbox, old_bbox)
                )
            else:
                blended_bbox = new_bbox

            self._history[key] = {"bbox": blended_bbox, "last_frame": frame_idx}

            obj_smoothed = dict(obj)
            obj_smoothed["bbox"] = blended_bbox
            smoothed.append(obj_smoothed)

        self._cleanup_old_tracks(frame_idx)
        return smoothed

    def _cleanup_old_tracks(self, current_frame: int):
        """Oublie les track_id qui n'ont pas été vus depuis trop longtemps."""
        expired_keys = [
            key for key, data in self._history.items()
            if current_frame - data["last_frame"] > self.max_missed_frames
        ]
        for key in expired_keys:
            del self._history[key]
