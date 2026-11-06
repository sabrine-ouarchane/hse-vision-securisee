"""
app/alerts/alert_manager.py

Gère le déclenchement des alertes HSE lorsqu'une violation est détectée
(absence d'EPI, usage du téléphone en zone active, etc.).

Pour le POC, les alertes sont écrites en console + dans un fichier log CSV.
Ce module est conçu pour être facilement étendu plus tard vers un vrai canal
(email, SMS, dashboard temps réel, base de données) sans changer l'appel
depuis le reste de l'application — seul le contenu de send_alert() changerait.

IMPORTANT : pour éviter le spam d'alertes (une personne sans casque génère
une violation à CHAQUE frame, soit ~30-60 alertes/seconde), ce module applique
un cooldown par track_id : une même personne ne redéclenche pas d'alerte tant
qu'un délai minimal (en nombre de frames) ne s'est pas écoulé.
"""

import csv
import secrets
from datetime import datetime
from pathlib import Path


class AlertManager:
    """
    Utilisation :
        alert_manager = AlertManager(log_path="data/outputs/alerts_log.csv")
        alert_manager.send_alert(track_id=3, message="Personne #3 : violation (casque)", frame_idx=120)
    """

    def __init__(self, log_path: str = "data/outputs/alerts_log.csv", cooldown_frames: int = 90):
        """
        Args:
            log_path: chemin du fichier CSV où toutes les alertes sont journalisées.
            cooldown_frames: nombre de frames minimum entre deux alertes pour
                             la même personne (évite le spam). Avec une vidéo
                             à ~30 FPS, 90 frames ≈ 3 secondes.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.cooldown_frames = cooldown_frames

        # Mémorise la dernière frame où une alerte a été envoyée, par (track_id, type d'alerte)
        # -> une alerte "violation EPI" et une alerte "proximité machine" pour
        # la même personne ne se bloquent plus mutuellement
        self._last_alert_frame: dict[tuple, int] = {}

        # Initialise le fichier CSV avec un en-tête si besoin
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "frame_idx", "track_id", "alert_type", "message",
                    "confidence", "severity", "source", "status", "event_id",
                ])

    def _is_in_cooldown(self, track_id: int, alert_type: str, frame_idx: int) -> bool:
        last_frame = self._last_alert_frame.get((track_id, alert_type))
        if last_frame is None:
            return False
        return (frame_idx - last_frame) < self.cooldown_frames

    def send_alert(
        self,
        track_id: int,
        message: str,
        frame_idx: int,
        alert_type: str = "epi",
        confidence: float | None = None,
        severity: str | None = None,
        source: str = "vision",
        status: str = "nouvelle",
    ) -> bool:
        """
        Déclenche une alerte si le cooldown est écoulé pour ce (track_id, alert_type).

        Args:
            alert_type: catégorie de l'alerte (ex. "epi", "proximity"). Deux
                types différents pour la même personne ont des cooldowns
                indépendants.

        Returns:
            L'identifiant de la ligne écrite (event_id, valeur vraie) si
            l'alerte a été envoyée, False si elle a été supprimée à cause du
            cooldown. L'event_id permet au journal d'audit de retrouver la
            ligne (contrôle de cohérence). Il est unique : deux alertes de la
            même seconde, même image, même personne et même type (ex. casque
            et gilet) reçoivent des suffixes aléatoires différents.
        """
        if track_id is not None and self._is_in_cooldown(track_id, alert_type, frame_idx):
            return False

        timestamp = datetime.now().isoformat(timespec="seconds")

        # 1) Affichage console (à remplacer/compléter plus tard par email/dashboard)
        print(f"[ALERTE HSE] {timestamp} | frame {frame_idx} | [{alert_type}] {message}")

        # 2) Journalisation dans le fichier CSV
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            event_id = f"{timestamp}|{frame_idx}|{track_id or ''}|{alert_type}|{secrets.token_hex(6)}"
            writer.writerow([
                timestamp,
                frame_idx,
                track_id,
                alert_type,
                message,
                "" if confidence is None else round(confidence, 3),
                severity or ("critique" if alert_type in {"epi", "proximity", "zone"} else "avertissement"),
                source,
                status,
                event_id,
            ])

        if track_id is not None:
            self._last_alert_frame[(track_id, alert_type)] = frame_idx

        return event_id
