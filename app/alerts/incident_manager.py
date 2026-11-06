"""
app/alerts/incident_manager.py

Transforme des violations confirmées image par image en INCIDENTS :
une situation (ex. personne #4 sans casque) ouvre un seul incident qui reste
actif tant qu'elle persiste, puis se clôture après quelques secondes sans
observation. Une alerte est journalisée à l'ouverture uniquement — plus de
répétition toutes les N images pour la même situation.

Avec un journal d'audit (app/security/audit_log.py), l'ouverture et la
clôture (avec la durée) de chaque incident y sont aussi consignées.

Pendant un sabotage de caméra confirmé, l'image ne prouve plus rien sur les
personnes : leurs incidents ouverts sont gelés (« non actualisée depuis … »),
puis repris si la personne est revue, ou clôturés avec le motif
« interrompu par sabotage caméra ».

Si ByteTrack attribue un nouvel identifiant à la même personne (occlusion,
sortie/retour dans le champ), l'incident récent du même type situé au même
endroit est repris au lieu d'en ouvrir un nouveau.
"""

import hashlib
import threading
from datetime import datetime
from pathlib import Path

import cv2

KIND_LABELS = {
    "casque": "Absence de casque",
    "gilet": "Absence de gilet haute visibilité",
    "harnais": "Absence de harnais (travail en hauteur)",
    "telephone": "Usage du téléphone en zone active",
    "proximite": "Proximité dangereuse homme-machine",
    # Incidents CYBER : sabotage de la caméra (app/security/tamper_detector.py)
    "obstruction": "CYBER · Obstruction de l'objectif",
    "deplacement": "CYBER · Déplacement de la caméra",
    "image_degradee": "CYBER · Image dégradée",
    "flux_fige": "CYBER · Flux figé",
    "rejeu": "CYBER · Rejeu suspecté",
}
CYBER_KINDS = {"obstruction", "deplacement", "image_degradee", "flux_fige", "rejeu"}
# Type compatible avec le journal lu par le dashboard (epi / proximity / cyber)
ALERT_TYPES = {"casque": "epi", "gilet": "epi", "harnais": "epi", "telephone": "epi", "proximite": "proximity",
               **{kind: "cyber" for kind in CYBER_KINDS}}


def _iou(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IncidentManager:
    def __init__(self, alert_manager=None, snapshot_dir=None, close_after_s: float = 3.0,
                 merge_s: float = 4.0, source_name: str = "", audit=None, audit_context: dict | None = None):
        self.alert_manager = alert_manager
        self.audit = audit
        self.audit_context = audit_context or {}
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.close_after_s = close_after_s
        self.merge_s = merge_s
        self.source_name = source_name
        self.incidents: list[dict] = []
        self._open: dict[tuple, dict] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def update(self, active: list[dict], now: float, frame_idx: int, annotated_frame=None, scale: float = 1.0,
               integrity: dict | None = None) -> list[dict]:
        """
        Args:
            active: situations confirmées sur cette image, chacune avec
                kind, track_id, bbox, severity, message, confidence, et
                started_t optionnel (début réel, ex. sabotage confirmé après coup).
            now: temps vidéo (s).
            annotated_frame: image annotée utilisée pour la capture (optionnelle).
            scale: facteur entre les coordonnées des bbox et annotated_frame.
            integrity: état de l'image fourni par le contrôle de sabotage
                ({"compromised": bool, "suspected": bool}). Image compromise :
                les incidents de personnes ouverts sont GELÉS (durée arrêtée à
                la dernière image fiable) et aucun n'est ouvert ; anomalie en
                observation : aucun n'est clôturé.
        Returns:
            Les incidents nouvellement ouverts.
        """
        integrity = integrity or {}
        compromised = bool(integrity.get("compromised"))
        suspected = bool(integrity.get("suspected"))
        opened, closed = [], []
        with self._lock:
            if compromised:
                self._freeze_open()
            seen = set()
            for situation in active:
                if compromised and situation["kind"] not in CYBER_KINDS:
                    continue  # image non fiable : elle ne prouve rien sur les personnes
                key = (situation["kind"], situation["track_id"])
                seen.add(key)
                incident = self._open.get(key) or self._find_mergeable(situation, now)
                if incident is None:
                    incident = self._create(situation, now, frame_idx, annotated_frame, scale)
                    opened.append(incident)
                elif incident.get("frozen"):
                    self._resume(incident, now)
                self._open[key] = incident
                incident["last_t"] = now
                if not suspected:
                    incident["reliable_t"] = now
                    incident["reliable_wall"] = datetime.now().astimezone().strftime("%H:%M:%S")
                incident["bbox"] = situation["bbox"]
                incident["status"] = "active"
                incident["ended_t"] = None
                if situation["track_id"] not in incident["track_ids"]:
                    incident["track_ids"].append(situation["track_id"])
                if situation.get("confidence"):
                    incident["peak_confidence"] = max(incident["peak_confidence"] or 0, situation["confidence"])
                if situation["severity"] == "critique":
                    incident["severity"] = "critique"
                incident["message"] = situation["message"]

            for key, incident in list(self._open.items()):
                if key in seen:
                    continue
                reason = None
                if incident["kind"] not in CYBER_KINDS and (compromised or suspected):
                    continue  # pas de clôture tant que l'image n'est pas fiable
                if incident.get("frozen"):
                    # Sabotage terminé : on laisse à la personne le temps d'être revue
                    if incident.get("released_t") is None:
                        incident["released_t"] = now
                    if now - incident["released_t"] <= self.close_after_s:
                        continue
                    reason = "interrompu par sabotage caméra"
                elif now - incident["last_t"] <= self.close_after_s:
                    continue
                incident["status"] = "terminee"
                incident["ended_t"] = incident["last_t"]
                del self._open[key]
                if all(other is not incident for other in self._open.values()):
                    if reason:
                        incident["close_reason"] = reason
                    closed.append(incident)
            for incident in self.incidents:
                incident["duration_s"] = self._duration(incident)
            for incident in closed:
                details = {"duration_s": incident["duration_s"]}
                if incident.get("close_reason"):
                    details["motif"] = incident["close_reason"]
                self._audit("incident_cloture", incident, **details)
        return opened

    @staticmethod
    def _duration(incident) -> float:
        """Durée observée, hors périodes où l'image n'était pas fiable."""
        end = incident["ended_t"] if incident["ended_t"] is not None else incident["last_t"]
        return round(end - incident["started_t"] - incident.get("paused_s", 0.0), 1)

    def _freeze_open(self):
        """Sabotage confirmé : les incidents de personnes ouverts ne sont plus actualisés."""
        frozen = []
        for incident in self._open.values():
            if incident["kind"] in CYBER_KINDS or incident.get("frozen") or any(f is incident for f in frozen):
                continue
            frozen.append(incident)
            incident["frozen"] = True
            incident["released_t"] = None
            incident["last_t"] = min(incident["last_t"], incident.get("reliable_t", incident["last_t"]))
            incident["frozen_since"] = incident.get("reliable_wall") or incident["wall_time"][11:19]
            incident["message_before_freeze"] = incident["message"]
            incident["message"] += (f" — non actualisée depuis {incident['frozen_since']}"
                                    f" (t = {incident['last_t']:.1f} s, image non fiable)")
            self._audit("incident_gele", incident, dernier_t_fiable=round(incident["last_t"], 2),
                        non_actualise_depuis=incident["frozen_since"])

    def _resume(self, incident, now):
        """Personne revue après le sabotage : la période gelée est exclue de la durée."""
        gap = max(0.0, now - incident["last_t"])
        incident["paused_s"] = incident.get("paused_s", 0.0) + gap
        incident["frozen"] = False
        incident["message"] = incident.pop("message_before_freeze", incident["message"])
        incident.pop("frozen_since", None)
        self._audit("incident_reprise", incident, interruption_s=round(gap, 1))

    def _find_mergeable(self, situation, now):
        for incident in reversed(self.incidents):
            # Incident gelé pendant un sabotage : la personne revue au même endroit
            # (souvent avec un nouvel identifiant de suivi) le reprend.
            if (incident["kind"] == situation["kind"] and incident.get("frozen")
                    and (situation["track_id"] in incident["track_ids"]
                         or _iou(incident["bbox"], situation["bbox"]) >= 0.3)):
                return incident
            # Même personne, même type, interruption brève (ex. casque masqué
            # quelques instants) : on rouvre l'incident au lieu d'en créer un.
            if (incident["kind"] == situation["kind"]
                    and situation["track_id"] in incident["track_ids"]
                    and incident["status"] == "terminee"
                    and now - incident["last_t"] <= self.close_after_s + self.merge_s):
                return incident
            if (incident["kind"] == situation["kind"]
                    and incident["last_t"] < now  # pas déjà mis à jour par une autre personne sur cette image
                    and now - incident["last_t"] <= self.merge_s
                    and situation["track_id"] not in incident["track_ids"]
                    and _iou(incident["bbox"], situation["bbox"]) >= 0.3):
                return incident
        return None

    def _create(self, situation, now, frame_idx, annotated_frame, scale):
        incident_id = self._next_id
        self._next_id += 1
        wall_time = datetime.now().astimezone().isoformat(timespec="seconds")
        incident = {
            "id": incident_id,
            "kind": situation["kind"],
            "label": KIND_LABELS[situation["kind"]],
            "severity": situation["severity"],
            "message": situation["message"],
            "track_ids": [situation["track_id"]],
            "bbox": situation["bbox"],
            "started_t": situation.get("started_t", now),
            "last_t": now,
            "ended_t": None,
            "duration_s": 0.0,
            "frame_idx": frame_idx,
            "wall_time": wall_time,
            "peak_confidence": situation.get("confidence"),
            "status": "active",
            "snapshot": None,
            "source": self.source_name,
        }
        if annotated_frame is not None and self.snapshot_dir is not None:
            incident["snapshot"] = self._save_snapshot(incident_id, annotated_frame, situation["bbox"], scale)
        self.incidents.append(incident)
        csv_event_id = None
        if self.alert_manager is not None:
            csv_event_id = self.alert_manager.send_alert(
                track_id=situation["track_id"],
                message=f"[{self.source_name}] {situation['message']}" if self.source_name else situation["message"],
                frame_idx=frame_idx,
                alert_type=ALERT_TYPES[situation["kind"]],
                confidence=situation.get("confidence"),
                severity=situation["severity"],
                source="vision-temps-reel",
            ) or None
        # Références pour le contrôle de cohérence : ligne CSV et empreinte de la capture
        snapshot_sha256 = None
        if incident["snapshot"]:
            snapshot_sha256 = hashlib.sha256((self.snapshot_dir / incident["snapshot"]).read_bytes()).hexdigest()
        self._audit("incident_ouvert", incident, message=incident["message"], severity=incident["severity"],
                    wall_time=wall_time, snapshot=incident["snapshot"], snapshot_sha256=snapshot_sha256,
                    csv_event_id=csv_event_id)
        return incident

    def close_all(self, reason: str = "fin de session"):
        """Clôture les incidents encore ouverts (fin de vidéo, arrêt de la session)."""
        with self._lock:
            closing = []
            for incident in self._open.values():
                if all(other is not incident for other in closing):
                    closing.append(incident)
            self._open.clear()
            for incident in closing:
                incident["status"] = "terminee"
                incident["ended_t"] = incident["last_t"]
                incident["duration_s"] = self._duration(incident)
                motif = "interrompu par sabotage caméra" if incident.get("frozen") else reason
                incident["close_reason"] = motif
                self._audit("incident_cloture", incident, duration_s=incident["duration_s"], motif=motif)

    def _audit(self, event, incident, **details):
        if self.audit is not None:
            self.audit.append(event, {
                **self.audit_context, "source": self.source_name, "incident_id": incident["id"],
                "type": incident["kind"], "categorie": ALERT_TYPES[incident["kind"]],
                "debut_t": round(incident["started_t"], 2), **details,
            })

    def _save_snapshot(self, incident_id, frame, bbox, scale):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(v * scale) for v in bbox)
        pad_x, pad_y = int((x2 - x1) * 0.6) + 20, int((y2 - y1) * 0.25) + 20
        crop = frame[max(0, y1 - pad_y):min(height, y2 + pad_y), max(0, x1 - pad_x):min(width, x2 + pad_x)]
        if crop.size == 0:
            return None
        path = self.snapshot_dir / f"incident_{incident_id:04d}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path.name

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(incident, track_ids=list(incident["track_ids"])) for incident in self.incidents]
