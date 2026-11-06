"""
app/epi/behavior_tracker.py

Décision TEMPORELLE des comportements dangereux, par personne suivie
(track_id ByteTrack) : absence de casque, de gilet, de harnais (si exigé) et
usage du téléphone en zone active.

Pourquoi ? Une image isolée ne suffit pas : un casque peut être masqué, la tête
hors champ, la personne trop petite, ou le modèle peut se tromper sur une
image. Ce module accumule les observations sur une fenêtre glissante (en
secondes de vidéo, indépendante du nombre d'images réellement analysées) et
ne confirme une violation que si elle est stable.

États par équipement :
    "present"      EPI positivement détecté de façon majoritaire
    "absent"       absence confirmée (violation)
    "a_verifier"   indices d'absence non confirmés (confiance faible ou instable)
    "indetermine"  observations insuffisantes
    "non_visible"  personne trop petite / zone du corps hors champ
    "non_requis"   harnais non exigé pour cette caméra

États téléphone : "detecte" (confirmé), "a_verifier", "non", "hors_zone".

Statut global d'une personne :
    "violation"    au moins une violation confirmée
    "a_verifier"   au moins un indice non confirmé
    "conforme"     tous les EPI requis positivement confirmés
    "indetermine"  preuves insuffisantes
"""

from collections import deque

ITEMS = {
    "casque": ("Hardhat", "NO-Hardhat"),
    "gilet": ("Safety Vest", "NO-Safety Vest"),
    "harnais": ("Harness", "NO-Harness"),
}
ITEM_LABELS = {"casque": "casque", "gilet": "gilet", "harnais": "harnais", "telephone": "téléphone"}


def point_in_polygon(x, y, polygon) -> bool:
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


class BehaviorTracker:
    def __init__(self, rules: dict, harness_required: bool = False, phone_active_area="whole_frame"):
        temporal = rules["temporal"]
        visibility = rules["visibility"]
        self.thresholds = rules["violation_confidence_thresholds"]
        self.window_s = temporal["window_s"]
        self.min_observations = temporal["min_observations"]
        self.min_duration_s = temporal["min_duration_s"]
        self.confirm_ratio = temporal["confirm_ratio"]
        self.release_ratio = temporal["release_ratio"]
        self.present_ratio = temporal["present_ratio"]
        self.phone_confirm_ratio = temporal["phone_confirm_ratio"]
        self.phone_min_hits = temporal["phone_min_hits"]
        self.track_expiry_s = temporal["track_expiry_s"]
        self.min_height_ratio = visibility["min_person_height_ratio"]
        self.min_height_px = visibility["min_person_height_px"]
        self.edge_margin_ratio = visibility["edge_margin_ratio"]
        self.harness_required = harness_required
        # "whole_frame" ou polygone en coordonnées relatives [[x, y], ...]
        self.phone_active_area = phone_active_area
        self._tracks: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Observation d'une image
    # ------------------------------------------------------------------
    def _observe_item(self, evidence: dict, item: str, visible: bool) -> str:
        if not visible:
            return "non_visible"
        positive_class, negative_class = ITEMS[item]
        present = evidence.get(positive_class, 0.0)
        absent = evidence.get(negative_class, 0.0)
        if absent >= self.thresholds[negative_class] and absent > present:
            return "absent"
        if present > 0:
            return "present"
        if absent > 0:
            return "absent_faible"
        return "non_vu"

    def _visibility(self, bbox, frame_shape):
        height, width = frame_shape
        x1, y1, x2, y2 = bbox
        person_height = y2 - y1
        margin = self.edge_margin_ratio * height
        reasons = []
        if person_height < max(self.min_height_px, self.min_height_ratio * height):
            reasons.append("personne trop petite / éloignée")
        head_visible = not reasons and y1 > margin
        if not reasons and not head_visible:
            reasons.append("tête hors champ")
        torso_visible = not reasons or reasons == ["tête hors champ"]
        return head_visible, torso_visible, reasons

    def _in_active_area(self, bbox, frame_shape) -> bool:
        if self.phone_active_area in (None, "whole_frame"):
            return True
        height, width = frame_shape
        x1, y1, x2, y2 = bbox
        return point_in_polygon((x1 + x2) / 2 / width, y2 / height, self.phone_active_area)

    # ------------------------------------------------------------------
    # Décision temporelle
    # ------------------------------------------------------------------
    def _decide_item(self, track: dict, item: str, now: float) -> str:
        timed = [(t, obs[item]) for t, obs in track["history"]
                 if now - t <= self.window_s and obs[item] != "non_visible"]
        seen = [o for _, o in timed]
        previous = track["states"].get(item)
        if not seen:
            return "non_visible"
        count = len(seen)
        # une violation n'est confirmée qu'après une durée minimale d'observation
        enough = count >= self.min_observations and now - timed[0][0] >= self.min_duration_s
        absent = seen.count("absent")
        weak = seen.count("absent_faible")
        present = seen.count("present")

        if previous == "absent" and absent / count >= self.release_ratio:
            return "absent"  # hystérésis : évite le clignotement d'une violation établie
        if enough and absent / count >= self.confirm_ratio:
            return "absent"
        if present / count >= self.present_ratio:
            return "present"
        if absent + weak > 0:
            return "a_verifier"
        return "indetermine"

    def _decide_phone(self, track: dict, now: float) -> tuple[str, float]:
        timed = [(t, obs["telephone"]) for t, obs in track["history"]
                 if now - t <= self.window_s and obs["telephone"][0] != "non_visible"]
        seen = [o for _, o in timed]
        if not seen:
            return "non", 0.0
        enough = now - timed[0][0] >= self.min_duration_s
        hits = sum(1 for state, _ in seen if state == "detecte")
        weak = sum(1 for state, _ in seen if state == "faible")
        best = max(conf for _, conf in seen)
        previous = track["states"].get("telephone")
        if previous == "detecte" and hits / len(seen) >= self.release_ratio:
            return "detecte", best
        if enough and hits >= self.phone_min_hits and hits / len(seen) >= self.phone_confirm_ratio:
            return "detecte", best
        if hits + weak > 0:
            return "a_verifier", best
        return "non", best

    def update(self, statuses: list[dict], now: float, frame_shape: tuple[int, int]) -> list[dict]:
        """
        Args:
            statuses: sortie de EPIComplianceChecker.check() pour l'image.
            now: temps de la vidéo en secondes (pas l'horloge murale).
            frame_shape: (hauteur, largeur) de l'image analysée.
        """
        persons = []
        for status in statuses:
            track_id = status.get("track_id")
            bbox = status["bbox"]
            evidence = status.get("evidence", {})
            head_visible, torso_visible, reasons = self._visibility(bbox, frame_shape)

            if track_id is None:
                persons.append(self._person(status, "indetermine", {}, ["suivi non établi"] + reasons, [], 0, {}))
                continue

            track = self._tracks.setdefault(track_id, {"history": deque(), "states": {}, "first_seen": now})
            track["last_seen"] = now
            phone_confidence = evidence.get("Phone", 0.0)
            if not torso_visible:
                phone_obs = ("non_visible", 0.0)
            elif phone_confidence >= self.thresholds["Phone"]:
                phone_obs = ("detecte", phone_confidence)
            elif phone_confidence > 0:
                phone_obs = ("faible", phone_confidence)
            else:
                phone_obs = ("non", 0.0)
            track["history"].append((now, {
                "casque": self._observe_item(evidence, "casque", head_visible),
                "gilet": self._observe_item(evidence, "gilet", torso_visible),
                "harnais": self._observe_item(evidence, "harnais", torso_visible),
                "telephone": phone_obs,
            }))
            while track["history"] and now - track["history"][0][0] > self.window_s:
                track["history"].popleft()

            states = {item: self._decide_item(track, item, now) for item in ITEMS}
            if not self.harness_required:
                states["harnais"] = "non_requis"
            phone_state, phone_conf = self._decide_phone(track, now)
            track["states"] = dict(states, telephone=phone_state)
            if phone_state != "non" and not self._in_active_area(bbox, frame_shape):
                phone_state = "hors_zone"
            states["telephone"] = phone_state

            required = ["casque", "gilet"] + (["harnais"] if self.harness_required else [])
            violations = [item for item in required if states[item] == "absent"]
            if phone_state == "detecte":
                violations.append("telephone")
            if violations:
                label = "violation"
            elif any(states[item] == "a_verifier" for item in required) or phone_state == "a_verifier":
                label = "a_verifier"
            elif all(states[item] == "present" for item in required):
                label = "conforme"
            else:
                label = "indetermine"
                if not reasons and len(track["history"]) < self.min_observations:
                    reasons = ["observation en cours"]
                elif not reasons:
                    reasons = ["EPI non identifiés avec certitude"]

            confidences = {
                "casque": evidence.get("NO-Hardhat") or evidence.get("Hardhat"),
                "gilet": evidence.get("NO-Safety Vest") or evidence.get("Safety Vest"),
                "harnais": evidence.get("NO-Harness") or evidence.get("Harness"),
                "telephone": phone_conf or None,
            }
            persons.append(self._person(status, label, states, reasons, violations, len(track["history"]), confidences))

        self._expire(now)
        return persons

    @staticmethod
    def _person(status, label, states, reasons, violations, observations, confidences):
        return {
            "track_id": status.get("track_id"),
            "bbox": status["bbox"],
            "confidence": status.get("confidence"),
            "status": label,
            "items": states,
            "violations": violations,
            "reasons": reasons,
            "observations": observations,
            "confidences": {k: round(v, 2) for k, v in confidences.items() if v},
        }

    def _expire(self, now: float):
        for track_id in [tid for tid, track in self._tracks.items() if now - track["last_seen"] > self.track_expiry_s]:
            del self._tracks[track_id]

    @staticmethod
    def describe_violations(person: dict) -> str:
        parts = []
        for item in person["violations"]:
            parts.append("usage du téléphone" if item == "telephone" else f"absence de {ITEM_LABELS[item]}")
        return ", ".join(parts)
