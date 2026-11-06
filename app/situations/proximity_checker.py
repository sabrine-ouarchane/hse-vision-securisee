"""
app/situations/proximity_checker.py

Situations dangereuses homme-machine (objectif 2.2 du cahier des charges) :
proximité entre une personne suivie et un équipement mobile (convoyeur,
roue-pelle, engin) détecté dans l'image.

Trois étages :
1. MachineTracker : stabilise les détections "Machine" (filtre des boîtes
   aberrantes, fusion des doublons, confirmation sur plusieurs images et
   maintien bref quand l'engin est momentanément masqué).
2. Mesure de distance :
   - SANS calibration : distance RELATIVE, exprimée en hauteurs de la
     personne (≈ 1,7 m, mais sans garantie). On combine l'écart horizontal
     entre les boîtes et l'écart "au sol" entre les pieds de la personne et
     la base de l'engin, pour ne pas confondre une personne au premier plan
     (devant l'engin dans l'image) avec une personne réellement à côté.
     Aucune valeur n'est présentée en mètres.
   - AVEC calibration (homographie image -> sol définie dans
     config/camera_profiles.json) : distance en mètres au sol.
3. Confirmation temporelle et prise en compte de l'état machine :
   - machine en marche confirmée       -> alerte de proximité dangereuse
   - machine à l'arrêt confirmée        -> pas d'alerte "machine en marche"
   - état inconnu                       -> incertitude affichée, règle de
     prudence configurable ("prudent" : traitée comme en marche, cf.
     S-HSE-22 ; "surveillance" : événement à vérifier)
"""

import math
from collections import deque

import cv2
import numpy as np


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(bbox_a, bbox_b) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_area = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union_area = _bbox_area(bbox_a) + _bbox_area(bbox_b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class MachineTracker:
    """Suivi léger et robuste des engins détectés (classe 'Machine')."""

    def __init__(self, cfg: dict, alpha: float = 0.4):
        self.max_area_ratio = cfg["max_machine_area_ratio"]
        self.min_confidence = cfg["machine_min_confidence"]
        self.confirm_hits = cfg["machine_confirm_hits"]
        self.hold_s = cfg["machine_hold_s"]
        self.alpha = alpha
        self._machines: list[dict] = []
        self._next_id = 1

    def update(self, detections: list[dict], now: float, frame_shape) -> list[dict]:
        frame_area = frame_shape[0] * frame_shape[1]
        candidates = sorted(
            (d for d in detections
             if d["class_name"] == "Machine"
             and d["confidence"] >= self.min_confidence
             and _bbox_area(d["bbox"]) <= self.max_area_ratio * frame_area),
            key=lambda d: d["confidence"], reverse=True,
        )
        matched = set()
        for detection in candidates:
            best, best_iou = None, 0.2
            for machine in self._machines:
                if machine["id"] in matched:
                    continue
                overlap = _iou(detection["bbox"], machine["bbox"])
                if overlap > best_iou:
                    best, best_iou = machine, overlap
            if best is None:
                if any(_iou(detection["bbox"], m["bbox"]) > 0.3 for m in self._machines if m["id"] in matched):
                    continue  # doublon d'un engin déjà mis à jour sur cette image
                best = {"id": self._next_id, "bbox": detection["bbox"], "hits": 0, "first_seen": now}
                self._next_id += 1
                self._machines.append(best)
            else:
                best["bbox"] = tuple(
                    int(self.alpha * new + (1 - self.alpha) * old) for new, old in zip(detection["bbox"], best["bbox"])
                )
            best["hits"] += 1
            best["last_seen"] = now
            best["confidence"] = detection["confidence"]
            matched.add(best["id"])

        self._machines = [m for m in self._machines if now - m["last_seen"] <= self.hold_s]
        return [
            dict(m, visible=m["id"] in matched)
            for m in self._machines
            if m["hits"] >= self.confirm_hits
        ]


class ProximityChecker:
    def __init__(self, cfg: dict, calibration: dict | None = None):
        self.danger_rel = cfg["danger_person_heights"]
        self.warning_rel = cfg["warning_person_heights"]
        self.danger_m = cfg["danger_m"]
        self.warning_m = cfg["warning_m"]
        self.ground_band_ratio = cfg["machine_ground_band_ratio"]
        # Un écart vertical au sol dans l'image couvre plus de distance réelle
        # qu'un écart horizontal (raccourci de perspective d'une caméra
        # oblique). Coefficient heuristique, réglable par caméra.
        self.depth_weight = cfg.get("ground_depth_weight", 1.0)
        self.confirm_window_s = cfg["confirm_window_s"]
        self.confirm_ratio = cfg["confirm_ratio"]
        self.min_observations = cfg["min_observations"]
        self.min_duration_s = cfg["min_duration_s"]
        self.release_s = cfg["release_s"]
        self.calibration = calibration
        self._homography_cache = {}
        self._tracks: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Géométrie
    # ------------------------------------------------------------------
    def _homography(self, frame_shape):
        if not self.calibration:
            return None
        key = tuple(frame_shape)
        if key not in self._homography_cache:
            height, width = frame_shape
            image_points = np.float32([[x * width, y * height] for x, y in self.calibration["image_points"]])
            ground_points = np.float32(self.calibration["ground_points_m"])
            self._homography_cache[key] = cv2.getPerspectiveTransform(image_points, ground_points)
        return self._homography_cache[key]

    @staticmethod
    def is_inside_machine(person_bbox, machine_bbox) -> bool:
        """
        Conducteur en cabine (ou personne masquée derrière l'engin) : personne
        incluse dans la boîte de l'engin et, soit ses pieds ne touchent pas la
        base de l'engin, soit seul son buste est visible (boîte trapue, le
        bas du corps étant caché par la carrosserie). Un piéton visible en
        entier devant l'engin reste évalué.
        """
        px1, py1, px2, py2 = person_bbox
        mx1, my1, mx2, my2 = machine_bbox
        width, height = max(1, px2 - px1), max(1, py2 - py1)
        inter = max(0, min(px2, mx2) - max(px1, mx1)) * max(0, min(py2, my2) - max(py1, my1))
        if inter / (width * height) < 0.85:
            return False
        feet_above_base = py2 < my2 - 0.08 * (my2 - my1)
        upper_body_only = height / width < 1.8
        return feet_above_base or upper_body_only

    def measure(self, person_bbox, machine_bbox, frame_shape) -> dict:
        px1, py1, px2, py2 = person_bbox
        mx1, my1, mx2, my2 = machine_bbox
        person_height = max(1, py2 - py1)
        foot_x, foot_y = (px1 + px2) / 2, py2

        # Point de la base de l'engin le plus proche des pieds de la personne
        band_top = my2 - self.ground_band_ratio * (my2 - my1)
        nearest_x = min(max(foot_x, mx1), mx2)
        nearest_y = min(max(foot_y, band_top), my2)
        dx = max(mx1 - px2, px1 - mx2, 0)
        dy = abs(foot_y - nearest_y)
        relative = math.hypot(dx, dy * self.depth_weight) / person_height

        distance_m = None
        homography = self._homography(frame_shape)
        if homography is not None:
            points = cv2.perspectiveTransform(
                np.float32([[[foot_x, foot_y]], [[nearest_x, nearest_y]]]), homography
            ).reshape(-1, 2)
            distance_m = float(np.linalg.norm(points[0] - points[1]))

        if distance_m is not None:
            level = "danger" if distance_m <= self.danger_m else "vigilance" if distance_m <= self.warning_m else "aucun"
        else:
            level = "danger" if relative <= self.danger_rel else "vigilance" if relative <= self.warning_rel else "aucun"
        return {
            "distance_rel": round(relative, 2),
            "distance_m": None if distance_m is None else round(distance_m, 1),
            "level": level,
            "person_point": (int(foot_x), int(foot_y)),
            "machine_point": (int(nearest_x), int(nearest_y)),
        }

    # ------------------------------------------------------------------
    # Décision
    # ------------------------------------------------------------------
    def check(self, persons: list[dict], machines: list[dict], now: float, frame_shape,
              machine_state: str, unknown_policy: str = "prudent") -> list[dict]:
        """
        Args:
            persons: personnes suivies (track_id + bbox).
            machines: sortie de MachineTracker.update().
            machine_state: "marche" | "arret" | "inconnu" (source déclarée, jamais devinée).
            unknown_policy: "prudent" ou "surveillance".
        """
        situations = []
        if not machines:
            self._expire(now)
            return situations

        for person in persons:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            measures = [
                (m, self.measure(person["bbox"], m["bbox"], frame_shape))
                for m in machines if not self.is_inside_machine(person["bbox"], m["bbox"])
            ]
            if not measures:
                continue
            machine, measure = min(
                measures, key=lambda pair: pair[1]["distance_m"] if pair[1]["distance_m"] is not None else pair[1]["distance_rel"]
            )

            track = self._tracks.setdefault(track_id, {"history": deque(), "confirmed": None, "last_danger": None})
            track["history"].append((now, measure["level"]))
            track["last_seen"] = now
            while track["history"] and now - track["history"][0][0] > self.confirm_window_s:
                track["history"].popleft()
            levels = [level for _, level in track["history"]]
            if measure["level"] == "danger":
                track["last_danger"] = now

            danger_ratio = levels.count("danger") / len(levels)
            enough = len(levels) >= self.min_observations and now - track["history"][0][0] >= self.min_duration_s
            if enough and danger_ratio >= self.confirm_ratio:
                track["confirmed"] = "danger"
            elif track["confirmed"] == "danger" and now - (track["last_danger"] or now) <= self.release_s:
                pass  # maintien : évite qu'une alerte clignote à chaque image
            elif measure["level"] != "aucun":
                track["confirmed"] = "vigilance"
            else:
                track["confirmed"] = None

            if track["confirmed"] is None:
                continue

            # Interprétation selon l'état machine déclaré
            if track["confirmed"] == "danger":
                if machine_state == "marche":
                    status, severity = "danger", "critique"
                elif machine_state == "arret":
                    status, severity = "machine_arretee", None
                elif unknown_policy == "prudent":
                    status, severity = "danger_etat_inconnu", "critique"
                else:
                    status, severity = "a_verifier", "avertissement"
            else:
                status, severity = "vigilance", None

            situations.append({
                "track_id": track_id,
                "bbox": person["bbox"],
                "machine_id": machine["id"],
                "machine_bbox": machine["bbox"],
                "machine_state": machine_state,
                "status": status,
                "severity": severity,
                "confidence": round(min(person.get("confidence") or 0.0, machine.get("confidence") or 0.0), 3),
                **measure,
            })

        self._expire(now)
        return situations

    def _expire(self, now: float, expiry_s: float = 4.0):
        for track_id in [tid for tid, t in self._tracks.items() if now - t.get("last_seen", now) > expiry_s]:
            del self._tracks[track_id]

    @staticmethod
    def describe_distance(situation: dict) -> str:
        if situation.get("distance_m") is not None:
            return f"≈ {situation['distance_m']:.1f} m (calibré)"
        return f"≈ {situation['distance_rel']:.1f} × hauteur de la personne (non calibré)"

    @staticmethod
    def format_danger_message(situation: dict) -> str:
        track_id = situation["track_id"] if situation["track_id"] is not None else "?"
        state = {
            "marche": "machine en marche",
            "arret": "machine à l'arrêt",
            "inconnu": "état machine non confirmé",
        }[situation["machine_state"]]
        return (f"Personne #{track_id} : proximité dangereuse avec l'engin #{situation['machine_id']} "
                f"({state}) — distance {ProximityChecker.describe_distance(situation)}")
