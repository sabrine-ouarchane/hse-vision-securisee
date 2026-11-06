"""
ZoneChecker (v1)

Détermine le niveau de risque (ROUGE/ORANGE/VERT) d'une zone à chaque frame,
en fusionnant :
  - présence humaine dans la zone (polygone)
  - violations EPI (brique 1, EPIComplianceChecker)
  - proximité machine dangereuse (brique 2, ProximityChecker)
  - lectures gaz simulées (GasSensorSimulator)
  - état marche/arrêt machine simulé (MachineStateSimulator)

Règle de fusion (cahier des charges, section 5) :
  ROUGE si gaz critique, proximité machine confirmée ou violation EPI.
  ORANGE si une donnée indispensable est indisponible ou l'état machine inconnu.

Les zones sont définies en coordonnées RELATIVES (0 à 1) dans camera_zones.json
(polygones dans l'image d'une caméra ; les zones du site sont dans zones.json),
donc ce module fonctionne quelle que soit la résolution/orientation de la
vidéo traitée : la conversion en pixels se fait à chaque frame avec la
taille réelle de cette frame.
"""

import json

from app.config import CAMERA_ZONES_PATH


class ZoneChecker:
    def __init__(self, zones_json_path=CAMERA_ZONES_PATH):
        with open(zones_json_path, "r", encoding="utf-8") as f:
            self.zones_config = json.load(f)["zones"]

    def get_polygon_px(self, zone_id, frame_width, frame_height):
        for z in self.zones_config:
            if z["id"] == zone_id:
                return [(round(x * frame_width), round(y * frame_height)) for x, y in z["polygon_relative"]]
        return None

    def _point_in_polygon(self, x, y, polygon_px):
        """Ray casting, sans dépendance externe (pas besoin de shapely/matplotlib)."""
        n = len(polygon_px)
        inside = False
        px, py = x, y
        x1, y1 = polygon_px[0]
        for i in range(n + 1):
            x2, y2 = polygon_px[i % n]
            if py > min(y1, y2):
                if py <= max(y1, y2):
                    if px <= max(x1, x2):
                        if y1 != y2:
                            xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                        if x1 == x2 or px <= xinters:
                            inside = not inside
            x1, y1 = x2, y2
        return inside

    def get_zone_for_point(self, x_px, y_px, frame_width, frame_height):
        """Retourne l'id de la zone contenant ce point pixel, ou None."""
        for z in self.zones_config:
            polygon_px = self.get_polygon_px(z["id"], frame_width, frame_height)
            if self._point_in_polygon(x_px, y_px, polygon_px):
                return z["id"]
        return None

    def get_zone_risk_level(self, zone_id, elapsed_seconds,
                             gas_simulator, machine_state_simulator,
                             has_ppe_violation=False, has_machine_proximity=False):
        """Retourne (niveau, raison) pour une zone à un instant donné."""
        gas_out, gas_reason, _ = gas_simulator.is_out_of_threshold(zone_id, elapsed_seconds)
        readings = gas_simulator.get_readings(zone_id, elapsed_seconds)
        machine_state = machine_state_simulator.get_state_label("default")
        machine_running = machine_state_simulator.is_running("default")

        if gas_out:
            return "ROUGE", f"gaz hors seuils S-HSE-23 ({gas_reason})"
        if has_machine_proximity and machine_running:
            return "ROUGE", "proximité machine en marche"
        if has_ppe_violation:
            return "ROUGE", "violation EPI en zone à risque"
        if readings is None:
            return "ORANGE", "mesures gaz indisponibles"
        if machine_state == "inconnu":
            return "ORANGE", "état machine inconnu ; prudence appliquée aux proximités"
        return "VERT", "aucune violation active ; mesures gaz disponibles"
