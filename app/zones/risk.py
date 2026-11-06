"""
app/zones/risk.py

Niveau de risque des zones dangereuses du site (étape 9, objectif 2.3).

Zones GAZ (seuils S-HSE-23, config/zones.json) :
- ROUGE  : un seuil dépassé, ou une donnée capteur non fiable ;
- ORANGE : une valeur au-delà du seuil orange, ou une mesure instable ;
- VERT   : toutes les valeurs sous le seuil orange, et stables.
  Seuil orange = référence + pourcentage_orange × (seuil − référence) ;
  pour O2, chaque borne (19,5 et 23,5 %) est traitée séparément à partir
  de l'air ambiant (20,9 %).
  Gaz à seuil 0 ppm (NO, NO2) : sous la limite de détection, la valeur vaut
  0 ; détecté sur une mesure → ORANGE (confirmation en cours) ; détecté
  pendant `confirmation_zero_s` → ROUGE.
  Instable : la pente sur la fenêtre glissante atteindrait le seuil avant
  `horizon_derive_s`, ou l'écart-type autour de la tendance dépasse
  `facteur_variabilite` × le bruit normal du capteur.

Zones MACHINE (S-HSE-22 : état inconnu = en marche) :
- VERT   : arrêt confirmé de toutes les machines de la zone ;
- ORANGE : en marche ou état inconnu, aucune personne détectée ou présence
  non vérifiable ;
- ROUGE  : en marche ou état inconnu ET personne détectée, ou proximité
  dangereuse active, ou caméra compromise (présence invérifiable) ;
  une caméra compromise empêche aussi le VERT (ORANGE au minimum).

Hystérésis : la montée est immédiate ; la descente se fait d'un niveau à la
fois, après `hysteresis_s` passées sous le niveau affiché.
Zones hors périmètre et surveillance arrêtée : GRIS, non évaluées.

Les fonctions sont pures (le temps est passé en argument) pour être testées
sur des séries synthétiques.
"""

from collections import deque
from datetime import datetime

from app.sensors.plausibility import FAULT_LABELS, PlausibilityMonitor, fmt
from app.sensors.provider import MACHINE_MEASURE, Reading
from app.zones.settings import sensor_spec

LEVELS = ("GRIS", "VERT", "ORANGE", "ROUGE")
RANK = {level: index for index, level in enumerate(LEVELS)}
# À niveau égal, la raison principale affichée suit cet ordre
CAUSE_PRIORITY = {"capteur": 0, "machine": 1, "gaz": 2, "instabilite": 3}
UNKNOWN_PRESENCE = {"etat": "inconnue", "proximite": False, "camera_compromise": False, "source": None}


# ---------------------------------------------------------------------------
# Règles élémentaires (sans état)
# ---------------------------------------------------------------------------
def orange_limits(norm: dict, percent: float) -> dict:
    """Seuils orange d'une mesure : {"bas": x} et/ou {"haut": y}."""
    share = percent / 100.0
    reference = norm.get("reference", 0)
    if "min" in norm:
        return {"bas": reference - share * (reference - norm["min"]), "haut": reference + share * (norm["max"] - reference)}
    return {"haut": reference + share * (norm["max"] - reference)}


def classify_value(norm: dict, spec: dict, value: float, percent: float) -> tuple[str, str | None]:
    """Niveau d'une valeur isolée (sans la confirmation des gaz à seuil 0) et sa raison."""
    label, unit = norm["libelle"], norm["unite"]
    shown = f"{label} = {fmt(value)} {unit}"
    if "min" in norm:
        limits = orange_limits(norm, percent)
        if value < norm["min"]:
            return "ROUGE", f"{shown} < {fmt(norm['min'])} {unit} (seuil S-HSE-23)"
        if value > norm["max"]:
            return "ROUGE", f"{shown} > {fmt(norm['max'])} {unit} (seuil S-HSE-23)"
        if value < limits["bas"]:
            return "ORANGE", f"{shown} < {fmt(limits['bas'])} {unit} (orange : {fmt(percent)} % de l'écart vers {fmt(norm['min'])} {unit})"
        if value > limits["haut"]:
            return "ORANGE", f"{shown} > {fmt(limits['haut'])} {unit} (orange : {fmt(percent)} % de l'écart vers {fmt(norm['max'])} {unit})"
        return "VERT", None
    if norm["max"] <= norm.get("reference", 0):  # seuil 0 ppm : toute présence détectable compte
        detection = spec.get("limite_detection") or spec["resolution"]
        if value < detection:
            return "VERT", None
        return "ORANGE", f"{label} détecté : {fmt(value)} {unit} (seuil {fmt(norm['max'])} {unit}, confirmation en cours)"
    if value >= norm["max"]:
        return "ROUGE", f"{shown} ≥ {fmt(norm['max'])} {unit} (seuil S-HSE-23)"
    limit = orange_limits(norm, percent)["haut"]
    if value >= limit:
        return "ORANGE", f"{shown} ≥ {fmt(limit)} {unit} ({fmt(percent)} % du seuil de {fmt(norm['max'])} {unit})"
    return "VERT", None


def _duration(seconds: float) -> str:
    return f"≈ {fmt(round(seconds / 60))} min" if seconds >= 90 else f"≈ {fmt(round(seconds))} s"


def instability(norm: dict, spec: dict, window, params: dict) -> str | None:
    """Raison d'instabilité sur la fenêtre [(t, valeur), ...], ou None si stable."""
    if len(window) < max(10, params["fenetre_s"] / 4):
        return None
    times = [t - window[0][0] for t, _ in window]
    values = [v for _, v in window]
    n = len(values)
    mean_t, mean_v = sum(times) / n, sum(values) / n
    var_t = sum((t - mean_t) ** 2 for t in times)
    if var_t <= 0:
        return None
    slope = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values)) / var_t
    residuals = [v - (mean_v + slope * (t - mean_t)) for t, v in zip(times, values)]
    spread = (sum(r * r for r in residuals) / max(n - 2, 1)) ** 0.5
    label, unit = norm["libelle"], norm["unite"]
    span = times[-1] - times[0]
    current = mean_v + slope * (times[-1] - mean_t)
    if abs(slope) * span >= 3 * spec["bruit"]:  # tendance nette, au-delà du bruit
        if "min" in norm:
            bound = norm["min"] if slope < 0 else norm["max"]
        elif norm["max"] <= norm.get("reference", 0):
            bound = spec.get("limite_detection") or spec["resolution"]
        else:
            bound = norm["max"]
        remaining = (bound - current) / slope
        if 0 < remaining <= params["horizon_derive_s"]:
            direction = "en hausse" if slope > 0 else "en baisse"
            return (f"{label} {direction} ({'+' if slope > 0 else ''}{fmt(slope * 60)} {unit}/min) : "
                    f"{fmt(bound)} {unit} atteint dans {_duration(remaining)}")
    if spread > params["facteur_variabilite"] * spec["bruit"]:
        return (f"{label} instable : écart-type {fmt(spread)} {unit} "
                f"(> {fmt(params['facteur_variabilite'])} × bruit normal {fmt(spec['bruit'])} {unit})")
    return None


def machine_level(state: str, presence: dict) -> tuple[str, str]:
    """Niveau d'une zone machine selon l'état agrégé de ses machines et la présence (caméra /live)."""
    running = state in ("marche", "inconnu")
    what = "machine en marche" if state == "marche" else "état machine inconnu (S-HSE-22 : considéré en marche)"
    if presence.get("camera_compromise"):
        if running:
            return "ROUGE", f"{what} et caméra compromise : présence non vérifiable"
        return "ORANGE", "arrêt confirmé, mais caméra compromise : la zone ne peut pas être déclarée sûre"
    if not running:
        return "VERT", "arrêt confirmé de toutes les machines"
    if presence.get("proximite"):
        return "ROUGE", f"{what} et proximité dangereuse homme-machine active"
    if presence.get("etat") == "presente":
        return "ROUGE", f"{what} et personne détectée dans la zone"
    if presence.get("etat") == "absente":
        return "ORANGE", f"{what}, aucune personne détectée"
    source = presence.get("source")
    detail = "caméra non analysée dans /live" if source else "aucune caméra associée"
    return "ORANGE", f"{what}, présence non vérifiable ({detail})"


def aggregate_machine_state(states: list[str]) -> str:
    if any(s == "marche" for s in states):
        return "marche"
    if not states or any(s != "arret" for s in states):
        return "inconnu"
    return "arret"


# ---------------------------------------------------------------------------
# Évaluateur avec état (fenêtres, plausibilité, confirmation, hystérésis)
# ---------------------------------------------------------------------------
class ZoneRiskEvaluator:
    def __init__(self, config: dict, started_at: datetime):
        self.config = config
        self.params = config["parametres"]
        self.started_at = started_at
        self.plausibility = PlausibilityMonitor(config, started_at)
        self._windows, self._latest, self._detected_since = {}, {}, {}
        self._shown = {}  # zone -> {"niveau", "raison", "plus_bas_depuis"}

    def ingest(self, readings: list[Reading], now: datetime):
        for reading in readings:
            self.plausibility.observe(reading, now)
            self._latest[reading.sensor_id] = reading
            if reading.measure == MACHINE_MEASURE or not isinstance(reading.value, (int, float)):
                continue
            window = self._windows.setdefault(reading.sensor_id, deque())
            t = reading.timestamp.timestamp()
            window.append((t, float(reading.value)))
            while window and window[0][0] < t - self.params["fenetre_s"]:
                window.popleft()
            self._track_detection(reading)

    def _track_detection(self, reading):
        """Début de la détection continue d'un gaz à seuil 0 ppm (au-dessus de la limite de détection)."""
        zone_sensor = self._sensor_config(reading.sensor_id)
        if zone_sensor is None:
            return
        norm = self.config["normes"][reading.measure]
        if norm["max"] > norm.get("reference", 0):
            return
        spec = sensor_spec(self.config, zone_sensor)
        detection = spec.get("limite_detection") or spec["resolution"]
        if reading.value >= detection:
            self._detected_since.setdefault(reading.sensor_id, reading.timestamp)
        else:
            self._detected_since.pop(reading.sensor_id, None)

    def _sensor_config(self, sensor_id):
        for zone in self.config["zones"]:
            for sensor in zone.get("capteurs", []):
                if sensor["id"] == sensor_id:
                    return sensor
        return None

    def evaluate(self, now: datetime, presence: dict | None = None, active: bool = True) -> dict:
        """État de chaque zone : niveau affiché, raison, causes, détail des capteurs et machines."""
        presence = presence or {}
        reliability = self.plausibility.status(now) if active else {}
        result = {}
        for zone in self.config["zones"]:
            zone_id = zone["id"]
            if zone["methode"] == "hors_perimetre" or not active:
                self._shown.pop(zone_id, None)
                reason = "hors périmètre (non évalué)" if zone["methode"] == "hors_perimetre" else "surveillance arrêtée"
                result[zone_id] = {"zone": zone_id, "nom": zone["nom"], "methode": zone["methode"], "niveau": "GRIS",
                                   "niveau_brut": "GRIS", "raison": reason, "raison_actuelle": reason, "causes": [],
                                   "capteurs": [], "machines": [], "presence": None}
                continue
            if zone["methode"] == "gaz":
                causes, details = self._gas_zone(zone, reliability, now)
                machines, zone_presence = [], None
                default_reason = (f"toutes les mesures sous {fmt(self.params['pourcentage_orange'])} % "
                                  "des seuils et stables")
            else:
                zone_presence = {**UNKNOWN_PRESENCE, **presence.get(zone_id, {})}
                causes, machines = self._machine_zone(zone, reliability, zone_presence)
                details = []
                default_reason = "arrêt confirmé de toutes les machines"
            causes.sort(key=lambda c: (-RANK[c["niveau"]], CAUSE_PRIORITY[c["type"]]))
            raw = causes[0]["niveau"] if causes else "VERT"
            raw_reason = causes[0]["raison"] if causes else default_reason
            level, reason = self._hysteresis(zone_id, raw, raw_reason, now)
            result[zone_id] = {"zone": zone_id, "nom": zone["nom"], "methode": zone["methode"], "niveau": level,
                               "niveau_brut": raw, "raison": reason, "raison_actuelle": raw_reason, "causes": causes,
                               "capteurs": details, "machines": machines, "presence": zone_presence}
        return result

    def _gas_zone(self, zone, reliability, now):
        causes, details = [], []
        percent = self.params["pourcentage_orange"]
        for sensor in zone["capteurs"]:
            sensor_id, measure = sensor["id"], sensor["mesure"]
            norm, spec = self.config["normes"][measure], sensor_spec(self.config, sensor)
            latest = self._latest.get(sensor_id)
            health = reliability.get(sensor_id, {"fiable": True, "defaut": None, "raison": None})
            value = latest.value if latest is not None else None
            level, reason, unstable = "VERT", None, None
            if not health["fiable"]:
                level, reason = "ROUGE", f"donnée capteur non fiable : {health['raison']}"
                causes.append({"type": "capteur", "cle": sensor_id, "niveau": "ROUGE", "raison": reason,
                               "defaut": health["defaut"], "defaut_libelle": FAULT_LABELS.get(health["defaut"])})
            elif isinstance(value, (int, float)):
                level, reason = classify_value(norm, spec, value, percent)
                since = self._detected_since.get(sensor_id)
                if level == "ORANGE" and since is not None and norm["max"] <= norm.get("reference", 0):
                    lasted = (latest.timestamp - since).total_seconds()
                    if lasted >= self.params["confirmation_zero_s"]:
                        level = "ROUGE"
                        reason = (f"{norm['libelle']} = {fmt(value)} {norm['unite']} > {fmt(norm['max'])} "
                                  f"{norm['unite']} (seuil S-HSE-23), confirmé depuis {fmt(round(lasted))} s")
                if level != "VERT":
                    causes.append({"type": "gaz", "cle": measure, "capteur": sensor_id, "niveau": level,
                                   "raison": reason})
                unstable = instability(norm, spec, list(self._windows.get(sensor_id, ())), self.params)
                if unstable:
                    causes.append({"type": "instabilite", "cle": measure, "capteur": sensor_id, "niveau": "ORANGE",
                                   "raison": unstable})
                    if level == "VERT":
                        level, reason = "ORANGE", unstable
            details.append({
                "capteur": sensor_id, "mesure": measure, "libelle": norm["libelle"], "unite": norm["unite"],
                "valeur": value, "horodatage": latest.timestamp.isoformat(timespec="seconds") if latest else None,
                "seuil_min": norm.get("min"), "seuil_max": norm["max"], "orange": orange_limits(norm, percent),
                "limite_detection": spec.get("limite_detection"), "niveau": level, "raison": reason,
                "instabilite": unstable, "fiable": health["fiable"], "defaut": health.get("defaut"),
                "raison_fiabilite": health.get("raison"),
            })
        return causes, details

    def _machine_zone(self, zone, reliability, presence):
        causes, machines, states = [], [], []
        for machine in zone["machines"]:
            sensor_id = f"{zone['id']}-{machine['id']}"
            latest = self._latest.get(sensor_id)
            health = reliability.get(sensor_id, {"fiable": True, "defaut": None, "raison": None})
            state = latest.value if latest is not None else "inconnu"
            if not health["fiable"]:
                state = "inconnu"
                causes.append({"type": "capteur", "cle": sensor_id, "niveau": "ROUGE",
                               "raison": f"donnée capteur non fiable : {health['raison']}",
                               "defaut": health["defaut"], "defaut_libelle": FAULT_LABELS.get(health["defaut"])})
            states.append(state)
            machines.append({"machine": machine["id"], "type": machine.get("type"), "capteur": sensor_id,
                             "etat": state, "fiable": health["fiable"], "raison_fiabilite": health.get("raison"),
                             "horodatage": latest.timestamp.isoformat(timespec="seconds") if latest else None})
        level, reason = machine_level(aggregate_machine_state(states), presence)
        if level != "VERT":
            causes.append({"type": "machine", "cle": "exposition", "niveau": level, "raison": reason})
        return causes, machines

    def _hysteresis(self, zone_id, raw, raw_reason, now):
        shown = self._shown.get(zone_id)
        if shown is None or RANK[raw] >= RANK[shown["niveau"]]:
            self._shown[zone_id] = {"niveau": raw, "raison": raw_reason, "plus_bas_depuis": None}
            return raw, raw_reason
        if shown["plus_bas_depuis"] is None:
            shown["plus_bas_depuis"] = now
        waited = (now - shown["plus_bas_depuis"]).total_seconds()
        delay = self.params["hysteresis_s"]
        if waited >= delay:
            lower = LEVELS[max(RANK[raw], RANK[shown["niveau"]] - 1)]
            shown["niveau"] = lower
            shown["plus_bas_depuis"] = now if RANK[lower] > RANK[raw] else None
            shown["raison"] = raw_reason if lower == raw else shown["raison"]
            if lower == raw:
                return raw, raw_reason
            waited = 0.0
        return shown["niveau"], (f"{shown['raison']} — maintenu en {shown['niveau']} (hystérésis : redescente "
                                 f"dans {fmt(round(delay - waited))} s si la situation reste meilleure)")
