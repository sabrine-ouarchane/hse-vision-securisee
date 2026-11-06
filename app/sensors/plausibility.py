"""
app/sensors/plausibility.py

Contrôles anti-falsification des mesures capteurs (étape 9).

Une mesure peut être fausse sans être dangereuse en apparence : capteur
débranché, valeur rejouée, valeur forcée par quelqu'un qui a accès au réseau
des capteurs. Chaque capteur est donc contrôlé à chaque mesure :

- donnée absente : aucune mesure reçue depuis `delai_absence_s` ;
- donnée en retard : horodatage de la source trop ancien (ou dans le futur) ;
- valeur figée : exactement la même valeur pendant `duree_figee_s` (sauf
  sous la limite de détection, où un 0 constant est normal) ;
- variation impossible : plus rapide que `vitesse_max_par_s` du capteur
  (une vraie fuite reste en dessous) ;
- hors plage : valeur hors de l'étendue de mesure du capteur ;
- valeur invalide : non numérique, ou état machine inconnu du protocole.

Un capteur pris en défaut reste « non fiable » au moins `quarantaine_s`,
puis tant que le défaut persiste. Même principe que la surveillance
compromise de la caméra : une donnée non fiable ne prouve pas que la zone
est sûre.

Limite connue : une valeur forcée à 0 sur un gaz dont la valeur normale est
proche de 0 (H2S, NO2...) reste plausible ; seules la redondance des
capteurs et l'authentification de la source la détecteraient.
"""

from datetime import datetime

from app.sensors.provider import MACHINE_MEASURE, MACHINE_STATES, Reading
from app.zones.settings import sensor_spec

FAULT_LABELS = {
    "absence": "donnée absente",
    "retard": "donnée en retard",
    "figee": "valeur figée",
    "saut": "variation impossible",
    "hors_plage": "valeur hors plage",
    "invalide": "valeur invalide",
}


def fmt(value) -> str:
    """Nombre au format français, sans zéros inutiles (2.0 -> « 2 », 19.92 -> « 19,92 »)."""
    if isinstance(value, str):
        return value
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return (text if text != "-0" else "0").replace(".", ",")


class _SensorState:
    __slots__ = ("last_value", "last_ts", "last_seen", "same_since", "fault", "reason", "until", "since")

    def __init__(self):
        self.last_value = self.last_ts = self.last_seen = self.same_since = None
        self.fault = self.reason = self.until = self.since = None


class PlausibilityMonitor:
    def __init__(self, config: dict, started_at: datetime):
        self.config = config
        self.params = config["parametres"]
        self.started_at = started_at
        self.specs, self.measures, self.labels = {}, {}, {}
        for zone in config["zones"]:
            for sensor in zone.get("capteurs", []):
                self.specs[sensor["id"]] = sensor_spec(config, sensor)
                self.measures[sensor["id"]] = sensor["mesure"]
                self.labels[sensor["id"]] = config["normes"][sensor["mesure"]]["libelle"]
            for machine in zone.get("machines", []):
                sensor_id = f"{zone['id']}-{machine['id']}"
                self.specs[sensor_id], self.measures[sensor_id] = {}, MACHINE_MEASURE
                self.labels[sensor_id] = f"état {machine['id']}"
        self._states = {sensor_id: _SensorState() for sensor_id in self.specs}

    def observe(self, reading: Reading, now: datetime):
        state = self._states.get(reading.sensor_id)
        if state is None:
            return  # capteur absent de la configuration : ignoré
        state.last_seen = now
        fault = self._check_reading(state, reading, now)
        if fault:
            self._flag(state, *fault, now)
        if isinstance(reading.value, (int, float)) and not isinstance(reading.value, bool) \
                or reading.measure == MACHINE_MEASURE:
            if reading.value != state.last_value:
                state.same_since = reading.timestamp
            state.last_value, state.last_ts = reading.value, reading.timestamp

    def _check_reading(self, state, reading, now):
        delay = self.params["delai_absence_s"]
        label, unit = self.labels[reading.sensor_id], reading.unit
        lag = (now - reading.timestamp).total_seconds()
        if lag > delay:
            return "retard", f"donnée en retard de {fmt(lag)} s"
        if lag < -delay:
            return "retard", f"horodatage dans le futur ({fmt(-lag)} s)"
        if reading.measure == MACHINE_MEASURE:
            if reading.value not in MACHINE_STATES:
                return "invalide", f"état machine invalide ({reading.value!r})"
            return None
        value = reading.value
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value:
            return "invalide", f"{label} : valeur invalide ({value!r})"
        spec = self.specs[reading.sensor_id]
        low, high = spec["plage"]
        if not low <= value <= high:
            return "hors_plage", f"{label} = {fmt(value)} {unit} hors de la plage du capteur ({fmt(low)} à {fmt(high)})"
        if state.last_value is not None and state.last_ts is not None:
            dt = (reading.timestamp - state.last_ts).total_seconds()
            if dt <= 0:
                return "retard", "horodatage non croissant (mesure rejouée ?)"
            rate = abs(value - state.last_value) / dt
            if rate > spec["vitesse_max_par_s"]:
                return "saut", (f"{label} : {fmt(state.last_value)} → {fmt(value)} {unit} en {fmt(dt)} s, "
                                f"variation impossible (max {fmt(spec['vitesse_max_par_s'])} {unit}/s)")
            detection = spec.get("limite_detection")
            ignored = detection is not None and value < detection
            if value == state.last_value and not ignored and state.same_since is not None:
                frozen = (reading.timestamp - state.same_since).total_seconds()
                if frozen >= self.params["duree_figee_s"]:
                    return "figee", f"{label} figé à {fmt(value)} {unit} depuis {fmt(frozen)} s"
        return None

    def _flag(self, state, fault, reason, now):
        if state.fault is None or state.until is None or now.timestamp() >= state.until:
            state.since = now
        state.fault, state.reason = fault, reason
        state.until = now.timestamp() + self.params["quarantaine_s"]

    def status(self, now: datetime) -> dict:
        """{sensor_id: {"fiable", "defaut", "raison", "depuis"}} à l'instant `now`."""
        out = {}
        delay = self.params["delai_absence_s"]
        for sensor_id, state in self._states.items():
            silent = (now - (state.last_seen or self.started_at)).total_seconds()
            if silent > delay:
                if state.fault != "absence":
                    self._flag(state, "absence", "", now)
                state.reason = (f"{self.labels[sensor_id]} : aucune donnée depuis {fmt(round(silent))} s"
                                if state.last_seen else f"{self.labels[sensor_id]} : aucune donnée reçue")
                state.until = now.timestamp() + self.params["quarantaine_s"]
            if state.fault is not None and now.timestamp() >= state.until:
                state.fault = state.reason = state.until = state.since = None
            out[sensor_id] = {
                "fiable": state.fault is None,
                "defaut": state.fault,
                "raison": state.reason,
                "depuis": state.since.isoformat(timespec="seconds") if state.since else None,
            }
        return out
