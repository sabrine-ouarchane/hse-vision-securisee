"""
app/sensors/simulated_provider.py

Simulateur de capteurs (SensorProvider), en l'absence de capteurs réels sur
le site (cahier des charges, objectif 2.3). Chaque capteur émet une mesure
par période (1 Hz par défaut) : horodatage, zone, capteur, mesure, valeur,
unité — comme le ferait une vraie source.

Reproductible : chaque valeur dépend uniquement de la graine, du capteur, du
numéro de seconde et du scénario en cours ; une démonstration rejouée avec
la même graine et les mêmes changements de scénario donne les mêmes mesures,
quelle que soit la fréquence de lecture.

Scénarios par zone (le capteur visé est le capteur principal de la zone, ou
celui qui est choisi) :
- gaz : stable, derive (vert → orange → rouge), fuite (pic en quelques
  secondes), retour (vers la normale) ;
- pannes / attaques : muet (plus de données), fige (dernière valeur
  répétée), saut (variation impossible), zero (valeur forcée à 0),
  hors_plage (valeur hors de l'étendue du capteur) ;
- machines : arret, marche, inconnu, declare (état déclaré dans
  config/machine_states.json, comme dans /live), muet.

Les courbes reprennent celles de l'ancien simulateur gaz
(dashboard_gas_simulator.py) : O2 traité vers sa borne basse, montée
linéaire pour la dérive, décroissance pour le retour.
"""

import math
import random
import threading
from datetime import datetime, timedelta

from app.sensors.provider import MACHINE_MEASURE, MACHINE_STATES, Reading, SensorInfo, SensorProvider
from app.zones.settings import sensor_spec

GAS_SCENARIOS = ("stable", "derive", "fuite", "retour", "muet", "fige", "saut", "zero", "hors_plage")
MACHINE_SCENARIOS = ("arret", "marche", "inconnu", "declare", "muet")
DEFAULT_SCENARIO = {"gaz": "stable", "machine": "declare"}
SCENARIO_LABELS = {
    "stable": "Stable", "derive": "Dérive lente", "fuite": "Fuite (pic brutal)", "retour": "Retour à la normale",
    "muet": "Capteur muet", "fige": "Valeur figée", "saut": "Saut impossible", "zero": "Valeur forcée à 0",
    "hors_plage": "Hors plage", "arret": "Arrêt", "marche": "Marche", "inconnu": "État inconnu",
    "declare": "État déclaré (machine_states.json)",
}
MAX_CATCH_UP = 120  # au plus 2 min de mesures rattrapées après une interruption


class SimulatedSensorProvider(SensorProvider):
    def __init__(self, config: dict, seed: int | None = None, declared_state=None):
        """
        Args:
            config: contenu de config/zones.json.
            seed: graine (défaut : simulation.graine du fichier).
            declared_state: fonction machine_id -> "marche" / "arret" / "inconnu"
                pour le scénario « declare » (ex. MachineStateProvider).
        """
        self.config = config
        simulation = config.get("simulation", {})
        self.seed = simulation.get("graine", 0) if seed is None else seed
        self.period_s = float(simulation.get("periode_s", 1))
        self.durations = {k: float(simulation.get(f"duree_{k}_s", v))
                          for k, v in (("derive", 180), ("fuite", 5), ("retour", 60))}
        self.declared_state = declared_state
        self._lock = threading.Lock()
        self._running = False
        self._t0 = None
        self._last_tick = -1
        self._last_value = {}
        self._zones = {z["id"]: z for z in config["zones"] if z["methode"] in ("gaz", "machine")}
        self._sensors = {}
        for zone in self._zones.values():
            for sensor in zone.get("capteurs", []):
                self._sensors[sensor["id"]] = {"zone": zone["id"], "measure": sensor["mesure"],
                                               "spec": sensor_spec(config, sensor), "machine": None}
            for machine in zone.get("machines", []):
                self._sensors[f"{zone['id']}-{machine['id']}"] = {"zone": zone["id"], "measure": MACHINE_MEASURE,
                                                                   "spec": {}, "machine": machine["id"]}
        self._scenarios = {zone_id: self._scenario_entry(DEFAULT_SCENARIO[zone["methode"]], None, 0)
                           for zone_id, zone in self._zones.items()}

    # ------------------------------------------------------------------
    # Interface SensorProvider
    def sensors(self) -> list[SensorInfo]:
        return [SensorInfo(sensor_id, s["zone"], s["measure"], self._unit(s), self.period_s, s["machine"])
                for sensor_id, s in self._sensors.items()]

    def poll(self, now: datetime) -> list[Reading]:
        with self._lock:
            if not self._running:
                return []
            current = self._tick_at(now)
            first = max(self._last_tick + 1, current - MAX_CATCH_UP + 1)
            readings = []
            for tick in range(first, current + 1):
                timestamp = self._t0 + timedelta(seconds=tick * self.period_s)
                for sensor_id, sensor in self._sensors.items():
                    value = self._value(sensor_id, sensor, tick)
                    if value is _SILENT:
                        continue
                    self._last_value[sensor_id] = value
                    readings.append(Reading(timestamp, sensor["zone"], sensor_id, sensor["measure"], value,
                                            self._unit(sensor)))
            self._last_tick = max(self._last_tick, current)
            return readings

    def status(self) -> dict:
        with self._lock:
            return {
                "actif": self._running,
                "source": "simulateur",
                "graine": self.seed,
                "demarre_a": self._t0.isoformat(timespec="seconds") if self._running else None,
                "scenarios": {zone_id: {"scenario": entry["scenario"], "capteur": entry["capteur"]}
                              for zone_id, entry in self._scenarios.items()},
            }

    # ------------------------------------------------------------------
    # Pilotage (propre au simulateur)
    @property
    def running(self) -> bool:
        return self._running

    def start(self, now: datetime):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._t0 = now
            self._last_tick = self._tick_at(now) - 1
            self._last_value.clear()
            for entry in self._scenarios.values():
                entry["start_tick"], entry["start_values"] = 0, {}

    def stop(self):
        with self._lock:
            self._running = False

    def scenarios_for(self, zone_id: str) -> tuple[str, ...]:
        return GAS_SCENARIOS if self._zones[zone_id]["methode"] == "gaz" else MACHINE_SCENARIOS

    def set_scenario(self, zone_id: str, scenario: str, sensor_id: str | None = None,
                     now: datetime | None = None) -> dict:
        """Change le scénario d'une zone ; il prend effet à la mesure suivante."""
        if zone_id not in self._zones:
            raise ValueError(f"Zone inconnue ou non simulée : {zone_id}")
        if scenario not in self.scenarios_for(zone_id):
            raise ValueError(f"Scénario « {scenario} » invalide pour la zone {zone_id}.")
        zone_sensors = [s for s, info in self._sensors.items() if info["zone"] == zone_id]
        if sensor_id is not None and sensor_id not in zone_sensors:
            raise ValueError(f"Capteur {sensor_id} absent de la zone {zone_id}.")
        if sensor_id is None and self._zones[zone_id]["methode"] == "gaz":
            sensor_id = self._zones[zone_id].get("capteur_principal") or zone_sensors[0]
        with self._lock:
            start = self._tick_at(now) + 1 if (self._running and now is not None) else self._last_tick + 1
            entry = self._scenario_entry(scenario, sensor_id, start)
            entry["start_values"] = {s: self._last_value[s] for s in zone_sensors if s in self._last_value}
            previous = self._scenarios[zone_id]
            self._scenarios[zone_id] = entry
            return {"zone": zone_id, "ancien": previous["scenario"], "nouveau": scenario, "capteur": sensor_id}

    # ------------------------------------------------------------------
    @staticmethod
    def _scenario_entry(scenario, sensor_id, start_tick):
        return {"scenario": scenario, "capteur": sensor_id, "start_tick": start_tick, "start_values": {}}

    def _tick_at(self, now):
        return math.floor((now - self._t0).total_seconds() / self.period_s + 1e-9)

    def _unit(self, sensor):
        if sensor["measure"] == MACHINE_MEASURE:
            return ""
        return self.config["normes"][sensor["measure"]]["unite"]

    def _value(self, sensor_id, sensor, tick):
        entry = self._scenarios[sensor["zone"]]
        if sensor["machine"] is not None:
            return self._machine_value(sensor, entry)
        targeted = entry["capteur"] == sensor_id
        scenario = entry["scenario"] if targeted else "stable"
        elapsed = (tick - entry["start_tick"]) * self.period_s
        spec = sensor["spec"]
        if scenario == "muet":
            return _SILENT
        start = entry["start_values"].get(sensor_id, spec["normale"])
        if scenario == "fige":
            return start
        if scenario == "zero":
            return 0.0
        rng = random.Random(f"{self.seed}|{sensor_id}|{tick}")
        noise = rng.gauss(0.0, spec["bruit"])
        low, high = spec["plage"]
        if scenario == "hors_plage":
            return self._round(high + 0.5 * (high - low) + abs(noise), spec)
        base = self._base_value(sensor, scenario, elapsed, start)
        return self._round(min(max(base + noise, low), high), spec)

    def _base_value(self, sensor, scenario, elapsed, start):
        spec, normal = sensor["spec"], sensor["spec"]["normale"]
        if scenario == "stable":
            return normal
        danger_slow, danger_fast = self._danger_levels(sensor)
        if scenario == "derive":
            return normal + (danger_slow - normal) * min(elapsed / self.durations["derive"], 1.0)
        if scenario == "fuite":
            # pic brutal mais physiquement possible : au plus la moitié de la vitesse maximale
            duration = max(self.durations["fuite"], abs(danger_fast - start) / (0.5 * spec["vitesse_max_par_s"]))
            return start + (danger_fast - start) * min(elapsed / duration, 1.0)
        if scenario == "retour":
            return normal + (start - normal) * math.exp(-3.0 * elapsed / self.durations["retour"])
        if scenario == "saut":
            direction = -1.0 if sensor["measure"] == "O2_pct" else 1.0
            return start + direction * 10 * spec["vitesse_max_par_s"]
        return normal

    def _danger_levels(self, sensor):
        """Niveaux atteints par la dérive (juste au-delà du seuil) et par la fuite (bien au-delà)."""
        norm = self.config["normes"][sensor["measure"]]
        if "min" in norm:  # O2 : danger vers la borne basse (appauvrissement)
            return norm["min"] - 1.0, norm["min"] - 3.5
        reference, threshold = norm.get("reference", 0), norm["max"]
        if threshold > reference:
            return reference + 1.25 * (threshold - reference), reference + 3.0 * (threshold - reference)
        detection = sensor["spec"].get("limite_detection") or sensor["spec"]["resolution"]
        return 5 * detection, 20 * detection  # gaz à seuil 0 ppm

    def _machine_value(self, sensor, entry):
        scenario = entry["scenario"]
        if scenario == "muet":
            if entry["capteur"] in (None, f"{sensor['zone']}-{sensor['machine']}"):
                return _SILENT
            scenario = "declare"  # les autres machines de la zone continuent d'émettre
        if scenario == "declare":
            state = self.declared_state(sensor["machine"]) if self.declared_state else "inconnu"
            return state if state in MACHINE_STATES else "inconnu"
        return scenario if scenario in MACHINE_STATES else "inconnu"

    @staticmethod
    def _round(value, spec):
        step = spec["resolution"]
        return round(round(value / step) * step, 6)


_SILENT = object()  # aucune mesure émise (capteur muet)
