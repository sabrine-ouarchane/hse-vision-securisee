"""Time-based gas sensor demonstration for the live dashboard."""

import json
import math
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from threading import RLock


THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "config" / "gas_thresholds.json"
with THRESHOLDS_PATH.open("r", encoding="utf-8") as thresholds_file:
    CONFIG = json.load(thresholds_file)


class DashboardGasSimulator:
    SCENARIOS = {"normal", "rising", "warning", "critical", "recovery", "unavailable"}
    SAMPLE_INTERVAL_S = 5

    def __init__(self):
        self._lock = RLock()
        self._active = True
        self._scenario = "normal"
        self._scenario_started = time.monotonic()
        self._last_sample = 0.0
        self._history = defaultdict(lambda: deque(maxlen=120))
        self._sensor_states = {}
        self._active_alerts = {}
        self._alerts = deque(maxlen=200)
        self._event_counter = 0

    def set_state(self, active=None, scenario=None):
        if scenario is not None and scenario not in self.SCENARIOS:
            raise ValueError(f"Scénario inconnu : {scenario}")
        with self._lock:
            if scenario is not None and scenario != self._scenario:
                self._scenario = scenario
                self._scenario_started = time.monotonic()
            if active is not None:
                self._active = bool(active)

    def _value(self, key, thresholds, elapsed):
        phase = min(elapsed / 120.0, 1.0)
        if key == "O2_pct":
            normal = thresholds["normal"]
            low_target = thresholds["critical_min"] - 0.1
            if self._scenario in {"critical", "warning"}:
                return thresholds["critical_min"] - 0.1 if self._scenario == "critical" else thresholds["warning_min"] - 0.1
            if self._scenario in {"rising", "recovery"}:
                progress = phase if self._scenario == "rising" else 1 - phase
                return normal + (low_target - normal) * progress
            return normal + math.sin(elapsed / 13.0) * 0.03

        normal = thresholds["normal"]
        warning = thresholds["warning"]
        critical = thresholds["critical"]
        if self._scenario == "warning":
            return warning + (critical - warning) * 0.25
        if self._scenario == "critical":
            return critical * 1.05
        if self._scenario == "rising":
            return normal + (critical * 1.05 - normal) * phase
        if self._scenario == "recovery":
            return normal + (critical * 1.05 - normal) * (1 - phase)
        return max(0.0, normal + math.sin(elapsed / 11.0) * max(normal * 0.08, 0.02))

    @staticmethod
    def _status(key, value, thresholds):
        if key == "O2_pct":
            if value <= thresholds["critical_min"] or value >= thresholds["critical_max"]:
                return "critique"
            if value < thresholds["warning_min"] or value > thresholds["warning_max"]:
                return "avertissement"
            return "normal"
        if value >= thresholds["critical"]:
            return "critique"
        if value >= thresholds["warning"]:
            return "avertissement"
        return "normal"

    def _record_status_change(self, zone_id, sensor, now):
        key = (zone_id, sensor["gas"])
        previous = self._sensor_states.get(key, "normal")
        current = sensor["status"]
        active_alert = self._active_alerts.get(key)
        if current == "normal":
            if active_alert:
                active_alert["status"] = "résolue"
                self._active_alerts.pop(key, None)
            self._sensor_states[key] = current
            return
        if current == previous:
            return
        if active_alert:
            active_alert["status"] = "résolue"
        self._event_counter += 1
        severity = "critique" if current == "critique" else "avertissement"
        event = {
            "event_id": f"gas-{zone_id}-{sensor['gas']}-{self._event_counter}",
            "timestamp": now,
            "frame_idx": "",
            "track_id": sensor["sensor_id"],
            "type": "gas",
            "zone_id": zone_id,
            "message": f"{sensor['gas']} {current} : {sensor['value']} {sensor['unit']}",
            "confidence": None,
            "severity": severity,
            "source": "capteur simulé",
            "status": "nouvelle",
        }
        self._alerts.append(event)
        self._active_alerts[key] = event
        self._sensor_states[key] = current

    def set_alert_status(self, event_id, status):
        if status not in {"nouvelle", "en cours", "résolue"}:
            raise ValueError("Statut invalide.")
        with self._lock:
            event = next((item for item in self._alerts if item["event_id"] == event_id), None)
            if event is None:
                return False
            event["status"] = status
            if status == "résolue":
                for key, active_event in list(self._active_alerts.items()):
                    if active_event is event:
                        self._active_alerts.pop(key, None)
            return True

    def snapshot(self, zone_ids):
        with self._lock:
            elapsed = time.monotonic() - self._scenario_started
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            configured_zones = set(CONFIG["zones"])
            zones = {}
            for zone_id in zone_ids:
                if zone_id not in configured_zones:
                    continue
                sensors = []
                values = {}
                for gas_key, limits in CONFIG["gases"].items():
                    unavailable = not self._active or self._scenario == "unavailable"
                    value = None if unavailable else round(self._value(gas_key, limits, elapsed), 3)
                    sensor_status = "indisponible" if unavailable else self._status(gas_key, value, limits)
                    sensor = {
                        "sensor_id": f"{zone_id}-{gas_key.lower()}",
                        "gas": gas_key,
                        "value": value,
                        "unit": limits["unit"],
                        "warning_threshold": (
                            [limits["warning_min"], limits["warning_max"]]
                            if gas_key == "O2_pct"
                            else limits["warning"]
                        ),
                        "critical_threshold": (
                            [limits["critical_min"], limits["critical_max"]]
                            if gas_key == "O2_pct"
                            else limits["critical"]
                        ),
                        "status": sensor_status,
                        "timestamp": now,
                    }
                    sensors.append(sensor)
                    values[gas_key] = value
                    self._record_status_change(zone_id, sensor, now)
                    if not unavailable:
                        if self._last_sample == 0 or time.monotonic() - self._last_sample >= self.SAMPLE_INTERVAL_S:
                            self._history[(zone_id, gas_key)].append({"timestamp": now, "value": value})
                zones[zone_id] = {"readings": values, "sensors": sensors}
            if zones:
                self._last_sample = time.monotonic()
            return {
                "active": self._active,
                "scenario": self._scenario,
                "scenarios": sorted(self.SCENARIOS),
                "demo_only": True,
                "sample_interval_s": self.SAMPLE_INTERVAL_S,
                "zones": zones,
                "alerts": list(self._alerts),
                "history": {
                    f"{zone_id}:{gas_key}": list(samples)
                    for (zone_id, gas_key), samples in self._history.items()
                },
            }
