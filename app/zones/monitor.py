"""
app/zones/monitor.py

Surveillance des zones dangereuses (étape 9) : une boucle à 1 Hz dans le
serveur lit la source de mesures (SensorProvider), classe les zones
(ZoneRiskEvaluator), ouvre et clôt les incidents, et alimente l'historique.

Incidents : un incident par cause passée au ROUGE (gaz dépassé, machine en
marche avec personne exposée, capteur non fiable), avec début, fin, durée
et raison. Il est écrit dans alerts_log.csv, dans le journal d'audit chaîné
(incident_ouvert / incident_cloture, session « zones ») et dans
data/outputs/zones/incidents.jsonl ; le contrôle de cohérence compare les
trois. Il se clôt quand la cause a disparu depuis `hysteresis_s` (comme le
niveau affiché), ou à l'arrêt de la surveillance.

Actions auditées avec leur auteur : démarrage / arrêt de la surveillance,
changement de scénario de simulation, modification (ou tentative refusée)
des réglages de config/zones.json.

La présence d'une personne dans une zone machine vient de /live : seule la
zone dont la caméra est analysée a une présence connue (une source à la fois).
"""

import threading
from datetime import datetime

from app.alerts.alert_manager import AlertManager
from app.alerts.incident_manager import CYBER_KINDS
from app.config import ZONES_PATH, load_camera_profile
from app.sensors.simulated_provider import SCENARIO_LABELS, SimulatedSensorProvider
from app.zones.history import ZoneHistory, downsample
from app.zones.risk import ZoneRiskEvaluator, orange_limits
from app.zones.settings import PARAM_BOUNDS, ConfigError, apply_parameter_changes, load_zones_config

SESSION_ID = "zones"
SYSTEM = {"username": "systeme", "role": "systeme"}
INCIDENT_KINDS = {
    "gaz": ("zone_gaz", "Zone · seuil gaz dépassé"),
    "machine": ("zone_machine", "Zone · exposition machine en marche"),
    "capteur": ("cyber", "CYBER · Donnée capteur non fiable"),
}


def _now():
    return datetime.now().astimezone()


class ZoneMonitor:
    def __init__(self, audit, alerts_path, data_dir, config_path=ZONES_PATH, presence=None, declared_state=None,
                 clock=_now):
        """
        Args:
            audit: journal d'audit (AuditLog).
            presence: fonction sans argument -> {zone_id: {"etat", "proximite", "camera_compromise", "source"}}
                (LiveController.zone_presence).
            declared_state: fonction machine_id -> état déclaré (scénario « declare »).
            clock: horloge (tests).
        """
        self.audit = audit
        self.config_path = config_path
        self.config = load_zones_config(config_path)
        self.presence_fn = presence
        self.clock = clock
        self.provider = SimulatedSensorProvider(self.config, declared_state=declared_state)
        self.alerts = AlertManager(str(alerts_path), cooldown_frames=0)
        self.history = ZoneHistory(data_dir)
        self.evaluator = None
        self._lock = threading.RLock()
        self._open = {}
        self._counter = 0
        self._last_present = {}
        now = self.clock()
        self.stopped = {"par": "systeme", "a": now.isoformat(timespec="seconds"), "motif": "démarrage du serveur"}
        self.started = None
        self.states = ZoneRiskEvaluator(self.config, now).evaluate(now, active=False)
        self._thread = None
        self._loop_stop = threading.Event()

    # ------------------------------------------------------------------
    # Pilotage (rôles vérifiés par le serveur)
    def start(self, user=None):
        user = user or SYSTEM
        with self._lock:
            if self.provider.running:
                raise RuntimeError("La surveillance des zones est déjà démarrée.")
            now = self.clock()
            self.provider.start(now)
            self.evaluator = ZoneRiskEvaluator(self.config, now)
            self.stopped = None
            self.started = {"par": user["username"], "a": now.isoformat(timespec="seconds")}
            self._log("surveillance_zones_demarree", {
                "source": "simulateur", "graine": self.provider.seed,
                "scenarios": self.provider.status()["scenarios"],
            }, user)
        self.tick()
        return self.status()

    def stop(self, user=None):
        user = user or SYSTEM
        with self._lock:
            if not self.provider.running:
                raise RuntimeError("La surveillance des zones est déjà arrêtée.")
            now = self.clock()
            self.provider.stop()
            self._close_all(now, "surveillance arrêtée")
            self.evaluator = None
            self.stopped = {"par": user["username"], "a": now.isoformat(timespec="seconds"), "motif": "arrêt demandé"}
            self.started = None
            self._log("surveillance_zones_arretee", {"motif": "arrêt demandé"}, user)
        self.tick()
        return self.status()

    def set_scenario(self, zone_id, scenario, sensor_id=None, user=None):
        user = user or SYSTEM
        for name, value in (("zone", zone_id), ("scénario", scenario)):
            if not isinstance(value, str) or not value or len(value) > 64:
                raise ValueError(f"{name.capitalize()} invalide.")
        if sensor_id is not None and (not isinstance(sensor_id, str) or len(sensor_id) > 64):
            raise ValueError("Capteur invalide.")
        with self._lock:
            change = self.provider.set_scenario(zone_id, scenario, sensor_id, self.clock())
            self._log("scenario_simulation_modifie", change, user)
        return change

    def update_parameters(self, changes, user=None):
        """Réglages modifiables depuis l'interface (admin), dans les bornes sûres ; refus audité."""
        user = user or SYSTEM
        with self._lock:
            try:
                config, diff = apply_parameter_changes(changes, self.config_path)
            except ConfigError as error:
                requested = changes if isinstance(changes, dict) else {"valeur": str(changes)[:200]}
                self._log("zones_parametres_refuses", {"demande": {str(k)[:64]: str(v)[:64]
                                                                   for k, v in list(requested.items())[:20]},
                                                       "motif": str(error)}, user)
                raise
            if diff:
                # même dictionnaire que celui de l'évaluateur et des contrôles de plausibilité
                self.config["parametres"].update(config["parametres"])
                self._log("zones_parametres_modifies", {
                    "champs": {name: {"avant": old, "apres": new} for name, (old, new) in diff.items()},
                }, user)
        return {"modifies": {name: new for name, (_, new) in diff.items()}, "parametres": self.parameters()}

    # ------------------------------------------------------------------
    # Boucle
    def start_loop(self, period_s: float = 1.0):
        if self._thread is not None:
            return
        self._loop_stop.clear()

        def loop():
            while not self._loop_stop.wait(period_s):
                try:
                    self.tick()
                except Exception as error:  # la boucle ne doit jamais s'arrêter seule
                    print(f"[ZONES] Erreur de la boucle de surveillance : {error}")
        self._thread = threading.Thread(target=loop, name="zones", daemon=True)
        self._thread.start()

    def stop_loop(self):
        self._loop_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def tick(self, now: datetime | None = None):
        with self._lock:
            now = now or self.clock()
            readings = self.provider.poll(now)
            if self.evaluator is None:
                states = ZoneRiskEvaluator(self.config, now).evaluate(now, active=False)
            else:
                self.evaluator.ingest(readings, now)
                states = self.evaluator.evaluate(now, self._presence(now))
                self._update_incidents(states, now)
            self.history.record_samples(readings)
            self.history.record_levels(now, states)
            self.states = states
            return states

    def _presence(self, now):
        try:
            raw = self.presence_fn() if self.presence_fn else {}
        except Exception:  # /live indisponible : présence inconnue
            raw = {}
        hold = self.config["parametres"]["presence_maintien_s"]
        presence = {}
        for zone_id, info in raw.items():
            info = dict(info)
            if info.get("etat") == "presente":
                self._last_present[zone_id] = now
            elif info.get("etat") == "absente" and zone_id in self._last_present \
                    and (now - self._last_present[zone_id]).total_seconds() <= hold:
                info["etat"] = "presente"  # personne vue il y a moins de presence_maintien_s
            presence[zone_id] = info
        return presence

    # ------------------------------------------------------------------
    # Incidents
    def _update_incidents(self, states, now):
        current = {}
        for zone_id, state in states.items():
            for cause in state["causes"]:
                if cause["niveau"] == "ROUGE" and cause["type"] in INCIDENT_KINDS:
                    current.setdefault((zone_id, cause["type"], cause["cle"]), (state, cause))
        for key, (state, cause) in current.items():
            incident = self._open.get(key)
            if incident is None:
                self._open[key] = self._open_incident(state, cause, now)
            else:
                incident["_vu"] = now
                incident["raison_finale"] = cause["raison"]
        delay = self.config["parametres"]["hysteresis_s"]
        for key in [k for k in self._open if k not in current]:
            if (now - self._open[key]["_vu"]).total_seconds() >= delay:
                self._close_incident(self._open.pop(key), "situation revenue sous le seuil rouge")

    def _open_incident(self, state, cause, now):
        self._counter += 1
        category, label = INCIDENT_KINDS[cause["type"]]
        incident = {
            "id": f"Z{now.strftime('%Y%m%d%H%M%S')}-{self._counter}",
            "zone": state["zone"], "zone_nom": state["nom"], "kind": cause["type"], "categorie": category,
            "label": label, "cle": cause["cle"], "raison": cause["raison"], "severity": "critique",
            "debut": now.isoformat(timespec="seconds"), "fin": None, "duree_s": None,
        }
        if cause["type"] == "gaz":
            incident["mesure"] = cause["cle"]
        if cause["type"] == "capteur":
            incident["defaut"] = cause.get("defaut")
        source = f"Zone {state['nom']}"
        incident["csv_event_id"] = self.alerts.send_alert(
            track_id=None, message=f"[{source}] {cause['raison']}", frame_idx="", alert_type=category,
            severity="critique", source=SESSION_ID) or None
        self.history.record_incident("ouvert", incident)
        self._log("incident_ouvert", {
            "session_id": SESSION_ID, "incident_id": incident["id"], "type": cause["type"], "categorie": category,
            "source": source, "zone": state["zone"], "cle": cause["cle"], "message": cause["raison"],
            "severity": "critique", "wall_time": incident["debut"], "csv_event_id": incident["csv_event_id"],
        })
        return {**incident, "_vu": now}

    def _close_incident(self, incident, motif):
        ended = incident.pop("_vu")
        incident.update(fin=ended.isoformat(timespec="seconds"), motif=motif,
                        duree_s=round((ended - datetime.fromisoformat(incident["debut"])).total_seconds(), 1))
        self.history.record_incident("clos", {k: incident[k] for k in
                                              ("id", "fin", "duree_s", "motif", "raison_finale") if k in incident})
        self._log("incident_cloture", {
            "session_id": SESSION_ID, "incident_id": incident["id"], "type": incident["kind"],
            "categorie": incident["categorie"], "zone": incident["zone"], "duree_s": incident["duree_s"],
            "motif": motif,
        })

    def _close_all(self, now, motif):
        for key in list(self._open):
            incident = self._open.pop(key)
            incident["_vu"] = now  # la cause était encore active : l'incident dure jusqu'à l'arrêt
            self._close_incident(incident, motif)

    # ------------------------------------------------------------------
    def _log(self, event, data, user=SYSTEM):
        if self.audit is not None:
            self.audit.append(event, data, utilisateur=user["username"], role=user["role"])

    def parameters(self) -> list[dict]:
        values = self.config["parametres"]
        return [{"nom": name, "valeur": values[name], "min": low, "max": high, "unite": unit, "libelle": label}
                for name, (low, high, unit, label) in PARAM_BOUNDS.items()]

    def statistics(self, start: datetime, end: datetime, video_incidents=()) -> dict:
        """
        Par zone, sur [start, end] : temps passé dans chaque niveau, incidents de zone (gaz, machine,
        capteur) et incidents vidéo (EPI, proximité, caméra) des caméras associées à la zone,
        dépassements par gaz, et classement des zones les plus dangereuses.
        """
        zone_ids = [zone["id"] for zone in self.config["zones"]]
        names = {zone["id"]: zone["nom"] for zone in self.config["zones"]}
        times = self.history.time_in_levels(start, end, zone_ids)
        zones = {zone_id: {"zone": zone_id, "nom": names[zone_id], "temps": times[zone_id],
                           "incidents": dict.fromkeys(("gaz", "machine", "capteur", "epi", "proximite", "camera"), 0),
                           "depassements": {}, "total": 0} for zone_id in zone_ids}
        recent = []
        for incident in self.history.incidents(start):
            if incident.get("zone") not in zones or datetime.fromisoformat(incident["debut"]) > end:
                continue
            entry = zones[incident["zone"]]
            entry["incidents"][incident["kind"]] += 1
            if incident["kind"] == "gaz":
                entry["depassements"][incident["mesure"]] = entry["depassements"].get(incident["mesure"], 0) + 1
            recent.append(incident)
        profiles = {}
        for incident in video_incidents:
            try:
                moment = datetime.fromisoformat(incident.get("wall_time") or "")
            except ValueError:
                continue
            if moment.tzinfo is None:
                moment = moment.astimezone()
            if not start <= moment <= end:
                continue
            video = incident.get("video") or ""
            if video not in profiles:
                profiles[video] = load_camera_profile(video).get("zone_id")
            zone_id = profiles[video]
            if zone_id in zones:
                kind = incident.get("kind")
                category = "proximite" if kind == "proximite" else ("camera" if kind in CYBER_KINDS else "epi")
                zones[zone_id]["incidents"][category] += 1
        for entry in zones.values():
            entry["total"] = sum(entry["incidents"].values())
        ranking = sorted((z for z in zones.values() if z["zone"] in self._evaluated_zones()),
                         key=lambda z: (-z["temps"]["ROUGE"], -z["total"], -z["temps"]["ORANGE"]))
        return {"debut": start.isoformat(timespec="seconds"), "fin": end.isoformat(timespec="seconds"),
                "zones": zones, "classement": [z["zone"] for z in ranking], "incidents_recents": recent[:50]}

    def _evaluated_zones(self):
        return {zone["id"] for zone in self.config["zones"] if zone["methode"] != "hors_perimetre"}

    def _sensor_info(self, zone, sensor) -> dict:
        norm = self.config["normes"][sensor["mesure"]]
        return {"capteur": sensor["id"], "zone": zone["id"], "zone_nom": zone["nom"], "mesure": sensor["mesure"],
                "libelle": norm["libelle"], "unite": norm["unite"], "seuil_min": norm.get("min"),
                "seuil_max": norm["max"],
                "orange": orange_limits(norm, self.config["parametres"]["pourcentage_orange"])}

    def series(self, sensor_id: str, start: datetime, end: datetime) -> dict:
        """Courbe d'un capteur gaz, avec ses seuils (ligne rouge) et sa limite orange."""
        for zone in self.config["zones"]:
            for sensor in zone.get("capteurs", []):
                if sensor["id"] == sensor_id:
                    return {**self._sensor_info(zone, sensor), **self.history.series(sensor_id, start, end)}
        raise LookupError(f"Capteur inconnu : {sensor_id}")

    def all_series(self, start: datetime, end: datetime, max_points: int = 600) -> dict:
        """
        Courbes de tous les capteurs, dans l'ordre des zones et des capteurs de config/zones.json,
        réduites à `max_points` points environ chacune. L'axe du temps est commun : il commence à la
        première mesure disponible de la période (et non au début théorique de la période).
        """
        sensors = [(zone, sensor) for zone in self.config["zones"] for sensor in zone.get("capteurs", [])]
        data = self.history.series_many([sensor["id"] for _, sensor in sensors], start, end)
        curves, first = [], None
        for zone, sensor in sensors:
            entry = data[sensor["id"]]
            points, step = downsample(entry["points"], max_points)
            if points:
                moment = datetime.fromisoformat(points[0][0])
                first = moment if first is None else min(first, moment)
            curves.append({**self._sensor_info(zone, sensor), "pas": entry["pas"], "regroupement_s": step,
                           "points": points})
        return {"debut": (first or start).isoformat(timespec="seconds"), "fin": end.isoformat(timespec="seconds"),
                "debut_periode": start.isoformat(timespec="seconds"), "capteurs": curves}

    def open_incidents(self) -> list[dict]:
        return [{k: v for k, v in incident.items() if not k.startswith("_")} for incident in self._open.values()]

    def status(self) -> dict:
        with self._lock:
            provider = self.provider.status()
            zones = []
            for zone in self.config["zones"]:
                state = dict(self.states.get(zone["id"], {}))
                state["plan"] = zone.get("plan")
                state["liste_capteurs"] = [{"id": c["id"], "mesure": c["mesure"],
                                            "libelle": self.config["normes"][c["mesure"]]["libelle"]}
                                           for c in zone.get("capteurs", [])]
                state["liste_machines"] = [{"id": m["id"], "type": m.get("type")} for m in zone.get("machines", [])]
                if zone["methode"] in ("gaz", "machine"):
                    state["scenario"] = provider["scenarios"][zone["id"]]
                    state["scenarios_possibles"] = [{"id": s, "libelle": SCENARIO_LABELS[s]}
                                                    for s in self.provider.scenarios_for(zone["id"])]
                zones.append(state)
            return {
                "actif": provider["actif"],
                "arret": self.stopped,
                "demarrage": self.started,
                "source": provider,
                "zones": zones,
                "incidents_ouverts": self.open_incidents(),
                "parametres": self.parameters(),
                "horodatage": self.clock().isoformat(timespec="seconds"),
            }

