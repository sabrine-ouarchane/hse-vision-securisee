"""
app/zones/history.py

Historique des zones dangereuses, pour les statistiques de la page /zones :

- niveaux_AAAAMMJJ.jsonl : un changement de niveau par ligne (zone, niveau,
  raison) — le temps passé en vert / orange / rouge s'en déduit ;
- mesures_AAAAMMJJ.jsonl : agrégat par minute de chaque capteur numérique
  (min, moyenne, max) pour les courbes sur un jour ou une semaine ;
- incidents.jsonl : ouverture et clôture des incidents de zone (jamais
  purgé : le contrôle de cohérence le compare au journal d'audit).

La dernière heure de mesures est gardée à la seconde, en mémoire.
Les fichiers datés sont supprimés après `retention_days` jours
(≈ 1 à 2 Mo par jour avec les 7 zones du POC).
"""

import json
import math
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

LEVELS = ("GRIS", "VERT", "ORANGE", "ROUGE")
INCIDENTS_FILE = "incidents.jsonl"


def _day(moment: datetime) -> str:
    return moment.strftime("%Y%m%d")


class ZoneHistory:
    def __init__(self, directory, retention_days: int = 8, memory_s: int = 3600):
        self.directory = Path(directory)
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._last_level = {}
        self._recent = defaultdict(lambda: deque(maxlen=memory_s))
        self._minute = {}  # capteur -> [minute, min, somme, max, n]
        self._last_purge = None
        self._cache = {}  # jour -> ((taille, mtime), agrégats par minute lus)

    @property
    def incidents_path(self) -> Path:
        return self.directory / INCIDENTS_FILE

    def _append(self, name, rows):
        if not rows:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / name).open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read(self, name):
        path = self.directory / name
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _days(self, start, end):
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= end:
            yield _day(day)
            day += timedelta(days=1)

    # ------------------------------------------------------------------
    # Écriture
    def record_levels(self, now: datetime, states: dict):
        with self._lock:
            rows = []
            for zone_id, state in states.items():
                if self._last_level.get(zone_id) != state["niveau"]:
                    self._last_level[zone_id] = state["niveau"]
                    rows.append({"t": now.isoformat(timespec="seconds"), "zone": zone_id,
                                 "niveau": state["niveau"], "raison": state["raison"]})
            self._append(f"niveaux_{_day(now)}.jsonl", rows)
            self._purge(now)

    def record_samples(self, readings):
        with self._lock:
            rows = []
            for reading in readings:
                value = reading.value
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                self._recent[reading.sensor_id].append((reading.timestamp.isoformat(timespec="seconds"), value))
                minute = reading.timestamp.replace(second=0, microsecond=0)
                current = self._minute.get(reading.sensor_id)
                if current is not None and current[0] != minute:
                    rows.append(self._minute_row(reading.sensor_id, current))
                    current = None
                if current is None:
                    self._minute[reading.sensor_id] = [minute, value, value, value, 1]
                else:
                    current[1] = min(current[1], value)
                    current[2] += value
                    current[3] = max(current[3], value)
                    current[4] += 1
            by_day = defaultdict(list)
            for row in rows:
                by_day[row["t"][:10].replace("-", "")].append(row)
            for day, day_rows in by_day.items():
                self._append(f"mesures_{day}.jsonl", day_rows)

    @staticmethod
    def _minute_row(sensor_id, current):
        minute, low, total, high, count = current
        return {"t": minute.isoformat(timespec="seconds"), "capteur": sensor_id, "min": low,
                "moy": round(total / count, 4), "max": high, "n": count}

    def record_incident(self, event: str, incident: dict):
        with self._lock:
            self._append(INCIDENTS_FILE, [{"evenement": event, **incident}])

    def _purge(self, now):
        if self._last_purge == _day(now) or not self.directory.is_dir():
            return
        self._last_purge = _day(now)
        limit = _day(now - timedelta(days=self.retention_days))
        for path in self.directory.glob("*_????????.jsonl"):
            if path.stem.rsplit("_", 1)[-1] < limit:
                path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Lecture
    def time_in_levels(self, start: datetime, end: datetime, zone_ids) -> dict:
        """Secondes passées dans chaque niveau, par zone, entre start et end (avant le 1er relevé : non compté)."""
        with self._lock:
            rows = [r for day in self._days(start - timedelta(days=self.retention_days), end)
                    for r in self._read(f"niveaux_{day}.jsonl")]
        totals = {zone: dict.fromkeys(LEVELS, 0.0) for zone in zone_ids}
        by_zone = defaultdict(list)
        for row in rows:
            try:
                by_zone[row["zone"]].append((datetime.fromisoformat(row["t"]), row["niveau"]))
            except (KeyError, ValueError):
                continue
        for zone, changes in by_zone.items():
            if zone not in totals:
                continue
            changes.sort()
            for index, (moment, level) in enumerate(changes):
                following = changes[index + 1][0] if index + 1 < len(changes) else end
                begin, finish = max(moment, start), min(following, end)
                if finish > begin and level in totals[zone]:
                    totals[zone][level] += (finish - begin).total_seconds()
        return totals

    def incidents(self, start: datetime | None = None) -> list[dict]:
        """Incidents de zone (ouverture + clôture fusionnées), du plus récent au plus ancien."""
        with self._lock:
            rows = self._read(INCIDENTS_FILE)
        merged = {}
        for row in rows:
            entry = merged.setdefault(row.get("id"), {})
            entry.update({k: v for k, v in row.items() if k != "evenement"})
        out = []
        for incident in merged.values():
            try:
                begin = datetime.fromisoformat(incident["debut"])
            except (KeyError, TypeError, ValueError):
                continue
            ended = incident.get("fin")
            if start is not None and ended and datetime.fromisoformat(ended) < start:
                continue
            out.append(incident)
        return sorted(out, key=lambda i: i["debut"], reverse=True)

    def _measures(self, day):
        """Agrégats par minute d'un jour, relus seulement quand le fichier a changé (rafraîchissement à 1 Hz)."""
        path = self.directory / f"mesures_{day}.jsonl"
        try:
            stat = path.stat()
        except OSError:
            self._cache.pop(day, None)
            return []
        key = (stat.st_size, stat.st_mtime_ns)
        cached = self._cache.get(day)
        if cached is None or cached[0] != key:
            rows = []
            for r in self._read(path.name):
                try:
                    rows.append((datetime.fromisoformat(r["t"]), r["capteur"], r["min"], r["moy"], r["max"]))
                except (KeyError, TypeError, ValueError):
                    continue
            cached = self._cache[day] = (key, rows)
        return cached[1]

    def series(self, sensor_id: str, start: datetime, end: datetime) -> dict:
        """Mesures d'un capteur : à la seconde sur une heure au plus, sinon agrégats par minute."""
        return self.series_many([sensor_id], start, end)[sensor_id]

    def series_many(self, sensor_ids, start: datetime, end: datetime) -> dict:
        """Comme series(), pour plusieurs capteurs en une seule lecture des fichiers."""
        with self._lock:
            if (end - start).total_seconds() <= 3600:
                return {sensor_id: {"pas": "seconde",
                                    "points": [[t, v] for t, v in self._recent.get(sensor_id, ())
                                               if start <= datetime.fromisoformat(t) <= end]}
                        for sensor_id in sensor_ids}
            points = {sensor_id: [] for sensor_id in sensor_ids}
            for day in self._days(start, end):
                for moment, sensor_id, low, mean, high in self._measures(day):
                    if sensor_id in points and start <= moment <= end:
                        points[sensor_id].append([moment.isoformat(timespec="seconds"), low, mean, high])
            for sensor_id in sensor_ids:
                current = self._minute.get(sensor_id)
                if current is not None and start <= current[0] <= end:
                    row = self._minute_row(sensor_id, current)
                    points[sensor_id].append([row["t"], row["min"], row["moy"], row["max"]])
        return {sensor_id: {"pas": "minute", "points": rows} for sensor_id, rows in points.items()}


def downsample(points, limit: int):
    """
    Réduit une courbe à `limit` points environ, par tranches de temps égales : minimum des minimums,
    moyenne, maximum des maximums (un pic bref reste visible). Renvoie (points, durée d'une tranche en s,
    None si la courbe est gardée telle quelle).
    """
    if len(points) <= limit:
        return points, None
    origin = datetime.fromisoformat(points[0][0])
    span = (datetime.fromisoformat(points[-1][0]) - origin).total_seconds()
    step = max(1, math.ceil(span / limit))
    buckets = {}
    for point in points:
        index = int((datetime.fromisoformat(point[0]) - origin).total_seconds() // step)
        low, mean, high = (point[1],) * 3 if len(point) == 2 else point[1:4]
        bucket = buckets.get(index)
        if bucket is None:
            buckets[index] = [point[0], low, mean, high, 1]
        else:
            bucket[1] = min(bucket[1], low)
            bucket[2] += mean
            bucket[3] = max(bucket[3], high)
            bucket[4] += 1
    return [[t, low, round(total / n, 4), high] for t, low, total, high, n in buckets.values()], step
