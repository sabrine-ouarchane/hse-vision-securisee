"""
app/sensors/provider.py

Interface unique entre les sources de mesures et le reste du système.

Le classement des zones (app/zones/risk.py), les contrôles de plausibilité
(app/sensors/plausibility.py), le moniteur et la page /zones ne connaissent
QUE cette interface. Le POC utilise un simulateur
(app/sensors/simulated_provider.py) ; une source réelle le remplacera sans
autre changement.

Exemple de source réelle (OSIsoft PI), à écrire le jour où les capteurs
existent :

    class PIProvider(SensorProvider):
        def __init__(self, config, client, tags):   # tags : {sensor_id: "BI.SECH.SO2.PV", ...}
            ...
        def sensors(self):
            return [SensorInfo(...) for chaque tag]
        def poll(self, now):
            # valeurs enregistrées depuis le dernier appel (« recorded values »),
            # avec l'horodatage de la SOURCE, pas celui de réception
            return [Reading(timestamp=v.timestamp, zone=..., sensor_id=..., measure=...,
                            value=v.value, unit=...) for v in client.recorded_values(...)]
        def status(self):
            return {"actif": True, "source": "OSIsoft PI", ...}

Règles du contrat :
- poll() ne renvoie que des mesures nouvelles, dans l'ordre chronologique ;
- une mesure absente n'est PAS remplacée par une valeur par défaut : c'est
  l'absence de mesure qui est détectée (plausibilité « donnée absente ») ;
- l'état d'une machine passe par la même interface (measure « etat », valeur
  « marche », « arret » ou « inconnu »), comme un « digital state » PI.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime

MACHINE_MEASURE = "etat"
MACHINE_STATES = ("marche", "arret", "inconnu")


@dataclass(frozen=True)
class Reading:
    """Une mesure, telle que l'enverrait un vrai capteur."""
    timestamp: datetime          # horodatage de la source (avec fuseau)
    zone: str
    sensor_id: str               # ex. "sechage-so2" ou "parcs-gerbeur_1"
    measure: str                 # ex. "SO2_ppm", ou "etat" pour une machine
    value: float | str | None
    unit: str

    def as_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat(timespec="seconds")
        return data


@dataclass(frozen=True)
class SensorInfo:
    """Inventaire : ce que la source annonce pour chaque capteur."""
    sensor_id: str
    zone: str
    measure: str
    unit: str
    period_s: float
    machine_id: str | None = None   # capteurs d'état machine uniquement


class SensorProvider(ABC):
    @abstractmethod
    def sensors(self) -> list[SensorInfo]:
        """Capteurs connus de la source."""

    @abstractmethod
    def poll(self, now: datetime) -> list[Reading]:
        """Mesures arrivées depuis le dernier appel."""

    @abstractmethod
    def status(self) -> dict:
        """État de la source : au minimum {"actif": bool, "source": str}."""
