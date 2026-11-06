"""
app/zones/settings.py

Chargement et validation de config/zones.json (zones dangereuses du site).

Deux niveaux de modification :
- le FICHIER (personne ayant accès au poste) : tout, y compris les seuils
  normatifs S-HSE-23, les capteurs et les zones ;
- l'INTERFACE (administrateur) : uniquement les réglages de `parametres`,
  chacun dans des bornes sûres fixées ICI, dans le code, et non dans le
  fichier. Une valeur hors bornes est refusée : un compte administrateur
  compromis ne peut ni relever un seuil normatif ni désactiver une alarme
  (pourcentage orange à 100 %, hystérésis d'une heure, etc.).
"""

import copy
import json
import os
import tempfile
from pathlib import Path

from app.config import ZONES_PATH

METHODS = {"gaz", "machine", "hors_perimetre"}

# Réglages modifiables depuis l'interface : (minimum, maximum, unité, libellé)
PARAM_BOUNDS = {
    "pourcentage_orange": (50, 90, "%", "Seuil orange (part du seuil normatif)"),
    "fenetre_s": (20, 300, "s", "Fenêtre glissante de stabilité"),
    "horizon_derive_s": (60, 1800, "s", "Horizon de dérive (seuil atteint dans moins de)"),
    "facteur_variabilite": (2, 10, "× bruit", "Variabilité anormale (écart-type au-delà de)"),
    "hysteresis_s": (5, 120, "s", "Délai avant de redescendre d'un niveau"),
    "delai_absence_s": (1, 10, "s", "Donnée absente ou en retard au-delà de"),
    "duree_figee_s": (10, 120, "s", "Valeur figée au-delà de"),
    "quarantaine_s": (30, 600, "s", "Capteur non fiable pendant au moins"),
    "confirmation_zero_s": (1, 10, "s", "Confirmation d'un gaz à seuil 0 ppm"),
    "presence_maintien_s": (1, 30, "s", "Présence maintenue après la dernière détection"),
}


class ConfigError(ValueError):
    """Configuration invalide ou modification refusée (message lisible)."""


def load_zones_config(path=ZONES_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate(config)
    return config


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate(config: dict):
    """Vérifie la cohérence du fichier ; lève ConfigError avec la première erreur trouvée."""
    norms, measures = config.get("normes", {}), config.get("mesures", {})
    for key, norm in norms.items():
        if key.startswith("_"):
            continue
        if not _number(norm.get("max")):
            raise ConfigError(f"normes.{key} : seuil 'max' manquant.")
        if "min" in norm and not _number(norm["min"]):
            raise ConfigError(f"normes.{key} : seuil 'min' invalide.")
        if key not in measures:
            raise ConfigError(f"mesures.{key} manquant (plage, bruit, vitesse).")
    for key, spec in measures.items():
        if key.startswith("_"):
            continue
        low, high = spec.get("plage", [None, None])
        if not (_number(low) and _number(high) and low < high):
            raise ConfigError(f"mesures.{key} : plage invalide.")
        for field in ("resolution", "bruit", "vitesse_max_par_s"):
            if not (_number(spec.get(field)) and spec[field] > 0):
                raise ConfigError(f"mesures.{key} : '{field}' doit être un nombre positif.")
    params = config.get("parametres", {})
    for name, (low, high, _, _) in PARAM_BOUNDS.items():
        if not (_number(params.get(name)) and low <= params[name] <= high):
            raise ConfigError(f"parametres.{name} : doit être compris entre {low} et {high}.")
    seen, sensors = set(), set()
    for zone in config.get("zones", []):
        zone_id = zone.get("id")
        if not zone_id or zone_id in seen:
            raise ConfigError(f"Zone sans identifiant ou en double : {zone_id!r}.")
        seen.add(zone_id)
        if zone.get("methode") not in METHODS:
            raise ConfigError(f"Zone {zone_id} : méthode inconnue ({zone.get('methode')}).")
        for sensor in zone.get("capteurs", []):
            if sensor.get("mesure") not in norms:
                raise ConfigError(f"Capteur {sensor.get('id')} : mesure sans seuil S-HSE-23 ({sensor.get('mesure')}).")
            if sensor.get("id") in sensors:
                raise ConfigError(f"Capteur en double : {sensor.get('id')}.")
            sensors.add(sensor.get("id"))
        plan = zone.get("plan")
        if plan is not None:
            points = plan.get("points", [])
            if len(points) < 3 or not all(len(pt) == 2 and all(_number(v) and 0 <= v <= 1000 for v in pt)
                                          for pt in points + [plan.get("etiquette", [0, 0])]):
                raise ConfigError(f"Zone {zone_id} : plan invalide (au moins 3 points dans le repère 1000 × 620).")
        if zone["methode"] == "gaz" and not zone.get("capteurs"):
            raise ConfigError(f"Zone {zone_id} : une zone gaz doit avoir au moins un capteur.")
        if zone["methode"] == "machine" and not zone.get("machines"):
            raise ConfigError(f"Zone {zone_id} : une zone machine doit avoir au moins une machine.")


def check_parameter_changes(config: dict, changes) -> dict:
    """
    Valide une modification de réglages venue de l'interface.

    Retourne {nom: (ancienne, nouvelle)} pour les valeurs réellement changées ;
    lève ConfigError si un champ n'est pas modifiable ou sort de ses bornes.
    """
    if not isinstance(changes, dict) or not changes:
        raise ConfigError("Aucun réglage à modifier.")
    current = config["parametres"]
    diff = {}
    for name, value in changes.items():
        if name not in PARAM_BOUNDS:
            raise ConfigError(f"« {name} » n'est pas modifiable depuis l'interface "
                              "(seuils S-HSE-23, capteurs et zones : fichier uniquement).")
        low, high, unit, _ = PARAM_BOUNDS[name]
        if not _number(value) or value != value:  # refuse aussi NaN
            raise ConfigError(f"{name} : valeur numérique attendue.")
        if not low <= value <= high:
            raise ConfigError(f"{name} = {value} refusé : bornes sûres {low} à {high} {unit}.")
        if value != current.get(name):
            diff[name] = (current.get(name), value)
    return diff


def apply_parameter_changes(changes, path=ZONES_PATH) -> tuple[dict, dict]:
    """Valide puis écrit les réglages dans le fichier (écriture atomique). Retourne (config, diff)."""
    path = Path(path)
    config = load_zones_config(path)
    diff = check_parameter_changes(config, changes)
    if not diff:
        return config, diff
    updated = copy.deepcopy(config)
    for name, (_, new) in diff.items():
        updated["parametres"][name] = new
    validate(updated)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(updated, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return updated, diff


def zone_by_id(config: dict) -> dict:
    return {zone["id"]: zone for zone in config["zones"]}


def sensor_spec(config: dict, sensor: dict) -> dict:
    """Caractéristiques d'un capteur : celles de sa mesure, avec ses surcharges éventuelles (ex. 'normale')."""
    spec = dict(config["mesures"][sensor["mesure"]])
    spec.update({k: v for k, v in sensor.items() if k not in ("id", "mesure")})
    return spec
