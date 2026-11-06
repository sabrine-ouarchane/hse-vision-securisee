"""
app/config.py

Chargement centralisé des fichiers de configuration du POC (règles de
détection, profils caméra). Les chemins relatifs sont résolus depuis la
racine du projet, pour que les scripts fonctionnent quel que soit le
répertoire courant.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DETECTION_RULES_PATH = CONFIG_DIR / "detection_rules.json"
CAMERA_PROFILES_PATH = CONFIG_DIR / "camera_profiles.json"
MACHINE_STATES_PATH = CONFIG_DIR / "machine_states.json"  # état d'exécution, non versionné
MACHINE_STATES_EXAMPLE_PATH = CONFIG_DIR / "machine_states.example.json"  # état par défaut, versionné
CAMERA_ZONES_PATH = CONFIG_DIR / "camera_zones.json"
ZONES_PATH = CONFIG_DIR / "zones.json"


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_json(path) -> dict:
    with resolve(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_detection_rules(path=DETECTION_RULES_PATH) -> dict:
    return load_json(path)


def load_camera_profile(source_name: str | None, path=CAMERA_PROFILES_PATH) -> dict:
    """Fusionne le profil 'default' avec le profil propre à la source (nom de fichier ou caméra)."""
    profiles = load_json(path)
    profile = dict(profiles.get("default", {}))
    if source_name:
        sources = profiles.get("sources", {})
        name = Path(str(source_name)).name
        if name not in sources and "__" in name:
            # Vidéo produite par tools/tamper_simulator.py (<vidéo>__<attaque>.mp4) :
            # même caméra que la vidéo d'origine.
            stem, attack = Path(name).stem.split("__", 1)
            original = sources.get(stem + Path(name).suffix)
            if original is not None:
                profile.update(original)
                profile["name"] = f"{original.get('name', stem)} · attaque simulée ({attack})"
        profile.update(sources.get(name, {}))
    profile.setdefault("name", str(source_name or "Caméra"))
    return profile
