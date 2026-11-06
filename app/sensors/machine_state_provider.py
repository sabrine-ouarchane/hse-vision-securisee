"""
app/sensors/machine_state_provider.py

Source de l'état marche / arrêt des équipements mobiles pour la détection des
situations homme-machine.

Le POC n'a accès à aucun signal automate (PLC, OSIsoft PI) : l'état n'est
donc JAMAIS deviné à partir de l'image. Il est déclaré dans
config/machine_states.json, soit depuis l'interface de surveillance, soit
par une passerelle externe qui écrit ce fichier. Le fichier est relu dès
qu'il change, ce qui permet de modifier l'état pendant une analyse en cours.

Un état absent, invalide ou trop ancien (max_age_s) vaut "inconnu".

config/machine_states.json est un état d'exécution, non versionné. S'il
n'existe pas, il est créé depuis config/machine_states.example.json (tous
les engins « inconnu », politique « prudent »). Un fichier existant n'est
jamais remplacé, même illisible : on garde alors la dernière version valide.
"""

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from app.config import MACHINE_STATES_EXAMPLE_PATH, MACHINE_STATES_PATH

VALID_STATES = {"marche", "arret", "inconnu"}
VALID_POLICIES = {"prudent", "surveillance"}


def _write_atomic(path: Path, data: dict):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class MachineStateProvider:
    def __init__(self, path=MACHINE_STATES_PATH, example=None):
        """
        Args:
            path: fichier d'état.
            example: état par défaut copié si `path` n'existe pas (par défaut :
                machine_states.example.json pour le fichier d'état de l'application,
                aucune copie pour un autre chemin).
        """
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime = None
        self._data = {"unknown_policy": "prudent", "max_age_s": None, "machines": {}}
        if example is None and self.path == Path(MACHINE_STATES_PATH):
            example = MACHINE_STATES_EXAMPLE_PATH
        self.initialized_from = self._initialize(Path(example)) if example is not None else None
        self._reload()

    def _initialize(self, example: Path):
        """Crée le fichier d'état depuis l'exemple s'il n'existe pas ; renvoie le nom de l'exemple, sinon None."""
        if self.path.exists() or not example.is_file():
            return None
        _write_atomic(self.path, json.loads(example.read_text(encoding="utf-8")))
        print(f"[INFO] {self.path.name} absent : créé depuis {example.name} (tous les engins « inconnu »).")
        return example.name

    def _reload(self):
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime == self._mtime:
            return
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return  # on garde la dernière configuration valide
        self._data = data
        self._mtime = mtime

    @property
    def unknown_policy(self) -> str:
        with self._lock:
            self._reload()
            policy = self._data.get("unknown_policy", "prudent")
        return policy if policy in VALID_POLICIES else "prudent"

    def get(self, machine_id: str) -> dict:
        with self._lock:
            self._reload()
            entry = dict(self._data.get("machines", {}).get(machine_id) or {})
            max_age = self._data.get("max_age_s")
        state = entry.get("state")
        if state not in VALID_STATES:
            state = "inconnu"
        if state != "inconnu" and max_age and entry.get("updated_at"):
            try:
                age = (datetime.now().astimezone() - datetime.fromisoformat(entry["updated_at"])).total_seconds()
                if age > max_age:
                    state = "inconnu"
                    entry["source"] = f"{entry.get('source', '?')} (expiré)"
            except ValueError:
                state = "inconnu"
        return {
            "machine_id": machine_id,
            "state": state,
            "source": entry.get("source", "non renseigné"),
            "updated_at": entry.get("updated_at"),
        }

    def set_state(self, machine_id: str, state: str, source: str = "interface", unknown_policy: str | None = None):
        if not isinstance(machine_id, str) or not machine_id or len(machine_id) > 64:
            raise ValueError("Identifiant d'équipement invalide.")
        if state not in VALID_STATES:
            raise ValueError(f"État invalide : {state}")
        if unknown_policy is not None and unknown_policy not in VALID_POLICIES:
            raise ValueError(f"Politique invalide : {unknown_policy}")
        with self._lock:
            self._reload()
            data = json.loads(json.dumps(self._data))
            data.setdefault("machines", {})[machine_id] = {
                "state": state,
                "source": source,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            if unknown_policy is not None:
                data["unknown_policy"] = unknown_policy
            _write_atomic(self.path, data)
            self._data = data
            self._mtime = self.path.stat().st_mtime
        return self.get(machine_id)
