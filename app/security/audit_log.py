"""
app/security/audit_log.py

Journal d'audit chaîné et signé : data/outputs/audit_log.jsonl.

Chaque ligne (incident, changement d'état machine, ajout de vidéo,
démarrage / arrêt de session, connexion, gestion des comptes…) contient :
- seq       : numéro d'ordre (1, 2, 3, …) ;
- utilisateur, role : auteur de l'action (« systeme » pour les événements
              produits par l'analyse elle-même) ;
- prev_hash : empreinte SHA-256 de l'entrée précédente ;
- hash      : empreinte SHA-256 de l'entrée elle-même ;
- hmac      : signature HMAC-SHA256 de l'entrée avec une clé secrète
              (config/secrets/audit.key, hors de data/ et exclue de git).

Modifier ou supprimer une ligne casse la chaîne ; recalculer toute la
chaîne n'est pas possible sans la clé. audit_head.json conserve le nombre
d'entrées et la dernière empreinte (lui aussi signé) pour détecter la
suppression des DERNIÈRES lignes.

Vérification : python tools/verify_audit.py
"""

import hashlib
import hmac
import json
import os
import secrets
import threading
from datetime import datetime
from pathlib import Path

from app.config import CONFIG_DIR, ROOT

AUDIT_PATH = ROOT / "data" / "outputs" / "audit_log.jsonl"
KEY_PATH = CONFIG_DIR / "secrets" / "audit.key"
GENESIS = "0" * 64


def _canonical(entry: dict) -> bytes:
    body = {k: v for k, v in entry.items() if k not in ("hash", "hmac")}
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def head_path_for(path) -> Path:
    return Path(path).with_name("audit_head.json")


def load_key(key_path=KEY_PATH, create: bool = False) -> bytes | None:
    key_path = Path(key_path)
    if key_path.is_file():
        return bytes.fromhex(key_path.read_text(encoding="ascii").strip())
    if not create:
        return None
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    key_path.write_text(key.hex(), encoding="ascii")
    return key


def _sign(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _head_payload(head: dict) -> bytes:
    return json.dumps({k: head[k] for k in ("count", "last_hash")}, sort_keys=True).encode("utf-8")


def verify(path=AUDIT_PATH, key_path=KEY_PATH) -> dict:
    """
    Vérifie la chaîne, les signatures et l'en-tête.
    Retourne {ok, count, errors, warnings}.
    """
    path = Path(path)
    errors, warnings = [], []
    key = load_key(key_path)
    if key is None:
        warnings.append(f"Clé {Path(key_path).name} absente : signatures HMAC non vérifiées.")
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    previous = GENESIS
    count = 0
    for number, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"Ligne {number} : illisible (JSON invalide).")
            previous = None
            continue
        count += 1
        payload = _canonical(entry)
        if entry.get("seq") != number:
            errors.append(f"Ligne {number} : numéro {entry.get('seq')} au lieu de {number} (ligne supprimée ou insérée avant).")
        if previous is not None and entry.get("prev_hash") != previous:
            errors.append(f"Ligne {number} : chaînage rompu (l'entrée précédente a été modifiée ou supprimée).")
        if hashlib.sha256(payload).hexdigest() != entry.get("hash"):
            errors.append(f"Ligne {number} : contenu modifié (empreinte SHA-256 différente).")
        if key is not None and not hmac.compare_digest(_sign(key, payload), str(entry.get("hmac", ""))):
            errors.append(f"Ligne {number} : signature HMAC invalide (entrée réécrite sans la clé).")
        previous = entry.get("hash")

    head_path = head_path_for(path)
    if head_path.is_file():
        head = json.loads(head_path.read_text(encoding="utf-8"))
        if key is not None and not hmac.compare_digest(_sign(key, _head_payload(head)), str(head.get("hmac", ""))):
            errors.append("audit_head.json : signature HMAC invalide (en-tête modifié).")
        if head.get("count", 0) > len(lines):
            errors.append(f"{head['count'] - len(lines)} dernière(s) ligne(s) supprimée(s) "
                          f"(en-tête : {head['count']} entrées, fichier : {len(lines)}).")
        elif head.get("count") != len(lines) or (lines and head.get("last_hash") != previous):
            errors.append("audit_head.json ne correspond pas à la fin du journal.")
    elif lines:
        warnings.append("audit_head.json absent : la suppression des dernières lignes n'est pas vérifiable.")
    return {"ok": not errors, "count": count, "errors": errors, "warnings": warnings}


def read_entries(path=AUDIT_PATH) -> list[dict]:
    """Entrées lisibles du journal (les lignes illisibles sont ignorées ; verify() les signale)."""
    path = Path(path)
    entries = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


class AuditLog:
    def __init__(self, path=AUDIT_PATH, key_path=KEY_PATH):
        self.path = Path(path)
        self.key_path = Path(key_path)
        self.head_path = head_path_for(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._key = load_key(key_path, create=True)
        self._lock = threading.Lock()
        self._seq, self._last_hash = 0, GENESIS
        if self.path.is_file():
            lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                try:
                    last = json.loads(lines[-1])
                    self._seq, self._last_hash = len(lines), last["hash"]
                except (json.JSONDecodeError, KeyError):
                    self._seq = len(lines)
        # Journal altéré depuis le dernier lancement : on le consigne (sans le réparer)
        check = verify(self.path, key_path)
        if not check["ok"]:
            print(f"[AUDIT] Journal d'audit altéré : {check['errors'][0]} — voir tools/verify_audit.py")
            self.append("alerte_integrite_journal", {"erreurs": check["errors"][:20]})

    def append(self, event: str, data: dict | None = None, utilisateur: str = "systeme",
               role: str = "systeme") -> dict:
        with self._lock:
            entry = {
                "seq": self._seq + 1,
                "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "utilisateur": utilisateur,
                "role": role,
                "event": event,
                "data": data or {},
                "prev_hash": self._last_hash,
            }
            payload = _canonical(entry)
            entry["hash"] = hashlib.sha256(payload).hexdigest()
            entry["hmac"] = _sign(self._key, payload)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._seq, self._last_hash = entry["seq"], entry["hash"]
            head = {"count": self._seq, "last_hash": self._last_hash,
                    "updated_at": entry["ts"]}
            head["hmac"] = _sign(self._key, _head_payload(head))
            temporary = self.head_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(head, indent=1), encoding="utf-8")
            temporary.replace(self.head_path)
            return entry
