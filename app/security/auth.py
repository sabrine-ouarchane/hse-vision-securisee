"""
app/security/auth.py

Comptes locaux, sessions et rôles du dashboard.

- Comptes dans config/users.json (exclu de git) ; mots de passe hachés avec
  hashlib.scrypt et un sel aléatoire, jamais stockés en clair.
- Aucun compte par défaut : le premier admin est créé avec
  tools/manage_users.py.
- Sessions en mémoire : identifiant aléatoire (secrets), expiration après
  30 min d'inactivité et 12 h au plus.
- Compte bloqué 15 min après 5 échecs de connexion (compteur conservé dans
  users.json : un redémarrage ne le remet pas à zéro).
- Rôles hiérarchiques : operateur < responsable_hse < admin. Le contrôle est
  fait côté serveur ; masquer un bouton dans la page n'est qu'un confort.

Chaque action est consignée dans le journal d'audit avec son auteur.
"""

import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

from app.config import CONFIG_DIR

USERS_PATH = CONFIG_DIR / "users.json"
ROLES = ("operateur", "responsable_hse", "admin")
ROLE_LABELS = {"operateur": "Opérateur", "responsable_hse": "Responsable HSE", "admin": "Administrateur"}
MIN_PASSWORD_LENGTH = 12
MAX_FAILURES = 5
LOCK_S = 15 * 60
IDLE_S = 30 * 60
MAX_SESSION_S = 12 * 3600
USERNAME_PATTERN = re.compile(r"^[a-z0-9._-]{3,32}$")
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


class AuthError(Exception):
    """Refus d'authentification ou d'opération sur un compte (message affichable)."""


def has_role(role: str | None, required: str) -> bool:
    return role in ROLES and ROLES.index(role) >= ROLES.index(required)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, **SCRYPT)
    return "scrypt${n}${r}${p}${salt}${digest}".format(
        **SCRYPT, salt=base64.b64encode(salt).decode(), digest=base64.b64encode(digest).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        computed = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt), dklen=32,
                                  n=int(n), r=int(r), p=int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(computed, base64.b64decode(digest))


# Haché factice : un identifiant inconnu coûte le même temps qu'un mauvais mot de passe
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def password_fingerprint(stored_hash: str) -> str:
    """Empreinte du haché (et non du mot de passe) consignée dans l'audit : détecte une modification directe de users.json."""
    return hashlib.sha256(stored_hash.encode("utf-8")).hexdigest()[:16]


def check_username(username) -> str:
    if not isinstance(username, str) or not USERNAME_PATTERN.match(username):
        raise AuthError("Identifiant invalide : 3 à 32 caractères parmi a-z, 0-9, « . », « _ », « - ».")
    return username


def check_password(password) -> str:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Mot de passe trop court : {MIN_PASSWORD_LENGTH} caractères minimum.")
    if len(password) > 256:
        raise AuthError("Mot de passe trop long.")
    return password


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class UserStore:
    """config/users.json ; écriture atomique, relu si le fichier change."""

    def __init__(self, path=USERS_PATH):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._mtime = None
        self._users = {}

    def _load(self):
        mtime = self.path.stat().st_mtime_ns if self.path.is_file() else None
        if mtime != self._mtime:
            self._users = json.loads(self.path.read_text(encoding="utf-8")).get("users", {}) if mtime else {}
            self._mtime = mtime
        return self._users

    def _save(self, users):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"users": users}, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(self.path)
        self._users, self._mtime = users, self.path.stat().st_mtime_ns

    def all(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._load()))

    def get(self, username):
        with self._lock:
            user = self._load().get(username)
            return dict(user) if user else None

    def update(self, username, **fields):
        with self._lock:
            users = json.loads(json.dumps(self._load()))
            if username not in users:
                raise AuthError("Compte inconnu.")
            users[username].update(fields)
            self._save(users)
            return users[username]

    def add(self, username, record):
        with self._lock:
            users = json.loads(json.dumps(self._load()))
            if username in users:
                raise AuthError("Ce compte existe déjà (les comptes ne sont jamais supprimés : le réactiver).")
            users[username] = record
            self._save(users)

    def active_admins(self) -> list[str]:
        return [name for name, user in self.all().items() if user.get("role") == "admin" and user.get("active")]


class AuthService:
    """Comptes + sessions + journalisation des actions."""

    def __init__(self, users_path=USERS_PATH, audit=None, clock=time.time):
        self.users = UserStore(users_path)
        self.audit = audit
        self.clock = clock
        self._sessions = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _log(self, event, data, actor=None):
        if self.audit is not None:
            actor = actor or {"username": "systeme", "role": "systeme"}
            self.audit.append(event, data, utilisateur=actor["username"], role=actor["role"])

    # ------------------------------------------------------------------
    def login(self, username, password) -> tuple[str, dict]:
        """Retourne (identifiant de session, utilisateur) ; lève AuthError sinon (message volontairement générique)."""
        attempted = str(username)[:64] if username is not None else ""
        user = self.users.get(username) if isinstance(username, str) else None
        refused = AuthError("Identifiant ou mot de passe incorrect.")
        if user is None:
            verify_password(str(password), _DUMMY_HASH)
            self._log("connexion_echouee", {"identifiant_tente": attempted, "motif": "identifiant inconnu"},
                      {"username": attempted or "?", "role": "inconnu"})
            raise refused
        actor = {"username": username, "role": user["role"]}
        now = self.clock()
        if user.get("locked_until") and now < user["locked_until"]:
            verify_password(str(password), _DUMMY_HASH)
            self._log("connexion_echouee", {"identifiant_tente": attempted, "motif": "compte bloqué"}, actor)
            raise AuthError(f"Compte temporairement bloqué après {MAX_FAILURES} échecs. Réessayer plus tard "
                            f"ou demander à un administrateur de le débloquer.")
        if not verify_password(str(password), user["password"]) or not user.get("active"):
            failures = user.get("failed", 0) + 1
            fields = {"failed": failures}
            motif = "compte désactivé" if user.get("active") is False else "mot de passe incorrect"
            self._log("connexion_echouee", {"identifiant_tente": attempted, "motif": motif, "echecs": failures}, actor)
            if failures >= MAX_FAILURES:
                fields.update(failed=0, locked_until=now + LOCK_S)
                self._log("compte_bloque", {"compte": username, "duree_s": LOCK_S}, actor)
            self.users.update(username, **fields)
            raise refused
        self.users.update(username, failed=0, locked_until=None, last_login=_now_iso())
        session_id = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[session_id] = {"username": username, "created": now, "last_seen": now}
        self._log("connexion_reussie", {"compte": username}, actor)
        return session_id, {"username": username, "role": user["role"]}

    def session(self, session_id):
        """Utilisateur de la session, ou None (absente, expirée, compte désactivé)."""
        if not isinstance(session_id, str) or not session_id:
            return None
        now = self.clock()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            expired = now - session["last_seen"] > IDLE_S or now - session["created"] > MAX_SESSION_S
            if expired:
                del self._sessions[session_id]
            else:
                session["last_seen"] = now
        user = self.users.get(session["username"])
        if expired:
            if user:
                self._log("session_expiree", {"compte": session["username"]},
                          {"username": session["username"], "role": user["role"]})
            return None
        if not user or not user.get("active"):
            self.logout(session_id)
            return None
        # Rôle relu à chaque requête : un changement s'applique immédiatement
        return {"username": session["username"], "role": user["role"]}

    def logout(self, session_id, actor=None):
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None and actor is not None:
            self._log("deconnexion", {"compte": session["username"]}, actor)

    def _close_sessions(self, username):
        with self._lock:
            for session_id in [sid for sid, s in self._sessions.items() if s["username"] == username]:
                del self._sessions[session_id]

    # ------------------------------------------------------------------
    # Gestion des comptes (admin dans le web, ou console via tools/manage_users.py)
    def create_user(self, username, password, role, actor) -> dict:
        check_username(username)
        check_password(password)
        if role not in ROLES:
            raise AuthError(f"Rôle inconnu : {role} ({', '.join(ROLES)}).")
        stored = hash_password(password)
        self.users.add(username, {
            "role": role, "password": stored, "active": True, "failed": 0, "locked_until": None,
            "created_at": _now_iso(), "created_by": actor["username"], "password_changed_at": _now_iso(),
        })
        self._log("compte_cree", {"compte": username, "role_compte": role, "actif": True,
                                  "empreinte_mot_de_passe": password_fingerprint(stored)}, actor)
        return self.public(username)

    def set_password(self, username, password, actor) -> dict:
        check_password(password)
        if self.users.get(username) is None:
            raise AuthError("Compte inconnu.")
        stored = hash_password(password)
        self.users.update(username, password=stored, password_changed_at=_now_iso(), failed=0, locked_until=None)
        self._close_sessions(username)
        self._log("mot_de_passe_change", {"compte": username,
                                          "empreinte_mot_de_passe": password_fingerprint(stored)}, actor)
        return self.public(username)

    def set_active(self, username, active: bool, actor) -> dict:
        user = self.users.get(username)
        if user is None:
            raise AuthError("Compte inconnu.")
        if not isinstance(active, bool):
            raise AuthError("État de compte invalide.")
        if not active:
            if username == actor["username"]:
                raise AuthError("Impossible de désactiver son propre compte.")
            if user["role"] == "admin" and self.users.active_admins() == [username]:
                raise AuthError("Impossible de désactiver le dernier administrateur actif.")
        self.users.update(username, active=active)
        if not active:
            self._close_sessions(username)
        self._log("compte_desactive" if not active else "compte_reactive", {"compte": username, "actif": active}, actor)
        return self.public(username)

    def unlock(self, username, actor) -> dict:
        if self.users.get(username) is None:
            raise AuthError("Compte inconnu.")
        self.users.update(username, failed=0, locked_until=None)
        self._log("compte_debloque", {"compte": username}, actor)
        return self.public(username)

    def public(self, username) -> dict:
        user = self.users.get(username)
        locked = bool(user.get("locked_until") and self.clock() < user["locked_until"])
        return {"username": username, "role": user["role"], "active": user.get("active", False), "locked": locked,
                "created_at": user.get("created_at"), "created_by": user.get("created_by"),
                "last_login": user.get("last_login"), "password_changed_at": user.get("password_changed_at")}

    def list_users(self) -> list[dict]:
        return [self.public(name) for name in sorted(self.users.all())]
