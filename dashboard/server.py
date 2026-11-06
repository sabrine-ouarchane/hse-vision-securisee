"""
dashboard/server.py

Serveur web local du POC : pages /live (analyse temps réel), /zones
(zones dangereuses du site), /audit (journal et comptes) et /login, API
correspondantes. Toutes les routes
exigent une session, sauf /login ; les rôles sont vérifiés côté serveur.
"""

import json
import os
import posixpath
import secrets
import socket
from datetime import timedelta
from http.cookies import CookieError, SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from app.config import ROOT
from app.security.audit_log import read_entries, verify
from app.security.auth import MAX_SESSION_S, ROLE_LABELS, AuthError, AuthService, has_role
from app.security.consistency import run_checks
from app.zones.monitor import ZoneMonitor
from app.zones.settings import ConfigError
from dashboard import live_api
from dashboard.live_api import LiveController, list_sources


LIVE = LiveController()
AUTH = AuthService(audit=LIVE.audit)
# Zones dangereuses (étape 9) : surveillance arrêtée au démarrage, boucle lancée par startup()
ZONES_DATA_PATH = ROOT / "data" / "outputs" / "zones"
ZONES = ZoneMonitor(LIVE.audit, live_api.ALERTS_PATH, ZONES_DATA_PATH, presence=LIVE.zone_presence,
                    declared_state=lambda machine_id: LIVE.machine_states.get(machine_id)["state"])
PERIODS = {"heure": 3600, "jour": 86400, "semaine": 7 * 86400}

# ---------------------------------------------------------------------------
# Contrôle d'accès (étape 7) : chaque route a un rôle minimal, vérifié côté
# serveur. Une route absente de cette table est refusée.
# ---------------------------------------------------------------------------
SESSION_COOKIE = "hse_session"
PAGES = {"/live": "live.html", "/zones": "zones.html", "/audit": "audit.html", "/login": "login.html"}
REDIRECTS = {"/": "/live"}  # l'ancien tableau de bord du site (/) est supprimé
PUBLIC_GET = {"/login", "/assets/logo.svg", "/assets/hse.css"}  # la page de connexion en a besoin
OPERATEUR_GET = {"/", "/live", "/zones", "/api/me", "/api/live/status", "/api/live/sources",
                 "/api/live/stream", "/api/consistency", "/api/zones/status", "/api/zones/stats",
                 "/api/zones/series", "/api/zones/courbes"}
OPERATEUR_GET_PREFIXES = ("/assets/", "/api/live/snapshot/")
ADMIN_GET = {"/audit", "/api/audit", "/api/users"}
POST_ROLES = {
    "/api/login": "public",
    "/api/logout": "operateur",
    "/api/live/start": "responsable_hse",
    "/api/live/stop": "responsable_hse",
    "/api/live/upload": "responsable_hse",
    "/api/live/machine-state": "responsable_hse",
    "/api/incidents/ack": "responsable_hse",
    "/api/incidents/archive": "responsable_hse",
    "/api/zones/simulation": "responsable_hse",
    "/api/zones/parametres": "admin",
    "/api/audit/verify": "admin",
    "/api/users/create": "admin",
    "/api/users/password": "admin",
    "/api/users/active": "admin",
    "/api/users/unlock": "admin",
}
SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
# Réponses non HTML (JSON, images, vidéo) : rien ne doit s'y exécuter
DEFAULT_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"


def page_csp(nonce):
    """Pages : scripts du site + script inline autorisé par nonce ; aucun domaine externe."""
    return (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; media-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )


_CONSISTENCY = {"report": None}


def run_consistency(actor=None):
    """Compare le journal d'audit à alerts_log.csv, incidents.json, captures et users.json ; consigne les anomalies."""
    entries = read_entries(LIVE.audit.path)
    report = run_checks(entries, live_api.ALERTS_PATH, live_api.LIVE_SNAPSHOTS_PATH, AUTH.users.all(),
                        ZONES.history.incidents_path)
    _CONSISTENCY["report"] = report
    if not report["ok"]:
        actor = actor or {"username": "systeme", "role": "systeme"}
        LIVE.audit.append("anomalie_coherence", {"nombre": report["count"], "anomalies": report["anomalies"][:20]},
                          utilisateur=actor["username"], role=actor["role"])
    return report


def consistency_summary():
    report = _CONSISTENCY["report"] or run_consistency()
    return {"ok": report["ok"], "count": report["count"], "checked_at": report["checked_at"]}


def startup():
    """Contrôles au lancement du serveur (run_live.py ou python -m dashboard.server)."""
    if not AUTH.users.active_admins():
        raise SystemExit("[ERREUR] Aucun compte administrateur actif. Créer le premier admin avec :\n"
                         "    python tools/manage_users.py create <identifiant> --role admin")
    integrity = verify(LIVE.audit.path, LIVE.audit.key_path)
    if not integrity["ok"]:
        print(f"[AUDIT] Journal d'audit altéré : {integrity['errors'][0]}")
    LIVE.preload_models()  # en arrière-plan : le premier « Démarrer l'analyse » n'attend plus torch ni YOLO
    LIVE.prepare_videos()  # copies 1280 px manquantes, dans un autre fil
    ZONES.start_loop()  # zones en gris (« surveillance arrêtée ») tant qu'un responsable HSE ne la démarre pas
    report = run_consistency()
    if not report["ok"]:
        print(f"[COHÉRENCE] {report['count']} anomalie(s) entre le journal d'audit et les fichiers "
              f"(détails : page /audit). Première : {report['anomalies'][0]['objet']} — "
              f"{report['anomalies'][0]['detail']}")


def filter_audit(entries, query):
    """Filtres de la page /audit : dates (AAAA-MM-JJ), utilisateur, type d'événement."""
    def first(name):
        return (query.get(name) or [""])[0].strip()

    date_from, date_to, user, event = first("du"), first("au"), first("utilisateur"), first("evenement")
    selected = [
        e for e in entries
        if (not date_from or e.get("ts", "")[:10] >= date_from)
        and (not date_to or e.get("ts", "")[:10] <= date_to)
        and (not user or e.get("utilisateur", "systeme") == user)
        and (not event or e.get("event") == event)
    ]
    return {
        "entries": selected[::-1][:1000],
        "total": len(selected),
        "users": sorted({e.get("utilisateur", "systeme") for e in entries}),
        "events": sorted({e.get("event", "") for e in entries}),
    }


class PortInUseError(OSError):
    pass


class HSEHTTPServer(ThreadingHTTPServer):
    """Serveur qui refuse de partager son port.

    http.server active SO_REUSEADDR (allow_reuse_address). Sous Windows, cette
    option permet à un second programme de se lier au même port qu'un serveur
    déjà à l'écoute : selon l'ordre des connexions, l'un ou l'autre répond
    (détournement de port local). SO_EXCLUSIVEADDRUSE interdit tout partage :
    si le port est pris, le démarrage échoue avec PortInUseError.
    """

    allow_reuse_address = False
    allow_reuse_port = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            super().server_bind()
        except OSError as error:
            raise PortInUseError(
                f"Le port {self.server_address[1]} est déjà utilisé par un autre programme "
                f"(ancien serveur encore ouvert ?) : démarrage refusé. Ferme-le, ou change PORT."
            ) from error


def required_role(method, path):
    """Rôle minimal d'une route : "public", un rôle, ou None (route refusée)."""
    if method == "POST":
        return POST_ROLES.get(path)
    if path in PUBLIC_GET:
        return "public"
    if path in OPERATEUR_GET or path.startswith(OPERATEUR_GET_PREFIXES):
        return "operateur"
    if path in ADMIN_GET:
        return "admin"
    return None



class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).resolve().parent), **kwargs)

    def end_headers(self):
        # En-têtes de sécurité sur toutes les réponses (pages, API, erreurs, fichiers)
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.send_header("Content-Security-Policy", getattr(self, "_csp", None) or DEFAULT_CSP)
        super().end_headers()

    # ------------------------------------------------------------------
    # Authentification et rôles
    def _session_id(self):
        try:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else None

    def _discard_body(self):
        """Requête refusée : un petit corps est lu (sinon Windows coupe la connexion avant la réponse) ; un gros envoi ferme la connexion."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if 0 < length <= 65536:
            self.rfile.read(length)
        elif length:
            self.close_connection = True

    def _authorize(self, method):
        """Vérifie session et rôle avant toute autre action ; renvoie False si la réponse est déjà envoyée."""
        self._csp = None
        self.user = None
        path = urlparse(self.path).path
        required = required_role(method, path)
        if required is None:
            self._discard_body()
            self.send_error(404)
            return False
        self.user = AUTH.session(self._session_id())
        if required == "public":
            return True
        if self.user is None:
            if method == "GET" and (path in PAGES or path in REDIRECTS):
                self.send_response(303)
                self.send_header("Location", f"/login?next={quote(path)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._discard_body()
                self._send_json({"error": "Session absente ou expirée : se reconnecter.", "login": "/login"}, 401)
            return False
        if not has_role(self.user["role"], required):
            self._discard_body()
            self._send_json({"error": f"Action réservée au rôle « {ROLE_LABELS[required]} » ou supérieur."}, 403)
            return False
        return True

    def _serve_page(self, path):
        """Page HTML avec un nonce aléatoire pour son script inline (CSP sans 'unsafe-inline' pour les scripts)."""
        nonce = secrets.token_urlsafe(16)
        html = (Path(self.directory) / PAGES[path]).read_text(encoding="utf-8")
        body = html.replace("<script>", f'<script nonce="{nonce}">').encode("utf-8")
        self._csp = page_csp(nonce)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self, limit=4096):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= limit:
            raise ValueError("Requête trop volumineuse.")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("Le corps de la requête doit être un objet JSON.")
        return payload

    def _login(self):
        if not self._same_origin():  # protection contre le « login CSRF »
            self._discard_body()
            self._send_json({"error": "Requête refusée (origine ou type de contenu invalide)."}, 403)
            return
        try:
            payload = self._read_json()
            session_id, user = AUTH.login(payload.get("username"), payload.get("password"))
        except AuthError as error:
            self._send_json({"error": str(error)}, 401)
            return
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)
            return
        body = json.dumps({"user": user}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Set-Cookie", f"{SESSION_COOKIE}={session_id}; HttpOnly; SameSite=Strict; Path=/; "
                                       f"Max-Age={MAX_SESSION_S}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _logout(self):
        if not self._same_origin():
            self._discard_body()
            self._send_json({"error": "Requête refusée (origine ou type de contenu invalide)."}, 403)
            return
        AUTH.logout(self._session_id(), self.user)
        body = b'{"logged_out": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Set-Cookie", f"{SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_error(405)

    def list_directory(self, path):
        self.send_error(404)
        return None

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _same_origin(self):
        """Refuse les requêtes de modification émises par un autre site (CSRF vers 127.0.0.1)."""
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        return content_type == "application/json" and (origin is None or origin == f"http://{host}")

    def _live_get(self, request_path):
        if request_path == "/api/live/status":
            self._send_json(LIVE.status())
        elif request_path == "/api/live/sources":
            sources = list_sources()
            machine_ids = sorted({s["machine_id"] for s in sources if s.get("machine_id")})
            self._send_json({"sources": sources, **LIVE.machine_overview(machine_ids)})
        elif request_path == "/api/live/stream":
            self._stream_live()
        elif request_path.startswith("/api/live/snapshot/"):
            session_id, _, filename = unquote(request_path[len("/api/live/snapshot/"):]).partition("/")
            path = LIVE.snapshot_path(session_id, filename)
            if path is None:
                self.send_error(404, "Capture introuvable.")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def _stream_live(self):
        session = LIVE.session
        if session is None:
            self.send_error(404, "Aucune analyse en cours.")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        seq = 0
        try:
            while LIVE.session is session:
                new_seq, jpeg = session.wait_jpeg(seq, timeout=2.0)
                if jpeg is None or new_seq == seq:
                    if not session.running:
                        break
                    continue
                seq = new_seq
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n"
                )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _live_upload(self):
        """Vidéo envoyée depuis le dashboard ; l'en-tête X-Filename impose une requête du même site."""
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{self.headers.get('Host', '')}":
            self._send_json({"error": "Requête refusée (origine invalide)."}, 403)
            return
        try:
            filename = unquote(self.headers.get("X-Filename", ""))
            length = int(self.headers.get("Content-Length", "0"))
            saved = LIVE.add_uploaded_video(filename, self.rfile, length, self.user)
        except (ValueError, OSError) as error:
            self.close_connection = True  # le corps non lu ne doit pas être réinterprété
            self._send_json({"error": str(error)}, 400)
            return
        self._send_json({"saved": True, "source": f"video:{saved}", "filename": saved})

    def _live_post(self, endpoint):
        if not self._same_origin():
            self._discard_body()
            self._send_json({"error": "Requête refusée (origine ou type de contenu invalide)."}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 4096:
                raise ValueError("Requête trop volumineuse.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Le corps de la requête doit être un objet JSON.")
            if endpoint == "/api/live/start":
                result = LIVE.start(payload.get("source"), payload.get("harness_required"), user=self.user)
            elif endpoint == "/api/live/stop":
                LIVE.stop(self.user)
                result = {"stopped": True}
            elif endpoint == "/api/live/machine-state":
                result = LIVE.set_machine_state(payload.get("machine_id"), payload.get("state"),
                                                payload.get("unknown_policy"), user=self.user)
            else:
                self._send_json({"error": "Point d'accès inconnu."}, 404)
                return
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)
            return
        except RuntimeError as error:
            self._send_json({"error": str(error)}, 409)
            return
        self._send_json(result)

    def _admin_post(self, endpoint):
        """Vérification d'intégrité et gestion des comptes (admin)."""
        if not self._same_origin():
            self._discard_body()
            self._send_json({"error": "Requête refusée (origine ou type de contenu invalide)."}, 403)
            return
        try:
            payload = self._read_json()
            if endpoint == "/api/audit/verify":
                integrity = verify(LIVE.audit.path, LIVE.audit.key_path)
                consistency = run_consistency(self.user)
                LIVE.audit.append("verification_integrite", {
                    "journal_intact": integrity["ok"], "erreurs_journal": len(integrity["errors"]),
                    "anomalies_coherence": consistency["count"],
                }, utilisateur=self.user["username"], role=self.user["role"])
                self._send_json({"integrity": integrity, "consistency": consistency})
                return
            username = payload.get("username")
            if endpoint == "/api/users/create":
                result = AUTH.create_user(username, payload.get("password"), payload.get("role"), self.user)
            elif endpoint == "/api/users/password":
                result = AUTH.set_password(username, payload.get("password"), self.user)
            elif endpoint == "/api/users/active":
                result = AUTH.set_active(username, payload.get("active"), self.user)
            else:
                result = AUTH.unlock(username, self.user)
        except AuthError as error:
            self._send_json({"error": str(error)}, 400)
            return
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)
            return
        self._send_json({"user": result})

    # ------------------------------------------------------------------
    # Zones dangereuses (étape 9)
    def _period(self):
        query = parse_qs(urlparse(self.path).query)
        period = (query.get("periode") or ["heure"])[0]
        if period not in PERIODS:
            raise ValueError(f"Période inconnue : {period} ({', '.join(PERIODS)}).")
        end = ZONES.clock()
        return period, end - timedelta(seconds=PERIODS[period]), end, query

    def _zones_get(self, request_path):
        try:
            if request_path == "/api/zones/status":
                status = ZONES.status()
                status["peut_simuler"] = has_role(self.user["role"], "responsable_hse")
                status["peut_regler"] = has_role(self.user["role"], "admin")
                self._send_json(status)
            elif request_path == "/api/zones/stats":
                period, start, end, _ = self._period()
                self._send_json({"periode": period, **ZONES.statistics(start, end, LIVE.all_incidents())})
            elif request_path == "/api/zones/series":
                period, start, end, query = self._period()
                sensor_id = (query.get("capteur") or [""])[0]
                self._send_json({"periode": period, **ZONES.series(sensor_id, start, end)})
            elif request_path == "/api/zones/courbes":
                period, start, end, _ = self._period()
                self._send_json({"periode": period, **ZONES.all_series(start, end)})
            else:
                self.send_error(404)
        except LookupError as error:
            self._send_json({"error": str(error)}, 404)
        except ValueError as error:
            self._send_json({"error": str(error)}, 400)

    def _zones_post(self, endpoint):
        """Simulation (responsable HSE) et réglages bornés (admin) ; rôle déjà vérifié par _authorize."""
        if not self._same_origin():
            self._discard_body()
            self._send_json({"error": "Requête refusée (origine ou type de contenu invalide)."}, 403)
            return
        try:
            payload = self._read_json()
            if endpoint == "/api/zones/parametres":
                result = ZONES.update_parameters(payload.get("parametres"), self.user)
            elif payload.get("action") == "demarrer":
                result = ZONES.start(self.user)
            elif payload.get("action") == "arreter":
                result = ZONES.stop(self.user)
            elif payload.get("action") == "scenario":
                result = ZONES.set_scenario(payload.get("zone"), payload.get("scenario"), payload.get("capteur"),
                                            self.user)
            else:
                raise ValueError("Action attendue : demarrer, arreter ou scenario.")
        except (ConfigError, ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)
            return
        except RuntimeError as error:
            self._send_json({"error": str(error)}, 409)
            return
        self._send_json(result)

    def _incident_review(self, endpoint):
        """Acquitter ou archiver (motif obligatoire) : aucune route ne supprime d'incident."""
        if not self._same_origin():
            self._discard_body()
            self._send_json({"error": "Requête refusée (origine ou type de contenu invalide)."}, 403)
            return
        try:
            payload = self._read_json()
            if endpoint == "/api/incidents/ack":
                result = LIVE.acknowledge(payload.get("session_id"), payload.get("incident_id"), self.user)
            else:
                result = LIVE.archive_incident(payload.get("session_id"), payload.get("incident_id"),
                                               payload.get("motif"), self.user)
        except LookupError as error:
            self._send_json({"error": str(error)}, 404)
            return
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)
            return
        self._send_json({"incident": result})

    def do_GET(self):
        if not self._authorize("GET"):
            return
        request_path = urlparse(self.path).path
        if request_path in PAGES:
            self._serve_page(request_path)
            return
        if request_path in REDIRECTS:
            self.send_response(303)
            self.send_header("Location", REDIRECTS[request_path])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if request_path == "/api/me":
            self._send_json({**self.user, "role_label": ROLE_LABELS[self.user["role"]]})
            return
        if request_path == "/api/consistency":
            summary = consistency_summary()
            if self.user["role"] == "admin":
                summary["anomalies"] = _CONSISTENCY["report"]["anomalies"]
            self._send_json(summary)
            return
        if request_path == "/api/audit":
            self._send_json(filter_audit(read_entries(LIVE.audit.path), parse_qs(urlparse(self.path).query)))
            return
        if request_path == "/api/users":
            self._send_json({"users": AUTH.list_users(), "roles": ROLE_LABELS})
            return
        if request_path.startswith("/api/live/"):
            self._live_get(request_path)
            return
        if request_path.startswith("/api/zones/"):
            self._zones_get(request_path)
            return
        if request_path.startswith("/assets/"):
            # Seul dashboard/assets est servi (pas les sources Python du dossier), sans « .. »
            normalized = posixpath.normpath(unquote(request_path))
            if not normalized.startswith("/assets/"):
                self.send_error(404)
                return
            self.path = normalized
            super().do_GET()
            return
        self.send_error(404)

    def do_POST(self):
        if not self._authorize("POST"):
            return
        endpoint = urlparse(self.path).path
        if endpoint == "/api/login":
            self._login()
            return
        if endpoint == "/api/logout":
            self._logout()
            return
        if endpoint == "/api/live/upload":
            self._live_upload()
            return
        if endpoint.startswith("/api/live/"):
            self._live_post(endpoint)
            return
        if endpoint in {"/api/zones/simulation", "/api/zones/parametres"}:
            self._zones_post(endpoint)
            return
        if endpoint in {"/api/incidents/ack", "/api/incidents/archive"}:
            self._incident_review(endpoint)
            return
        if endpoint == "/api/audit/verify" or endpoint.startswith("/api/users/"):
            self._admin_post(endpoint)
            return
        self.send_error(404)


if __name__ == "__main__":
    startup()
    port = int(os.environ.get("DASHBOARD_PORT", "8000"))
    try:
        server = HSEHTTPServer(("127.0.0.1", port), DashboardHandler)
    except PortInUseError as error:
        raise SystemExit(f"[ERREUR] {error}")
    print(f"Dashboard local : http://127.0.0.1:{port}/login")
    server.serve_forever()
