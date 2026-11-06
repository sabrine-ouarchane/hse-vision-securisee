import json
import os
import tempfile
import unittest
from pathlib import Path

from app.security.audit_log import AuditLog, read_entries, verify
from app.security.auth import (
    IDLE_S, LOCK_S, MAX_FAILURES, AuthError, AuthService, hash_password, has_role, verify_password,
)

PASSWORD = "mot-de-passe-solide"
CONSOLE = {"username": "console:test", "role": "console"}


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.audit = AuditLog(self.root / "audit_log.jsonl", self.root / "secrets" / "audit.key")
        self.clock = Clock()
        self.auth = AuthService(self.root / "users.json", audit=self.audit, clock=self.clock)
        self.auth.create_user("admin1", PASSWORD, "admin", CONSOLE)

    def tearDown(self):
        self.directory.cleanup()

    def events(self, event=None):
        return [e for e in read_entries(self.audit.path) if event is None or e["event"] == event]


class PasswordTests(unittest.TestCase):
    def test_hash_is_salted_and_never_plain(self):
        first, second = hash_password(PASSWORD), hash_password(PASSWORD)
        self.assertNotEqual(first, second)
        self.assertNotIn(PASSWORD, first)
        self.assertTrue(first.startswith("scrypt$"))
        self.assertTrue(verify_password(PASSWORD, first))
        self.assertFalse(verify_password(PASSWORD + "x", first))
        self.assertFalse(verify_password(PASSWORD, "corrompu"))

    def test_role_hierarchy(self):
        self.assertTrue(has_role("admin", "responsable_hse"))
        self.assertTrue(has_role("responsable_hse", "operateur"))
        self.assertFalse(has_role("operateur", "responsable_hse"))
        self.assertFalse(has_role("inconnu", "operateur"))


class LoginTests(AuthTestCase):
    def test_users_file_has_no_plain_password(self):
        text = (self.root / "users.json").read_text(encoding="utf-8")
        self.assertNotIn(PASSWORD, text)

    def test_correct_login_opens_session(self):
        session_id, user = self.auth.login("admin1", PASSWORD)
        self.assertGreaterEqual(len(session_id), 40)
        self.assertEqual(self.auth.session(session_id), {"username": "admin1", "role": "admin"})
        self.assertEqual(self.events("connexion_reussie")[0]["utilisateur"], "admin1")

    def test_wrong_password_and_unknown_user_get_same_message(self):
        with self.assertRaisesRegex(AuthError, "Identifiant ou mot de passe incorrect"):
            self.auth.login("admin1", "mauvais-mot-de-passe")
        with self.assertRaisesRegex(AuthError, "Identifiant ou mot de passe incorrect"):
            self.auth.login("personne", PASSWORD)
        failures = self.events("connexion_echouee")
        self.assertEqual([e["data"]["identifiant_tente"] for e in failures], ["admin1", "personne"])
        self.assertEqual(failures[0]["data"]["motif"], "mot de passe incorrect")

    def test_account_locked_after_five_failures_then_released(self):
        for _ in range(MAX_FAILURES):
            with self.assertRaises(AuthError):
                self.auth.login("admin1", "mauvais-mot-de-passe")
        with self.assertRaisesRegex(AuthError, "bloqué"):
            self.auth.login("admin1", PASSWORD)  # même le bon mot de passe est refusé
        self.assertEqual(len(self.events("compte_bloque")), 1)
        self.clock.now += LOCK_S + 1
        self.auth.login("admin1", PASSWORD)

    def test_lock_survives_restart(self):
        for _ in range(MAX_FAILURES):
            with self.assertRaises(AuthError):
                self.auth.login("admin1", "mauvais-mot-de-passe")
        restarted = AuthService(self.root / "users.json", audit=self.audit, clock=self.clock)
        with self.assertRaisesRegex(AuthError, "bloqué"):
            restarted.login("admin1", PASSWORD)

    def test_session_expires_after_inactivity(self):
        session_id, _ = self.auth.login("admin1", PASSWORD)
        self.clock.now += IDLE_S - 10
        self.assertIsNotNone(self.auth.session(session_id))  # activité : l'inactivité repart de zéro
        self.clock.now += IDLE_S + 1
        self.assertIsNone(self.auth.session(session_id))
        self.assertEqual(len(self.events("session_expiree")), 1)

    def test_logout_and_disabled_account(self):
        self.auth.create_user("op1", PASSWORD, "operateur", {"username": "admin1", "role": "admin"})
        session_id, _ = self.auth.login("op1", PASSWORD)
        self.auth.set_active("op1", False, {"username": "admin1", "role": "admin"})
        self.assertIsNone(self.auth.session(session_id))  # sessions fermées à la désactivation
        with self.assertRaises(AuthError):
            self.auth.login("op1", PASSWORD)
        other, _ = self.auth.login("admin1", PASSWORD)
        self.auth.logout(other, {"username": "admin1", "role": "admin"})
        self.assertIsNone(self.auth.session(other))
        self.assertEqual(self.events("deconnexion")[0]["utilisateur"], "admin1")


class AccountManagementTests(AuthTestCase):
    ADMIN = {"username": "admin1", "role": "admin"}

    def test_cannot_disable_self_or_last_admin(self):
        with self.assertRaisesRegex(AuthError, "propre compte"):
            self.auth.set_active("admin1", False, self.ADMIN)
        with self.assertRaisesRegex(AuthError, "dernier administrateur"):
            self.auth.set_active("admin1", False, CONSOLE)
        self.auth.create_user("admin2", PASSWORD, "admin", self.ADMIN)
        self.auth.set_active("admin1", False, {"username": "admin2", "role": "admin"})
        self.assertFalse(self.auth.public("admin1")["active"])

    def test_no_account_deletion_and_no_duplicates(self):
        self.assertFalse(hasattr(self.auth, "delete_user"))
        with self.assertRaisesRegex(AuthError, "existe déjà"):
            self.auth.create_user("admin1", PASSWORD, "admin", self.ADMIN)

    def test_password_rules(self):
        with self.assertRaisesRegex(AuthError, "12 caractères"):
            self.auth.create_user("op2", "court", "operateur", self.ADMIN)
        with self.assertRaisesRegex(AuthError, "Identifiant invalide"):
            self.auth.create_user("Op 2", PASSWORD, "operateur", self.ADMIN)
        with self.assertRaisesRegex(AuthError, "Rôle inconnu"):
            self.auth.create_user("op2", PASSWORD, "superviseur", self.ADMIN)

    def test_account_actions_are_audited_with_author(self):
        self.auth.create_user("hse1", PASSWORD, "responsable_hse", self.ADMIN)
        self.auth.set_password("hse1", PASSWORD + "2", self.ADMIN)
        self.auth.set_active("hse1", False, self.ADMIN)
        self.auth.unlock("hse1", self.ADMIN)
        events = [(e["event"], e["utilisateur"], e["role"]) for e in self.events()][1:]
        self.assertEqual(events, [
            ("compte_cree", "admin1", "admin"), ("mot_de_passe_change", "admin1", "admin"),
            ("compte_desactive", "admin1", "admin"), ("compte_debloque", "admin1", "admin"),
        ])
        self.assertTrue(verify(self.audit.path, self.audit.key_path)["ok"])
        self.assertNotIn(PASSWORD, self.audit.path.read_text(encoding="utf-8"))

    def test_console_tool_creates_first_admin(self):
        from tools.manage_users import main

        auth = AuthService(self.root / "autres_users.json", audit=self.audit)
        self.assertEqual(main(["create", "chef", "--role", "admin"], auth, lambda: PASSWORD), 0)
        self.assertEqual(main(["create", "chef2", "--role", "admin"], auth, lambda: "court"), 1)
        users = json.loads((self.root / "autres_users.json").read_text(encoding="utf-8"))["users"]
        self.assertEqual(list(users), ["chef"])
        self.assertTrue(self.events("compte_cree")[-1]["utilisateur"].startswith("console:"))


class HttpTestCase(unittest.TestCase):
    """Vrai serveur HTTP sur un port libre, comptes et journal d'audit temporaires."""

    ROLES = {"admin1": "admin", "hse1": "responsable_hse", "op1": "operateur"}

    def setUp(self):
        import threading
        from unittest.mock import patch

        import dashboard.server as server

        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.audit = AuditLog(self.root / "audit_log.jsonl", self.root / "secrets" / "audit.key")
        self.auth = AuthService(self.root / "users.json", audit=self.audit)
        for username, role in self.ROLES.items():
            self.auth.create_user(username, PASSWORD, role, CONSOLE)
        self.patches = [patch.object(server, "AUTH", self.auth)]
        for item in self.patches:
            item.start()
        self.server = server.HSEHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        for item in self.patches:
            item.stop()
        self.directory.cleanup()

    def request(self, method, path, body=None, cookie=None, headers=None, raw=None):
        """Retourne (statut, en-têtes, corps) ; n'abandonne pas sur les codes d'erreur."""
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        all_headers = dict(headers or {})
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            all_headers.setdefault("Content-Type", "application/json")
        if cookie:
            all_headers["Cookie"] = f"hse_session={cookie}"
        connection.request(method, path, body=raw, headers=all_headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, response.headers, data

    def login(self, username):
        status, headers, _ = self.request("POST", "/api/login", {"username": username, "password": PASSWORD})
        self.assertEqual(status, 200)
        return headers["Set-Cookie"].split(";")[0].split("=", 1)[1]


class AccessControlHttpTests(HttpTestCase):
    PROTECTED_GET = ["/api/me", "/api/live/status", "/api/live/sources", "/api/live/stream",
                     "/api/live/snapshot/20261029_100000/incident_0001.jpg", "/assets/autre.png",
                     "/api/consistency", "/api/audit", "/api/users"]
    PROTECTED_POST = ["/api/live/start", "/api/live/stop", "/api/live/machine-state", "/api/live/upload",
                      "/api/incidents/ack", "/api/incidents/archive", "/api/audit/verify", "/api/users/create",
                      "/api/users/active", "/api/users/password", "/api/users/unlock", "/api/logout"]
    # Ancien tableau de bord du site (version 1) : routes supprimées
    REMOVED_GET = ["/index.html", "/api/dashboard", "/media/demo.mp4", "/stream/demo.mp4",
                   "/assets/leaflet-1.9.4/leaflet.js", "/assets/site-plan.png",
                   "/assets/dashboard-background.png"]
    REMOVED_POST = ["/api/alert-status", "/api/gas-simulation", "/api/map-zone"]

    def test_access_without_session_is_refused(self):
        for page in ("/", "/live", "/audit"):
            status, headers, _ = self.request("GET", page)
            self.assertEqual((status, headers["Location"]), (303, f"/login?next={page}"), page)
        for path in self.PROTECTED_GET:
            self.assertEqual(self.request("GET", path)[0], 401, path)
        for path in self.PROTECTED_POST:
            self.assertEqual(self.request("POST", path, {})[0], 401, path)
        self.assertEqual(self.request("GET", "/api/me", cookie="session-inventee")[0], 401)
        self.assertEqual(self.request("GET", "/login")[0], 200)
        self.assertEqual(self.request("GET", "/assets/logo.svg")[0], 200)

    def test_login_page_shows_logo_and_logging_notice_once(self):
        page = self.request("GET", "/login")[2].decode("utf-8")
        self.assertEqual(page.count("/assets/logo.svg"), 1)  # grand logo sous le titre, plus de petit logo
        self.assertEqual(page.count("connexions sont journalisées"), 1)  # sous le bouton seulement
        self.assertLess(page.index("<h1>Surveillance HSE</h1>"), page.index("logo.svg"))

    def test_shared_stylesheet_is_public_alone_and_used_by_every_page(self):
        status, headers, body = self.request("GET", "/assets/hse.css")
        self.assertEqual((status, headers["Content-Type"].split(";")[0]), (200, "text/css"))
        css = body.decode("utf-8")
        for forbidden in ("http:", "https:", "@import", "//"):
            self.assertNotIn(forbidden, css.replace("/*", "").replace("*/", ""), forbidden)
        # seul ce fichier est public : le reste de /assets/ exige toujours une session
        self.assertEqual(self.request("GET", "/assets/autre.css")[0], 401)
        cookie = self.login("admin1")
        for page in ("/login", "/live", "/zones", "/audit"):
            html = self.request("GET", page, cookie=cookie)[2].decode("utf-8")
            self.assertIn('<link rel="stylesheet" href="/assets/hse.css">', html, page)
            if page != "/login":
                self.assertIn('class="topbar"', html, page)
                self.assertIn(f'href="{page}" aria-current="page"', html, page)

    def test_source_files_and_unknown_routes_are_not_served(self):
        cookie = self.login("admin1")
        for path in ("/server.py", "/live_api.py", "/live.html", "/index.html", "/assets/../server.py",
                     "/assets/%2e%2e/server.py", "/assets/", "/inconnu"):
            self.assertEqual(self.request("GET", path, cookie=cookie)[0], 404, path)
        self.assertEqual(self.request("HEAD", "/live", cookie=cookie)[0], 405)

    def test_operator_cannot_call_privileged_api_directly(self):
        cookie = self.login("op1")
        self.assertEqual(self.request("GET", "/api/live/status", cookie=cookie)[0], 200)
        for path in self.PROTECTED_POST[:-1]:
            status, _, data = self.request("POST", path, {"machine_id": "engin_pilote", "state": "arret"}, cookie=cookie)
            self.assertEqual(status, 403, path)
            self.assertIn("réservée", json.loads(data)["error"])
        for path in ("/audit", "/api/audit", "/api/users"):
            self.assertEqual(self.request("GET", path, cookie=cookie)[0], 403, path)

    def test_hse_manager_cannot_use_admin_routes(self):
        cookie = self.login("hse1")
        for path in ("/audit", "/api/audit", "/api/users"):
            self.assertEqual(self.request("GET", path, cookie=cookie)[0], 403, path)
        for path in ("/api/users/create", "/api/users/active", "/api/audit/verify"):
            self.assertEqual(self.request("POST", path, {"username": "x"}, cookie=cookie)[0], 403, path)
        self.assertEqual(json.loads(self.request("GET", "/api/me", cookie=cookie)[2])["role"], "responsable_hse")

    def test_wrong_password_lockout_and_logout_over_http(self):
        for _ in range(MAX_FAILURES):
            status, _, data = self.request("POST", "/api/login", {"username": "op1", "password": "faux-mot-de-passe"})
            self.assertEqual(status, 401)
        status, _, data = self.request("POST", "/api/login", {"username": "op1", "password": PASSWORD})
        self.assertEqual(status, 401)
        self.assertIn("bloqué", json.loads(data)["error"])
        cookie = self.login("hse1")
        self.assertEqual(self.request("POST", "/api/logout", {}, cookie=cookie)[0], 200)
        self.assertEqual(self.request("GET", "/api/me", cookie=cookie)[0], 401)

    def test_login_csrf_is_refused(self):
        foreign = {"Origin": "http://site-malveillant.example"}
        status, headers, _ = self.request("POST", "/api/login", {"username": "admin1", "password": PASSWORD},
                                          headers=foreign)
        self.assertEqual(status, 403)
        self.assertIsNone(headers["Set-Cookie"])
        form = b"username=admin1&password=" + PASSWORD.encode()
        status, _, _ = self.request("POST", "/api/login", raw=form,
                                    headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(status, 403)  # formulaire HTML d'un autre site : pas de JSON

    def test_session_cookie_flags_and_security_headers(self):
        status, headers, _ = self.request("POST", "/api/login", {"username": "op1", "password": PASSWORD})
        cookie_header = headers["Set-Cookie"]
        for flag in ("HttpOnly", "SameSite=Strict", "Path=/"):
            self.assertIn(flag, cookie_header)
        cookie = cookie_header.split(";")[0].split("=", 1)[1]
        for method, path in (("GET", "/live"), ("GET", "/api/me"), ("GET", "/inconnu"), ("GET", "/login"),
                             ("POST", "/api/users/create")):
            _, headers, _ = self.request(method, path, {} if method == "POST" else None, cookie=cookie)
            self.assertEqual(headers["X-Frame-Options"], "DENY", path)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff", path)
            self.assertEqual(headers["Referrer-Policy"], "no-referrer", path)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"], path)

    def test_pages_use_nonce_and_no_external_domain(self):
        cookie = self.login("admin1")
        for page in ("/live", "/audit", "/login"):
            status, headers, body = self.request("GET", page, cookie=cookie)
            self.assertEqual(status, 200, page)
            csp = headers["Content-Security-Policy"]
            nonce = csp.split("'nonce-")[1].split("'")[0]
            self.assertIn(f'<script nonce="{nonce}">'.encode(), body)
            self.assertNotIn("'unsafe-inline'", csp.split("script-src")[1].split(";")[0])
            # Plus aucun CDN ni fond de carte externe (OpenStreetMap retiré avec l'ancien tableau de bord)
            self.assertNotIn("http", csp, page)
            self.assertNotIn(b"https://", body, page)
            self.assertNotIn(b"leaflet", body.lower(), page)
        other_nonce = self.request("GET", "/live", cookie=cookie)[1]["Content-Security-Policy"]
        self.assertNotEqual(csp, other_nonce)  # nonce différent à chaque réponse

    def test_root_redirects_to_live(self):
        cookie = self.login("op1")
        status, headers, _ = self.request("GET", "/", cookie=cookie)
        self.assertEqual((status, headers["Location"]), (303, "/live"))

    def test_removed_site_dashboard_routes_return_404_even_for_admin(self):
        import dashboard.server as server

        cookie = self.login("admin1")
        for path in self.REMOVED_GET:
            self.assertEqual(self.request("GET", path, cookie=cookie)[0], 404, path)
            if not path.startswith("/assets/"):  # fichiers supprimés du dossier assets : 404
                self.assertIsNone(server.required_role("GET", path), path)
        for path in self.REMOVED_POST:
            self.assertEqual(self.request("POST", path, {}, cookie=cookie)[0], 404, path)
            self.assertNotIn(path, server.POST_ROLES)


class LiveHttpTestCase(HttpTestCase):
    """Comme HttpTestCase, avec un LiveController isolé (fichiers temporaires) et un incident archivé."""

    SESSION = "20261029_100000"

    def setUp(self):
        from unittest.mock import patch

        import dashboard.live_api as live_api
        import dashboard.server as server
        from app.sensors.machine_state_provider import MachineStateProvider

        super().setUp()
        data = self.root / "data"
        paths = {"TEST_VIDEOS_PATH": data / "videos", "LIVE_SNAPSHOTS_PATH": data / "live_incidents",
                 "ALERTS_PATH": data / "alerts_log.csv", "AUDIT_LOG_PATH": data / "inutilise.jsonl",
                 "REVIEWS_PATH": data / "incident_reviews.json"}
        for name, value in paths.items():
            item = patch.object(live_api, name, value)
            item.start()
            self.patches.append(item)
        session_dir = paths["LIVE_SNAPSHOTS_PATH"] / self.SESSION
        session_dir.mkdir(parents=True)
        self.incident = {"id": 1, "kind": "casque", "label": "Absence de casque", "severity": "critique",
                         "message": "Personne #2 : absence de casque", "track_ids": [2], "started_t": 1.0,
                         "last_t": 4.0, "ended_t": 4.0, "duration_s": 3.0, "frame_idx": 10,
                         "wall_time": "2026-10-29T10:00:05+00:00", "peak_confidence": 0.9, "status": "terminee",
                         "snapshot": None, "source": "Caméra test", "session_id": self.SESSION,
                         "video": "test.mp4", "camera_name": "Caméra test"}
        (session_dir / "incidents.json").write_text(json.dumps([self.incident]), encoding="utf-8")
        self.live = live_api.LiveController()
        self.live.audit = self.audit
        self.live.machine_states = MachineStateProvider(self.root / "machine_states.json")
        item = patch.object(server, "LIVE", self.live)
        item.start()
        self.patches.append(item)


class TraceabilityHttpTests(LiveHttpTestCase):
    def test_machine_state_is_declared_by_named_user(self):
        cookie = self.login("hse1")
        status, _, _ = self.request("POST", "/api/live/machine-state",
                                    {"machine_id": "engin_pilote", "state": "marche"}, cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(self.live.machine_states.get("engin_pilote")["source"], "hse1")
        entry = [e for e in read_entries(self.audit.path) if e["event"] == "etat_machine_modifie"][-1]
        self.assertEqual((entry["utilisateur"], entry["role"]), ("hse1", "responsable_hse"))
        self.assertEqual((entry["data"]["avant"], entry["data"]["apres"]), ("inconnu", "marche"))

    def test_acknowledge_and_archive_without_deletion(self):
        cookie = self.login("hse1")
        target = {"session_id": self.SESSION, "incident_id": 1}
        self.assertEqual(self.request("POST", "/api/incidents/ack", target, cookie=cookie)[0], 200)
        self.assertEqual(self.request("POST", "/api/incidents/ack", target, cookie=cookie)[0], 400)  # déjà acquitté
        for motif in (None, "", "   ", "ok"):
            status, _, data = self.request("POST", "/api/incidents/archive", {**target, "motif": motif}, cookie=cookie)
            self.assertEqual(status, 400, motif)
            self.assertIn("Motif d'archivage obligatoire", json.loads(data)["error"])
        status, _, _ = self.request("POST", "/api/incidents/archive",
                                    {**target, "motif": "Fausse alerte vérifiée sur la capture"}, cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(self.request("POST", "/api/incidents/archive", {"session_id": self.SESSION,
                         "incident_id": 99, "motif": "inexistant"}, cookie=cookie)[0], 404)
        incidents = json.loads(self.request("GET", "/api/live/status", cookie=cookie)[2])["all_incidents"]
        self.assertEqual(len(incidents), 1)  # toujours présent : archivé, pas supprimé
        review = incidents[0]["review"]
        self.assertEqual((review["acquitte_par"], review["archive_par"]), ("hse1", "hse1"))
        self.assertEqual(review["motif_archivage"], "Fausse alerte vérifiée sur la capture")
        self.assertEqual(json.loads((self.root / "data/live_incidents" / self.SESSION / "incidents.json")
                                    .read_text(encoding="utf-8")), [self.incident])
        events = {e["event"]: e for e in read_entries(self.audit.path)}
        self.assertEqual(events["incident_archive"]["data"]["motif"], "Fausse alerte vérifiée sur la capture")
        self.assertEqual(events["incident_acquitte"]["utilisateur"], "hse1")

    def test_every_user_action_is_audited_with_its_author(self):
        cookie = self.login("hse1")
        self.request("POST", "/api/live/machine-state", {"machine_id": "engin_pilote", "state": "arret"}, cookie=cookie)
        self.request("POST", "/api/incidents/ack", {"session_id": self.SESSION, "incident_id": 1}, cookie=cookie)
        self.request("POST", "/api/live/upload", raw=b"pas une video", cookie=cookie,
                     headers={"Content-Type": "application/octet-stream", "X-Filename": "piege.mp4"})
        self.login("admin1")
        self.request("POST", "/api/logout", {}, cookie=cookie)
        self.request("POST", "/api/login", {"username": "op1", "password": "mauvais-mot-de-passe"})
        authors = {e["event"]: (e["utilisateur"], e["role"]) for e in read_entries(self.audit.path)}
        self.assertEqual(authors["connexion_reussie"], ("admin1", "admin"))
        self.assertEqual(authors["etat_machine_modifie"], ("hse1", "responsable_hse"))
        self.assertEqual(authors["incident_acquitte"], ("hse1", "responsable_hse"))
        self.assertEqual(authors["video_refusee"], ("hse1", "responsable_hse"))
        self.assertEqual(authors["deconnexion"], ("hse1", "responsable_hse"))
        self.assertEqual(authors["connexion_echouee"], ("op1", "operateur"))
        self.assertEqual(authors["compte_cree"], ("console:test", "console"))
        self.assertTrue(verify(self.audit.path, self.audit.key_path)["ok"])


class ConsistencyHttpTests(LiveHttpTestCase):
    CHECKED = "20261029_110000"

    def setUp(self):
        from unittest.mock import patch

        import dashboard.server as server

        super().setUp()
        item = patch.object(server, "_CONSISTENCY", {"report": None})
        item.start()
        self.patches.append(item)
        self.data = self.root / "data"
        self.make_session(self.CHECKED)

    def make_session(self, session_id):
        """Session terminée réelle : incidents journalisés, CSV, captures, incidents.json."""
        import numpy as np

        from app.alerts.alert_manager import AlertManager
        from app.alerts.incident_manager import IncidentManager

        directory = self.data / "live_incidents" / session_id
        manager = IncidentManager(AlertManager(str(self.data / "alerts_log.csv"), cooldown_frames=0),
                                  snapshot_dir=directory, source_name="Caméra test", audit=self.audit,
                                  audit_context={"session_id": session_id})
        frame = np.full((240, 320, 3), 90, np.uint8)
        situations = [{"kind": kind, "track_id": 1, "bbox": (100, 40, 160, 220), "severity": "critique",
                       "confidence": 0.9, "message": f"Personne #1 : absence de {kind}"} for kind in ("casque", "gilet")]
        for step in range(3):
            manager.update(situations, 1 + step * 0.5, step, frame)
        manager.close_all()
        incidents = [dict({k: v for k, v in i.items() if k != "bbox"}, session_id=session_id)
                     for i in manager.snapshot()]
        (directory / "incidents.json").write_text(json.dumps(incidents, ensure_ascii=False), encoding="utf-8")
        self.audit.append("session_arretee", {"session_id": session_id})

    def verify_as_admin(self):
        status, _, data = self.request("POST", "/api/audit/verify", {}, cookie=self.login("admin1"))
        self.assertEqual(status, 200)
        return json.loads(data)

    def details(self, report):
        return [f"{a['objet']} — {a['detail']}" for a in report["consistency"]["anomalies"]]

    def test_clean_files_are_consistent(self):
        report = self.verify_as_admin()
        self.assertTrue(report["integrity"]["ok"])
        self.assertEqual(report["consistency"]["anomalies"], [])

    def test_incident_deleted_from_csv_is_detected(self):
        csv_path = self.data / "alerts_log.csv"
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        csv_path.write_text("\n".join(line for line in lines if "gilet" not in line) + "\n", encoding="utf-8")
        report = self.verify_as_admin()
        self.assertEqual(self.details(report),
                         [f"incident #2 · session {self.CHECKED} · gilet — ligne supprimée de alerts_log.csv"])
        operator = self.login("op1")
        summary = json.loads(self.request("GET", "/api/consistency", cookie=operator)[2])
        self.assertEqual((summary["ok"], summary["count"]), (False, 1))
        self.assertNotIn("anomalies", summary)  # le détail est réservé à l'admin
        recorded = [e for e in read_entries(self.audit.path) if e["event"] == "anomalie_coherence"]
        self.assertEqual(recorded[-1]["utilisateur"], "admin1")

    def test_modified_csv_incidents_file_and_capture_are_detected(self):
        csv_path = self.data / "alerts_log.csv"
        csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("absence de casque", "rien à signaler"),
                            encoding="utf-8")
        incidents_path = self.data / "live_incidents" / self.CHECKED / "incidents.json"
        incidents = json.loads(incidents_path.read_text(encoding="utf-8"))
        incidents_path.write_text(json.dumps([i for i in incidents if i["id"] != 2]), encoding="utf-8")
        (self.data / "live_incidents" / self.CHECKED / "incident_0001.jpg").write_bytes(b"autre image")
        details = self.details(self.verify_as_admin())
        self.assertIn(f"incident #1 · session {self.CHECKED} · casque — message modifié dans alerts_log.csv", details)
        self.assertIn(f"incident #2 · session {self.CHECKED} · gilet — incident supprimé de incidents.json", details)
        self.assertIn(f"incident #1 · session {self.CHECKED} · casque — capture incident_0001.jpg modifiée", details)

    def test_accounts_changed_outside_the_application_are_detected(self):
        import dashboard.server as server

        path = self.root / "users.json"
        users = json.loads(path.read_text(encoding="utf-8"))
        users["users"]["intrus"] = dict(users["users"]["op1"])     # compte ajouté à la main
        users["users"]["op1"]["role"] = "admin"                     # élévation de privilège à la main
        users["users"]["hse1"]["password"] = hash_password("mot-de-passe-pirate")
        del users["users"]["admin1"]["password"]
        users["users"]["admin1"]["password"] = hash_password(PASSWORD)  # même mot de passe, haché différent
        path.write_text(json.dumps(users), encoding="utf-8")
        server.run_consistency()  # comme au démarrage
        anomalies = [f"{a['objet']} — {a['detail']}" for a in server._CONSISTENCY["report"]["anomalies"]]
        self.assertIn("compte intrus — présent dans users.json sans création journalisée", anomalies)
        self.assertIn("compte op1 — rôle « admin » au lieu de « operateur » (journal)", anomalies)
        self.assertIn("compte hse1 — mot de passe modifié hors de l'application", anomalies)
        self.assertIn("compte admin1 — mot de passe modifié hors de l'application", anomalies)
        self.assertEqual([e for e in read_entries(self.audit.path)][-1]["event"], "anomalie_coherence")

    def test_account_disabled_by_hand_is_detected(self):
        path = self.root / "users.json"
        users = json.loads(path.read_text(encoding="utf-8"))
        users["users"]["hse1"]["active"] = False
        path.write_text(json.dumps(users), encoding="utf-8")
        self.assertIn("compte hse1 — désactivé dans users.json, contrairement au journal",
                      self.details(self.verify_as_admin()))


class AccountAdminHttpTests(LiveHttpTestCase):
    def test_admin_manages_accounts_with_guards(self):
        admin = self.login("admin1")
        status, _, data = self.request("POST", "/api/users/create",
                                       {"username": "op2", "password": PASSWORD, "role": "operateur"}, cookie=admin)
        self.assertEqual((status, json.loads(data)["user"]["role"]), (200, "operateur"))
        for payload, message in (
                ({"username": "op2", "password": PASSWORD, "role": "operateur"}, "existe déjà"),
                ({"username": "op3", "password": "court", "role": "operateur"}, "12 caractères")):
            status, _, data = self.request("POST", "/api/users/create", payload, cookie=admin)
            self.assertEqual(status, 400)
            self.assertIn(message, json.loads(data)["error"])
        status, _, data = self.request("POST", "/api/users/active", {"username": "admin1", "active": False}, cookie=admin)
        self.assertEqual(status, 400)
        self.assertIn("propre compte", json.loads(data)["error"])
        hse = self.login("hse1")
        self.assertEqual(self.request("POST", "/api/users/active", {"username": "hse1", "active": False},
                                      cookie=admin)[0], 200)
        self.assertEqual(self.request("GET", "/api/me", cookie=hse)[0], 401)  # session fermée
        self.assertEqual(self.request("POST", "/api/users/delete", {"username": "op2"}, cookie=admin)[0], 404)
        users = {u["username"]: u for u in json.loads(self.request("GET", "/api/users", cookie=admin)[2])["users"]}
        self.assertEqual(set(users), {"admin1", "hse1", "op1", "op2"})
        self.assertFalse(users["hse1"]["active"])
        self.assertNotIn("password", users["op1"])

    def test_last_active_admin_cannot_be_disabled_over_http(self):
        admin = self.login("admin1")
        self.request("POST", "/api/users/create", {"username": "admin2", "password": PASSWORD, "role": "admin"},
                     cookie=admin)
        second = self.login("admin2")
        self.assertEqual(self.request("POST", "/api/users/active", {"username": "admin1", "active": False},
                                      cookie=second)[0], 200)
        status, _, data = self.request("POST", "/api/users/active", {"username": "admin2", "active": False},
                                       cookie=second)
        self.assertEqual(status, 400)  # propre compte, et dernier admin actif

    def test_audit_page_filters(self):
        admin = self.login("admin1")
        self.login("hse1")
        query = "/api/audit?utilisateur=hse1&evenement=connexion_reussie"
        data = json.loads(self.request("GET", query, cookie=admin)[2])
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["entries"][0]["utilisateur"], "hse1")
        self.assertIn("compte_cree", data["events"])
        self.assertEqual(self.request("GET", "/audit", cookie=admin)[0], 200)

    def test_startup_refuses_without_admin(self):
        from unittest.mock import patch

        import dashboard.server as server

        empty = AuthService(self.root / "aucun.json", audit=self.audit)
        with patch.object(server, "AUTH", empty), self.assertRaises(SystemExit) as error:
            server.startup()
        self.assertIn("manage_users.py create", str(error.exception))



class PortHijackTests(unittest.TestCase):
    """Port déjà occupé : le serveur refuse de démarrer au lieu de partager le port (détournement local)."""

    def setUp(self):
        import dashboard.server as server

        self.server_module = server
        self.servers = []

    def tearDown(self):
        for item in self.servers:
            item.server_close()

    def open(self, server_class, port=0):
        item = server_class(("127.0.0.1", port), self.server_module.DashboardHandler)
        self.servers.append(item)
        return item

    @unittest.skipUnless(os.name == "nt", "comportement propre à Windows")
    def test_vulnerability_old_server_class_shares_the_port_on_windows(self):
        from http.server import ThreadingHTTPServer

        first = self.open(ThreadingHTTPServer)
        second = self.open(ThreadingHTTPServer, first.server_port)  # SO_REUSEADDR : accepté sous Windows
        self.assertEqual(first.server_port, second.server_port)

    def test_hse_server_refuses_a_port_already_in_use(self):
        from http.server import ThreadingHTTPServer

        old = self.open(ThreadingHTTPServer)  # ancien serveur resté ouvert
        with self.assertRaises(self.server_module.PortInUseError) as context:
            self.open(self.server_module.HSEHTTPServer, old.server_port)
        self.assertIn("déjà utilisé", str(context.exception))

    def test_other_program_cannot_take_the_hse_server_port(self):
        from http.server import ThreadingHTTPServer

        hse = self.open(self.server_module.HSEHTTPServer)
        self.assertFalse(hse.allow_reuse_address)
        with self.assertRaises(OSError):
            self.open(ThreadingHTTPServer, hse.server_port)

    def test_run_live_exits_with_clear_message_when_port_is_taken(self):
        from unittest.mock import patch

        import run_live

        occupied = self.open(self.server_module.HSEHTTPServer)
        with (patch.object(run_live, "startup"), patch.object(run_live, "PORT", occupied.server_port),
              patch.object(run_live, "LIVE") as live):
            with self.assertRaises(SystemExit) as context:
                run_live.main()
        self.assertIn(f"port {occupied.server_port} est déjà utilisé", str(context.exception.code))
        live.start_source.assert_not_called()


class RunLiveStartupTests(unittest.TestCase):
    def test_starts_server_on_login_page_without_any_analysis(self):
        from unittest.mock import patch

        import run_live

        self.assertIsNone(run_live.INPUT_VIDEO)
        self.assertFalse(hasattr(run_live, "LOOP"))
        with (patch.object(run_live, "startup") as startup, patch.object(run_live, "PORT", 0),
              patch.object(run_live, "LIVE") as live, patch.object(run_live, "webbrowser") as browser,
              patch.object(run_live.time, "sleep", side_effect=KeyboardInterrupt)):
            live.session = None
            run_live.main()
        startup.assert_called_once()
        live.start_source.assert_not_called()
        live.start.assert_not_called()
        self.assertTrue(browser.open.call_args[0][0].endswith("/login"))


class EventIdTests(unittest.TestCase):
    def test_two_alerts_of_same_frame_person_and_type_get_distinct_event_ids(self):
        import csv

        from app.alerts.alert_manager import AlertManager

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts_log.csv"
            alerts = AlertManager(path, cooldown_frames=0)
            first = alerts.send_alert(3, "Personne #3 : casque absent", 120, alert_type="epi")
            second = alerts.send_alert(3, "Personne #3 : gilet absent", 120, alert_type="epi")
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertNotEqual(first, second)
        self.assertEqual([r["event_id"] for r in rows], [first, second])
        self.assertTrue(first.startswith(rows[0]["timestamp"] + "|120|3|epi|"))


class NoEmojiTests(unittest.TestCase):
    """Ni emoji ni pictogramme Unicode dans l'interface, les messages du serveur et de la console.

    Un état s'affiche par un mot, une pastille CSS ou une icône SVG. Les symboles typographiques
    (→, ≥, ≈, ×, °, ±) restent autorisés. Les messages déjà enregistrés (alerts_log.csv, journal
    d'audit) ne sont pas concernés : ce sont des données, jamais réécrites.
    """

    FORBIDDEN = [
        (0x2300, 0x23FF),    # symboles techniques divers (⌛, ⏱, ⏺…)
        (0x25A0, 0x25FF),    # formes géométriques (●, ■, ▲, ▶…)
        (0x2600, 0x27BF),    # symboles divers et dingbats (⚠, ☀, ✓, ✗, ✔…)
        (0x2B00, 0x2BFF),    # flèches et symboles divers (⬆, ⭐…)
        (0x1F000, 0x1FAFF),  # emojis
        (0xFE0F, 0xFE0F),    # sélecteur de présentation emoji
        (0x200D, 0x200D),    # liaison des emojis composés
    ]

    def test_no_emoji_in_pages_styles_and_messages(self):
        root = Path(__file__).resolve().parent
        files = [*root.glob("dashboard/*.html"), *root.glob("dashboard/assets/*.css"), *root.glob("dashboard/*.py"),
                 *root.glob("app/**/*.py"), *root.glob("tools/**/*.py"), *root.glob("config/*.json"),
                 root / "run_live.py"]
        self.assertTrue(any(path.name == "hse.css" for path in files))
        found = []
        for path in files:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for char in line:
                    if any(low <= ord(char) <= high for low, high in self.FORBIDDEN):
                        found.append(f"{path.relative_to(root)}:{number} U+{ord(char):04X}")
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
