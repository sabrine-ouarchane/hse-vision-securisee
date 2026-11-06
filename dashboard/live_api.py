"""API de la surveillance temps réel (détection des comportements et situations homme-machine)."""

import csv
import json
import os
import re
import threading
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from app.config import CAMERA_PROFILES_PATH, ROOT, load_camera_profile, load_detection_rules, load_json
from app.live.video_prep import in_progress
from app.security.audit_log import AuditLog
from app.sensors.machine_state_provider import MachineStateProvider

TEST_VIDEOS_PATH = ROOT / "data" / "test_videos"
LIVE_SNAPSHOTS_PATH = ROOT / "data" / "outputs" / "live_incidents"
ALERTS_PATH = ROOT / "data" / "outputs" / "alerts_log.csv"
AUDIT_LOG_PATH = ROOT / "data" / "outputs" / "audit_log.jsonl"
# Acquittements et archivages : les incidents eux-mêmes ne sont jamais modifiés ni supprimés
REVIEWS_PATH = ROOT / "data" / "outputs" / "incident_reviews.json"
SYSTEM = {"username": "systeme", "role": "systeme"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_CAMERA_INDEX = 3
MAX_UPLOAD_BYTES = 4 * 1024 ** 3
INCIDENTS_FILE = "incidents.json"


class ModelsLoadingError(RuntimeError):
    """Démarrage demandé pendant le chargement des modèles (réponse 409, l'interface réessaie)."""


def save_uploaded_video(filename, stream, length):
    """Enregistre une vidéo envoyée depuis le dashboard dans data/test_videos."""
    if not isinstance(filename, str) or not filename:
        raise ValueError("Nom de fichier manquant.")
    name = Path(filename.replace("\\", "/")).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Format non pris en charge ({suffix or 'sans extension'}) : mp4, avi, mov ou mkv.")
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")  # vidéo -> video
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:80] or "video"
    if not 0 < length <= MAX_UPLOAD_BYTES:
        raise ValueError("Fichier vide ou trop volumineux (4 Go maximum).")
    TEST_VIDEOS_PATH.mkdir(parents=True, exist_ok=True)
    target = TEST_VIDEOS_PATH / f"{stem}{suffix}"
    index = 2
    while target.exists():
        target = TEST_VIDEOS_PATH / f"{stem}_{index}{suffix}"
        index += 1
    temporary = TEST_VIDEOS_PATH / f".upload_{uuid.uuid4().hex}.part"
    try:
        remaining = length
        with temporary.open("wb") as output:
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Envoi interrompu.")
                output.write(chunk)
                remaining -= len(chunk)
        check_readable_video(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target.name


def check_readable_video(path):
    """
    L'extension ne prouve rien : le fichier doit être une vidéo qu'OpenCV sait
    ouvrir et dont il lit au moins une image. Sinon il est refusé (et supprimé
    par l'appelant).
    """
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read() if capture.isOpened() else (False, None)
    finally:
        capture.release()
    if not ok or frame is None or frame.size == 0:
        raise ValueError("Fichier refusé : ce n'est pas une vidéo lisible (aucune image décodable).")


def list_sources():
    sources = []
    if TEST_VIDEOS_PATH.is_dir():
        for path in sorted(TEST_VIDEOS_PATH.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                profile = load_camera_profile(path.name)
                sources.append({
                    "id": f"video:{path.name}",
                    "label": profile.get("name", path.stem),
                    "filename": path.name,
                    "kind": "video",
                    "machine_id": profile.get("machine_id"),
                    "harness_required": bool(profile.get("harness_required")),
                })
    for index in range(MAX_CAMERA_INDEX + 1):
        sources.append({"id": f"camera:{index}", "label": f"Caméra locale {index}", "kind": "camera"})
    if os.environ.get("HSE_CAMERA_URL"):
        sources.append({"id": "camera:url", "label": "Caméra IP (HSE_CAMERA_URL)", "kind": "camera"})
    return sources


def resolve_source(source_id):
    """Ne résout que des sources connues : aucun chemin arbitraire n'est accepté."""
    if not isinstance(source_id, str):
        raise ValueError("Source invalide.")
    kind, _, value = source_id.partition(":")
    if kind == "video":
        if Path(value).name != value or Path(value).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Vidéo invalide.")
        path = TEST_VIDEOS_PATH / value
        if not path.is_file():
            raise ValueError("Vidéo introuvable dans data/test_videos.")
        return str(path), value
    if kind == "camera":
        if value == "url" and os.environ.get("HSE_CAMERA_URL"):
            return os.environ["HSE_CAMERA_URL"], "camera_ip"
        if value.isdigit() and 0 <= int(value) <= MAX_CAMERA_INDEX:
            return int(value), f"camera_{value}"
    raise ValueError("Source inconnue.")


class LiveController:
    """Une seule session d'analyse à la fois ; les modèles sont chargés une fois."""

    def __init__(self):
        self._lock = threading.Lock()
        self._detector = None
        # Chargement des modèles : « non_demarre » (chargé au premier démarrage d'analyse),
        # « chargement » (fil lancé par preload_models au démarrage du serveur), « pret » ou « erreur ».
        self._models_lock = threading.Lock()
        self._models = {"state": "non_demarre", "error": None, "duration_s": None}
        self._models_done = threading.Event()  # levé à la fin d'un chargement (réussi ou non)
        self.session = None
        self.machine_states = MachineStateProvider()
        # Incidents des vidéos précédentes, du plus récent au plus ancien ;
        # rechargés depuis data/outputs/live_incidents au démarrage du serveur.
        self.history = self._load_history()
        self._history_lock = threading.Lock()
        # Journal d'audit chaîné et signé (tools/verify_audit.py)
        self.audit = AuditLog(AUDIT_LOG_PATH)
        self._reviews_lock = threading.Lock()
        if self.machine_states.initialized_from:
            self._log("etat_machines_initialise", {"depuis": self.machine_states.initialized_from,
                                                   "politique": self.machine_states.unknown_policy})

    def _log(self, event, data, user=None):
        user = user or SYSTEM
        self.audit.append(event, data, utilisateur=user["username"], role=user["role"])

    @classmethod
    def _load_history(cls):
        history = []
        if LIVE_SNAPSHOTS_PATH.is_dir():
            cls._recover_sessions_from_log()
            for path in sorted(LIVE_SNAPSHOTS_PATH.glob(f"*/{INCIDENTS_FILE}"), reverse=True):
                try:
                    history.extend(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        return history

    @staticmethod
    def _recover_sessions_from_log():
        """
        Les sessions analysées avant l'ajout de l'historique n'ont pas de
        incidents.json. Leurs alertes figurent toutefois dans alerts_log.csv
        et leurs captures dans leur dossier : on reconstitue leur historique
        (chaque alerte est rattachée à la dernière session démarrée avant elle).
        """
        sessions = []
        for directory in LIVE_SNAPSHOTS_PATH.iterdir():
            try:
                sessions.append((datetime.strptime(directory.name, "%Y%m%d_%H%M%S"), directory))
            except ValueError:
                continue
        missing = {d for _, d in sessions if not (d / INCIDENTS_FILE).exists() and any(d.glob("incident_*.jpg"))}
        if not missing or not ALERTS_PATH.is_file():
            return
        from app.alerts.incident_manager import KIND_LABELS

        keywords = [("casque", "casque"), ("gilet", "gilet"), ("harnais", "harnais"),
                    ("téléphone", "telephone"), ("proximité", "proximite")]
        sessions.sort()
        recovered = {d: [] for d in missing}
        # Captures associées par horodatage (le journal peut ne pas contenir toutes les alertes)
        captures = {
            d: [(datetime.fromtimestamp(f.stat().st_mtime), f.name) for f in sorted(d.glob("incident_*.jpg"))]
            for d in missing
        }
        with ALERTS_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r.get("source") == "vision-temps-reel"]
        for row in rows:
            try:
                timestamp = datetime.fromisoformat(row["timestamp"]).replace(tzinfo=None)
            except (TypeError, ValueError):
                continue
            owner = next((d for start, d in reversed(sessions) if start <= timestamp), None)
            if owner not in recovered:
                continue
            message = row.get("message") or ""
            match = re.match(r"\[(.+?)\] (.*)", message)
            camera, text = (match.group(1), match.group(2)) if match else ("Caméra", message)
            kind = next((k for word, k in keywords if word in text.lower()), "casque")
            number = len(recovered[owner]) + 1
            nearest = min(captures[owner], key=lambda c: abs((c[0] - timestamp).total_seconds()), default=None)
            snapshot = None
            if nearest is not None and abs((nearest[0] - timestamp).total_seconds()) <= 2:
                snapshot = nearest[1]
                captures[owner].remove(nearest)
            try:
                confidence = float(row.get("confidence") or "") or None
            except ValueError:
                confidence = None
            recovered[owner].append({
                "id": number, "kind": kind, "label": KIND_LABELS[kind],
                "severity": row.get("severity") or "critique", "message": text,
                "track_ids": [int(t) for t in re.findall(r"#(\d+)", text)[:1]],
                "started_t": 0.0, "last_t": 0.0, "ended_t": 0.0, "duration_s": 0.0,
                "frame_idx": row.get("frame_idx"), "wall_time": row["timestamp"],
                "peak_confidence": confidence, "status": "terminee",
                "snapshot": snapshot,
                "source": camera, "session_id": owner.name, "video": camera,
                "camera_name": camera, "recovered": True,
            })
        for directory, incidents in recovered.items():
            if incidents:
                (directory / INCIDENTS_FILE).write_text(
                    json.dumps(incidents[::-1], ensure_ascii=False, indent=1), encoding="utf-8"
                )

    @staticmethod
    def _session_incidents(session):
        status = session.status()
        incidents = []
        for incident in status["incidents"]:
            entry = dict(incident, session_id=status["session_id"], video=status["source"],
                         camera_name=status["camera_name"])
            if not status["running"] and entry["status"] == "active":
                entry["status"] = "terminee"
            incidents.append(entry)
        return incidents

    def _archive(self, session, user=None):
        """Sauvegarde les incidents d'une session (appelé à l'arrêt ou au changement de vidéo)."""
        first = not getattr(session, "audit_closed", False)
        if first:
            session.audit_closed = True
            session.pipeline.incidents.close_all()  # durée des incidents encore ouverts journalisée
        incidents = self._session_incidents(session)
        if first:
            stats = session.status()["stats"]
            ended = stats["ended"] or stats["error"]
            self._log("session_arretee", {
                "session_id": session.session_id, "source": session.source_label,
                "motif": "fin de la vidéo" if stats["ended"] else ("erreur" if stats["error"] else "arrêt demandé"),
                "images_analysees": stats["analysed"], "incidents": len(incidents),
            }, None if ended else user)
        with self._history_lock:
            history = [i for i in self.history if i["session_id"] != session.session_id]
            self.history = incidents + history
            if incidents:
                session.snapshot_dir.mkdir(parents=True, exist_ok=True)
                (session.snapshot_dir / INCIDENTS_FILE).write_text(
                    json.dumps(incidents, ensure_ascii=False, indent=1), encoding="utf-8"
                )

    def all_incidents(self):
        session = self.session
        current = self._session_incidents(session) if session is not None else []
        current_id = session.session_id if session is not None else None
        incidents = current + [i for i in self.history if i["session_id"] != current_id]
        reviews = self._load_reviews()
        return [dict(i, review=reviews.get(f"{i['session_id']}:{i['id']}")) for i in incidents]

    # ------------------------------------------------------------------
    # Acquittement et archivage (sans suppression)
    @staticmethod
    def _load_reviews():
        if not REVIEWS_PATH.is_file():
            return {}
        try:
            return json.loads(REVIEWS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _review(self, session_id, incident_id, user, update):
        if not isinstance(session_id, str) or not isinstance(incident_id, int) or isinstance(incident_id, bool):
            raise ValueError("Incident invalide.")
        incident = next((i for i in self.all_incidents()
                         if i["session_id"] == session_id and i["id"] == incident_id), None)
        if incident is None:
            raise LookupError("Incident introuvable.")
        key = f"{session_id}:{incident_id}"
        with self._reviews_lock:
            reviews = self._load_reviews()
            review = reviews.get(key, {})
            update(review, incident)
            reviews[key] = review
            REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = REVIEWS_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(reviews, ensure_ascii=False, indent=1), encoding="utf-8")
            temporary.replace(REVIEWS_PATH)
        return dict(incident, review=review)

    def acknowledge(self, session_id, incident_id, user):
        def update(review, incident):
            if review.get("acquitte_par"):
                raise ValueError(f"Déjà acquitté par {review['acquitte_par']}.")
            review.update(acquitte_par=user["username"], acquitte_role=user["role"],
                          acquitte_le=datetime.now().astimezone().isoformat(timespec="seconds"))
            self._log("incident_acquitte", {"session_id": session_id, "incident_id": incident_id,
                                            "type": incident["kind"], "message": incident["message"]}, user)
        return self._review(session_id, incident_id, user, update)

    def archive_incident(self, session_id, incident_id, motif, user):
        motif = motif.strip() if isinstance(motif, str) else ""
        if len(motif) < 5:
            raise ValueError("Motif d'archivage obligatoire (5 caractères minimum).")
        if len(motif) > 500:
            raise ValueError("Motif d'archivage trop long (500 caractères maximum).")

        def update(review, incident):
            if review.get("archive_par"):
                raise ValueError(f"Déjà archivé par {review['archive_par']}.")
            review.update(archive_par=user["username"], archive_role=user["role"], motif_archivage=motif,
                          archive_le=datetime.now().astimezone().isoformat(timespec="seconds"))
            self._log("incident_archive", {"session_id": session_id, "incident_id": incident_id,
                                           "type": incident["kind"], "motif": motif}, user)
        return self._review(session_id, incident_id, user, update)

    # ------------------------------------------------------------------
    # Modèles : chargés une fois, en arrière-plan dès le démarrage du serveur
    def preload_models(self):
        """
        Import de torch, vérification SHA-256, chargement et préchauffage de YOLO
        (≈ 15 s sur le poste de développement) dans un fil séparé : le serveur
        répond pendant ce temps et /live affiche « Chargement des modèles… ».
        """
        with self._models_lock:
            if self._models["state"] in ("chargement", "pret"):
                return
            self._models = {"state": "chargement", "error": None, "duration_s": None}
            self._models_done.clear()
        threading.Thread(target=self._load_models, name="hse-modeles", daemon=True).start()

    def _load_models(self):
        started = time.perf_counter()
        try:
            from app.detection.hse_detector import HSEDetector  # import lourd (torch)
            detector = HSEDetector(load_detection_rules())
            detector.warmup()
        except Exception as error:  # noqa: BLE001 — affiché dans /live, redonné au prochain démarrage
            with self._models_lock:
                self._models = {"state": "erreur", "error": str(error), "duration_s": None}
            print(f"[MODÈLES] Chargement impossible : {error}")
        else:
            with self._models_lock:
                self._detector = detector
                self._models = {"state": "pret", "error": None,
                                "duration_s": round(time.perf_counter() - started, 1)}
        finally:
            self._models_done.set()

    def wait_for_models(self, timeout=None):
        """Attend la fin d'un chargement lancé par preload_models (run_live.py avec INPUT_VIDEO)."""
        if self.models_status()["state"] == "chargement":
            self._models_done.wait(timeout)

    def models_status(self):
        with self._models_lock:
            return dict(self._models)

    def _get_detector(self):
        """
        Modèles déjà chargés : empreintes revérifiées à chaque démarrage d'analyse.
        Chargement en cours : refus (l'interface attend « pret »). Jamais lancé
        (sans serveur : tests, scripts) ou en erreur : chargement immédiat, qui
        redonne l'erreur s'il y a lieu.
        """
        with self._models_lock:
            state, detector = self._models["state"], self._detector
        if state == "chargement":
            raise ModelsLoadingError("Chargement des modèles en cours : réessayer dans quelques secondes.")
        if detector is not None:
            try:
                detector.verify_models()
            except Exception as error:
                with self._models_lock:  # poids modifiés depuis leur chargement : plus jamais utilisés
                    self._detector = None
                    self._models = {"state": "erreur", "error": str(error), "duration_s": None}
                raise
            return detector
        with self._models_lock:
            self._models = {"state": "chargement", "error": None, "duration_s": None}
            self._models_done.clear()
        self._load_models()
        with self._models_lock:
            if self._detector is None:
                raise RuntimeError(self._models["error"])
            return self._detector

    # ------------------------------------------------------------------
    # Copies 1280 px (app/live/video_prep.py), préparées hors du clic « Démarrer l'analyse »
    def prepare_videos(self, names=None):
        """
        Convertit en arrière-plan les vidéos de data/test_videos pas encore en cache
        (toutes au démarrage du serveur, ou celles données après un ajout), dans un fil
        distinct de celui des modèles. Un démarrage demandé avant la fin attend la
        conversion de sa vidéo (bouton « Préparation de la vidéo… »).
        """
        from app.live.video_prep import prepare_video

        if names is None:
            names = [s["filename"] for s in list_sources() if s["kind"] == "video"]
        paths = [TEST_VIDEOS_PATH / name for name in names]

        def run():
            for path in paths:
                try:
                    prepare_video(path)
                except Exception as error:  # noqa: BLE001 — refaite au démarrage de l'analyse
                    print(f"[VIDÉO] Préparation impossible de {path.name} : {error}")

        thread = threading.Thread(target=run, name="hse-videos", daemon=True)
        thread.start()
        return thread

    def start(self, source_id, harness_required=None, user=None):
        """Démarrage depuis l'interface : seules les sources connues sont acceptées."""
        source, label = resolve_source(source_id)
        return self.start_source(source, label, harness_required, user)

    def start_source(self, source, label, harness_required=None, user=None):
        """Démarrage local (script run_live.py) : fichier quelconque, caméra ou URL RTSP."""
        from app.live.session import LiveSession
        from app.live.video_prep import prepare_video

        if harness_required is not None and not isinstance(harness_required, bool):
            raise ValueError("harness_required doit être un booléen.")
        detector = self._get_detector()  # avant d'arrêter l'analyse en cours : un refus la laisse tourner
        if isinstance(source, str) and Path(source).is_file():
            # attend la copie 1280 px si elle est en cours en arrière-plan ; le nom d'origine (label) garde le profil
            source = str(prepare_video(source))
        with self._lock:
            if self.session is not None:
                self.session.stop()
                self._archive(self.session, user)
            session = LiveSession(
                source, label,
                detector=detector,
                machine_states=self.machine_states,
                alerts_path=ALERTS_PATH,
                snapshot_dir=LIVE_SNAPSHOTS_PATH,
                harness_required=harness_required,
                audit=self.audit,
            )
            session.start()
            self.session = session
        status = session.status()
        self._log("session_demarree", {
            "session_id": status["session_id"], "source": label, "camera": status["camera_name"],
            "harnais_exige": status["harness_required"],
        }, user)
        return status

    def stop(self, user=None):
        with self._lock:
            if self.session is not None:
                self.session.stop()
                self._archive(self.session, user)

    def status(self):
        session = self.session
        status = {"running": False, "session_id": None} if session is None else session.status()
        status["models"] = self.models_status()
        status["videos_en_preparation"] = in_progress()
        if (session is not None and not status["running"] and status["stats"]["ended"]
                and not getattr(session, "archived", False)):
            self._archive(session)  # vidéo terminée : incidents sauvegardés sur disque
            session.archived = True
        status["all_incidents"] = self.all_incidents()[:300]
        return status

    def set_machine_state(self, machine_id, state, unknown_policy=None, source=None, user=None):
        """État déclaré ; « déclaré par <nom> » affiché dans /live, ancien → nouveau dans l'audit."""
        before = self.machine_states.get(machine_id)["state"] if isinstance(machine_id, str) else None
        policy_before = self.machine_states.unknown_policy
        source = source or (user["username"] if user else "interface opérateur")
        result = self.machine_states.set_state(machine_id, state, source=source, unknown_policy=unknown_policy)
        self._log("etat_machine_modifie", {
            "machine_id": machine_id, "avant": before, "apres": state,
            "politique_avant": policy_before, "politique_apres": self.machine_states.unknown_policy, "source": source,
        }, user)
        return result

    def add_uploaded_video(self, filename, stream, length, user=None):
        """Vidéo envoyée depuis le dashboard : contrôlée, enregistrée, journalisée."""
        try:
            saved = save_uploaded_video(filename, stream, length)
        except ValueError as error:
            self._log("video_refusee", {"nom_fourni": str(filename)[:200], "taille": length, "motif": str(error)}, user)
            raise
        self._log("video_ajoutee", {"fichier": saved, "nom_fourni": str(filename)[:200], "taille": length}, user)
        self.prepare_videos([saved])  # copie 1280 px dès l'ajout, pas au clic
        return saved

    def zone_presence(self):
        """
        Présence par zone du site (page /zones), d'après les caméras associées à une zone
        (« zone_id » dans camera_profiles.json). /live n'analyse qu'une source à la fois :
        seule la zone de cette source a une présence connue ; les autres sont « inconnue ».
        """
        profiles = load_json(CAMERA_PROFILES_PATH)
        presence = {}
        for name, profile in profiles.get("sources", {}).items():
            if profile.get("zone_id"):
                presence.setdefault(profile["zone_id"], {
                    "etat": "inconnue", "proximite": False, "camera_compromise": False,
                    "source": load_camera_profile(name).get("name", name),
                })
        session = self.session
        if session is None or not session.running:
            return presence
        zone_id = session.pipeline.profile.get("zone_id")
        status = session.status()
        if not zone_id or not status["stats"].get("analysed"):
            return presence
        integrity = status["camera_integrity"]
        presence[zone_id] = {
            "etat": "presente" if status["counts"].get("persons") else "absente",
            "proximite": any(s["severity"] is not None for s in status["situations"]),
            "camera_compromise": bool(integrity.get("compromised")),
            "source": status["camera_name"],
            "analysee": True,
        }
        return presence

    def machine_overview(self, machine_ids):
        return {
            "unknown_policy": self.machine_states.unknown_policy,
            "machines": [self.machine_states.get(machine_id) for machine_id in machine_ids],
        }

    def snapshot_path(self, session_id, filename):
        if (not isinstance(session_id, str) or not session_id.replace("_", "").isdigit()
                or Path(filename).name != filename or not filename.endswith(".jpg")):
            return None
        path = LIVE_SNAPSHOTS_PATH / session_id / filename
        return path if path.is_file() else None
