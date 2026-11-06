"""
run_live.py

Démarre le serveur web local et ouvre le navigateur sur /login. Aucune
analyse ne démarre seule : un responsable HSE ou un administrateur choisit
la source dans /live et clique sur « Démarrer l'analyse » (tracé dans le
journal d'audit avec son nom). Les incidents des analyses précédentes
restent affichés dans l'historique.

Le serveur refuse de démarrer si le port est déjà pris (pas de partage de
port : voir HSEHTTPServer dans dashboard/server.py).

Les alertes de toutes les analyses s'affichent aussi dans ce terminal.

Usage :
    python run_live.py
Ctrl+C pour tout arrêter.
"""

import sys
import threading
import time
import webbrowser
from pathlib import Path

# ===========================================================================
# Réglages (facultatifs)
# ===========================================================================
INPUT_VIDEO = None
# None (par défaut) : aucune analyse au lancement, elle se démarre depuis /live.
# Pour un essai local sans passer par l'interface (tracé « systeme » dans l'audit) :
#   INPUT_VIDEO = r"D:\videos\ma_video.mp4"   # n'importe quelle vidéo du PC
#   INPUT_VIDEO = 0                                     # webcam

MACHINE_STATE = None        # None = garder l'état actuel ; sinon "marche", "arret" ou "inconnu"
HARNESS_REQUIRED = False    # True pour une scène de travail en hauteur
PORT = 8000
OPEN_BROWSER = True
# ===========================================================================

from app.config import ROOT  # noqa: E402  (imports après les réglages, pour la lisibilité)
from dashboard.server import LIVE, DashboardHandler, HSEHTTPServer, PortInUseError, startup  # noqa: E402


class QuietHandler(DashboardHandler):
    """Le dashboard interroge le serveur chaque seconde : on n'affiche pas ces requêtes."""

    def log_message(self, format, *args):
        pass


def resolve_input(value):
    if isinstance(value, int) or str(value).isdigit():
        return int(value), f"camera_{value}"
    if str(value).startswith(("rtsp://", "http://", "https://")):
        return str(value), "camera_ip"
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        sys.exit(f"[ERREUR] Vidéo introuvable : {path}")
    return str(path), path.name


def print_summary(status):
    stats = status["stats"]
    print(f"\n[FIN] {status['source']} : {stats['analysed']} image(s) analysée(s), "
          f"{len(status['incidents'])} incident(s).")
    for incident in reversed(status["incidents"]):
        print(f"  #{incident['id']} {incident['label']} — {incident['message']} (durée {incident['duration_s']:.1f} s)")
    if stats["error"]:
        print(f"[ERREUR] {stats['error']}")
    print("[INFO] Dashboard toujours ouvert : démarre une autre analyse depuis /live, ou Ctrl+C pour quitter.\n")


def main():
    startup()  # refuse de démarrer sans compte admin ; contrôle d'intégrité et de cohérence
    first = resolve_input(INPUT_VIDEO) if INPUT_VIDEO is not None else None

    try:
        server = HSEHTTPServer(("127.0.0.1", PORT), QuietHandler)
    except PortInUseError as error:
        sys.exit(f"[ERREUR] {error}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/login"

    if first is not None:
        source, label = first
        print(f"[INFO] Chargement des modèles et ouverture de : {label}")
        LIVE.wait_for_models()  # chargement lancé en arrière-plan par startup()
        status = LIVE.start_source(source, label, harness_required=HARNESS_REQUIRED or None)
        if MACHINE_STATE:
            LIVE.set_machine_state(status["machine_id"], MACHINE_STATE, source="run_live.py")

    print(f"[INFO] Dashboard : {url} (connexion requise ; comptes : python tools/manage_users.py list)")
    if first is None:
        print("[INFO] Aucune analyse en cours : la démarrer depuis /live (responsable HSE ou administrateur).")
    print("[INFO] Ctrl+C pour tout arrêter.\n")
    if OPEN_BROWSER:
        webbrowser.open(url)

    watched = None       # session suivie dans le terminal
    summarized = set()   # sessions dont le bilan a déjà été affiché
    last_print = 0.0
    try:
        while True:
            time.sleep(0.5)
            session = LIVE.session
            if session is None:
                continue
            if session is not watched:
                watched = session
                last_print = time.monotonic()
                print(f"[INFO] Nouvelle analyse : {session.source_label} "
                      f"(engin {session.pipeline.machine_id} : {LIVE.machine_states.get(session.pipeline.machine_id)['state']})")
            if session.running:
                if time.monotonic() - last_print >= 5:
                    last_print = time.monotonic()
                    s = session.status()
                    stats, counts = s["stats"], s["counts"]
                    camera = s["camera_integrity"]
                    camera_text = ("COMPROMISE (" + ", ".join(e["label"] for e in camera["active"]) + ")"
                                   if camera["compromised"] else camera["state"])
                    print(f"[DIRECT] {s['source']} t={stats['media_t']:.1f}s · caméra {camera_text} · "
                          f"{stats['analysis_fps']:.1f} analyses/s · "
                          f"personnes {counts.get('persons', 0)} · violations {counts.get('violations', 0)} · "
                          f"à vérifier {counts.get('a_verifier', 0)} · proximités {counts.get('proximites', 0)} · "
                          f"incidents {len(s['incidents'])}")
            elif session.session_id not in summarized:
                summarized.add(session.session_id)
                LIVE.status()  # sauvegarde les incidents de la vidéo terminée
                print_summary(session.status())
    except KeyboardInterrupt:
        print("\n[INFO] Arrêt demandé.")
    finally:
        LIVE.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
