"""
app/live/video_prep.py

Décoder une vidéo 4K en temps réel occupe à lui seul une grande partie du
CPU (≈ 42 img/s maximum mesurés sur le poste de développement), au
détriment de l'analyse. Même en 1080×1920, le décodage prend 210 à 290 ms de
CPU par seconde de vidéo (mesure du 02/11/2026), alors qu'environ une image
par seconde est analysée. Le modèle analyse de toute façon en 960 px : toute
vidéo dont le plus grand côté dépasse 1280 px est donc remplacée par une
copie réduite à 1280 px, préparée une seule fois et mise en cache dans
data/cache : à l'ajout de la vidéo, ou en arrière-plan au démarrage du
serveur pour les vidéos de data/test_videos pas encore en cache.
"""

import threading
from pathlib import Path

import cv2

from app.config import ROOT

CACHE_PATH = ROOT / "data" / "cache"
# Une conversion à la fois par vidéo : un démarrage d'analyse demandé pendant la
# préparation en arrière-plan (démarrage du serveur, ajout d'une vidéo) attend
# sa fin au lieu de convertir une seconde fois.
_locks = {}
_locks_guard = threading.Lock()
_in_progress = set()


def _lock_for(cached: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(cached, threading.Lock())


def in_progress() -> list[str]:
    """Noms des vidéos en cours de conversion (affiché dans /live)."""
    with _locks_guard:
        return sorted(_in_progress)


def prepare_video(path, max_side: int = 1280, log=print) -> Path:
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max(width, height) <= max_side:
        capture.release()
        return path
    scale = max_side / max(width, height)
    size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
    cached = CACHE_PATH / f"{path.stem}_{size[0]}x{size[1]}.mp4"
    with _lock_for(cached):
        if cached.is_file() and cached.stat().st_mtime >= path.stat().st_mtime:
            capture.release()
            return cached
        with _locks_guard:
            _in_progress.add(path.name)
        try:
            _convert(capture, path, cached, size, log)
        finally:
            with _locks_guard:
                _in_progress.discard(path.name)
    return cached


def _convert(capture, path, cached, size, log):
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(".tmp.mp4")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"[INFO] {path.name} ({width}x{height}) : préparation d'une copie {size[0]}x{size[1]} "
        f"pour l'analyse temps réel (une seule fois)…")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            count += 1
            if count % 200 == 0:
                log(f"        {count}/{total} images")
    finally:
        capture.release()
        writer.release()
    temporary.replace(cached)
