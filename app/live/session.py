"""
app/live/session.py

Analyse en direct d'une caméra ou d'une vidéo utilisée comme caméra simulée.

Deux fils d'exécution :
- lecture : la source est lue à sa cadence réelle (une vidéo n'est jamais
  lue plus vite que sa cadence d'origine) ; seule la dernière image est
  conservée ;
- analyse : traite la dernière image disponible, puis la suivante, etc.
  Les images produites pendant une analyse sont ignorées, comme pour un flux
  de caméra réel sur un poste sans GPU. Les détections apparaissent donc
  progressivement, à la cadence d'analyse effective (affichée).

La première image lue est publiée aussitôt, brute, avec un bandeau « analyse
en cours » dessiné dans l'image : le flux s'affiche sans attendre la première
analyse (≈ 1 s sur CPU), sans qu'une image sans cadre passe pour conforme.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from app.live.pipeline import HSEPipeline


class LiveSession:
    def __init__(self, source, source_label: str, *, detector, machine_states, alerts_path,
                 snapshot_dir, harness_required: bool | None = None, jpeg_quality: int = 80,
                 audit=None):
        self.source = source
        self.source_label = source_label
        self.is_camera = not isinstance(source, (str, Path)) or str(source).startswith(("rtsp://", "http://", "https://"))
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_dir = Path(snapshot_dir) / self.session_id
        self.pipeline = HSEPipeline(
            source_label,
            detector=detector,
            alerts_path=alerts_path,
            snapshot_dir=self.snapshot_dir,
            machine_states=machine_states,
            harness_required=harness_required,
            audit=audit,
            audit_context={"session_id": self.session_id},
        )
        self.jpeg_quality = jpeg_quality

        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._new_frame = threading.Condition(self._frame_lock)
        self._latest = None  # (seq, frame, media_t, frame_idx, capture_time)
        self._output = threading.Condition()
        self._jpeg = None
        self._jpeg_seq = 0
        self._result = {}
        self._stats = {
            "source_frames": 0, "analysed": 0, "skipped": 0,
            "analysis_fps": 0.0, "inference_ms": None, "latency_ms": None,
            "source_fps": None, "media_t": 0.0, "ended": False, "error": None,
            "preview": False,  # image affichée = image brute en attente de la première analyse
        }
        self._threads = []
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    def start(self):
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Impossible d'ouvrir la source : {self.source_label}")
        self._threads = [
            threading.Thread(target=self._read_loop, args=(capture,), name="hse-reader", daemon=True),
            threading.Thread(target=self._analyse_loop, name="hse-analyser", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        with self._output:
            self._output.notify_all()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout)

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    # ------------------------------------------------------------------
    def _read_loop(self, capture):
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and 1 < fps < 240 else 25.0
        self._stats["source_fps"] = round(fps, 1)
        seq = 0
        frame_idx = 0
        started = time.perf_counter()
        try:
            while not self._stop.is_set():
                if not self.is_camera:
                    # Cadence réelle : saute les images dont l'heure est passée, attend les suivantes.
                    target = int((time.perf_counter() - started) * fps)
                    while frame_idx < target - 1 and capture.grab():
                        frame_idx += 1
                    ahead = frame_idx / fps - (time.perf_counter() - started)
                    if ahead > 0:
                        time.sleep(min(ahead, 0.05))
                        continue
                ok, frame = capture.read()
                if not ok:
                    self._stats["ended"] = True
                    break
                frame_idx += 1
                seq += 1
                media_t = (time.perf_counter() - started) if self.is_camera else frame_idx / fps
                with self._new_frame:
                    self._latest = (seq, frame, media_t, frame_idx, time.perf_counter())
                    self._stats["source_frames"] = seq
                    self._new_frame.notify_all()
                if seq == 1:
                    self._publish_preview(frame)
        except Exception as error:  # noqa: BLE001 — remonté à l'interface
            self._stats["error"] = f"Lecture : {error}"
        finally:
            capture.release()
            with self._new_frame:
                self._new_frame.notify_all()

    def _analyse_loop(self):
        last_seq = 0
        durations = []
        try:
            while not self._stop.is_set():
                with self._new_frame:
                    while not self._stop.is_set() and (self._latest is None or self._latest[0] == last_seq):
                        if self._stats["ended"] or self._stats["error"]:
                            return
                        self._new_frame.wait(0.5)
                    if self._stop.is_set():
                        return
                    seq, frame, media_t, frame_idx, captured = self._latest
                self._stats["skipped"] += max(0, seq - last_seq - 1) if last_seq else 0
                last_seq = seq

                started = time.perf_counter()
                result = self.pipeline.process(
                    frame, media_t, frame_idx,
                    fps=self._stats["analysis_fps"],
                    timestamp=datetime.now().strftime("%H:%M:%S") if self.is_camera else f"t = {media_t:.1f} s",
                )
                ok, jpeg = cv2.imencode(".jpg", result["annotated"], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                durations.append(time.perf_counter() - started)
                durations = durations[-10:]
                self._stats.update({
                    "analysed": self._stats["analysed"] + 1,
                    "analysis_fps": round(len(durations) / sum(durations), 2),
                    "inference_ms": result["inference_ms"],
                    "latency_ms": round((time.perf_counter() - captured) * 1000),
                    "media_t": round(media_t, 1),
                })
                with self._output:
                    if ok:
                        self._jpeg = jpeg.tobytes()
                        self._jpeg_seq += 1
                        self._stats["preview"] = False
                    self._result = {k: v for k, v in result.items() if k != "annotated"}
                    self._output.notify_all()
        except Exception as error:  # noqa: BLE001
            self._stats["error"] = f"Analyse : {error}"
            raise

    def _publish_preview(self, frame):
        """Image brute avec bandeau, publiée seulement si aucune image analysée ne l'a précédée."""
        width, height, scale = self.pipeline.display_size(frame.shape[1], frame.shape[0])
        display = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
        ok, jpeg = cv2.imencode(".jpg", self.pipeline.renderer.render_pending(display),
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        with self._output:
            if ok and self._jpeg_seq == 0:
                self._jpeg = jpeg.tobytes()
                self._jpeg_seq = 1
                self._stats["preview"] = True
                self._output.notify_all()

    # ------------------------------------------------------------------
    def wait_jpeg(self, after_seq: int, timeout: float = 2.0):
        """Bloque jusqu'à une nouvelle image (brute puis annotée) ; renvoie (seq, jpeg)."""
        with self._output:
            if self._jpeg_seq <= after_seq and not self._stop.is_set():
                self._output.wait(timeout)
            return self._jpeg_seq, self._jpeg

    def status(self) -> dict:
        with self._output:
            result = dict(self._result)
        persons = [
            {
                "track_id": p["track_id"], "status": p["status"], "items": p["items"],
                "reasons": p["reasons"], "confidences": p["confidences"], "observations": p["observations"],
            }
            for p in result.get("persons", [])
        ]
        situations = [
            {k: s[k] for k in ("track_id", "machine_id", "status", "severity", "distance_rel", "distance_m", "level")}
            for s in result.get("situations", [])
        ]
        incidents = [
            {k: v for k, v in incident.items() if k != "bbox"}
            for incident in reversed(self.pipeline.incidents.snapshot())
        ]
        return {
            "running": self.running and not self._stop.is_set(),
            "session_id": self.session_id,
            "source": self.source_label,
            "camera_name": self.pipeline.profile["name"],
            "is_camera": self.is_camera,
            "harness_required": self.pipeline.harness_required,
            "machine_id": self.pipeline.machine_id,
            "started_at": self.started_at,
            "stats": dict(self._stats),
            "counts": result.get("counts", {}),
            "machine": result.get("machine") or self.pipeline.machine_states.get(self.pipeline.machine_id),
            "unknown_policy": self.pipeline.machine_states.unknown_policy,
            "camera_integrity": result.get("camera") or {"state": "initialisation", "compromised": False, "active": []},
            "persons": persons,
            "situations": situations,
            "incidents": incidents[:100],
            "capabilities": self.pipeline.detector.capabilities(),
        }
