"""
test_live_startup.py

Délai avant la première image dans /live (diagnostic du 02/11/2026) :
- modèles chargés une seule fois, en arrière-plan au démarrage du serveur,
  empreintes SHA-256 revérifiées à chaque démarrage d'analyse ;
- première image brute publiée tout de suite, avec un bandeau « analyse en
  cours » dessiné dans l'image ;
- copie 1280 px en cache pour toute vidéo dont le plus grand côté dépasse 1280.

Aucun vrai modèle n'est chargé : un faux détecteur remplace app.detection.hse_detector.
"""

import io
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from test_auth import LiveHttpTestCase


def write_video(path, width, height, frames=12, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(frames):
        frame = np.full((height, width, 3), 90, np.uint8)
        cv2.circle(frame, (40 + 10 * index, height // 3), 25, (230, 230, 230), -1)
        writer.write(frame)
    writer.release()
    return path


class FakeDetector:
    """Remplace HSEDetector : chargement bloqué tant que `release` n'est pas levé, analyse lente."""

    release = threading.Event()
    integrity_error = None
    analysis_s = 0.0

    def __init__(self, rules):
        self.verified = 0
        if not FakeDetector.release.wait(10):
            raise RuntimeError("chargement factice jamais terminé")

    def warmup(self):
        pass

    def verify_models(self):
        self.verified += 1
        if FakeDetector.integrity_error:
            raise RuntimeError(FakeDetector.integrity_error)

    def reset(self):
        pass

    def detect(self, frame):
        time.sleep(FakeDetector.analysis_s)
        return []

    def capabilities(self):
        return {}


def fake_detector_module():
    FakeDetector.release.clear()
    FakeDetector.integrity_error = None
    module = types.ModuleType("app.detection.hse_detector")
    module.HSEDetector = FakeDetector
    return patch.dict(sys.modules, {"app.detection.hse_detector": module})


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
    return condition()


class ModelPreloadTests(LiveHttpTestCase):
    def setUp(self):
        super().setUp()
        modules = fake_detector_module()
        modules.start()
        self.patches.append(modules)
        videos = self.root / "data" / "videos"
        videos.mkdir(parents=True)
        write_video(videos / "test.mp4", 320, 240)

    def tearDown(self):
        FakeDetector.release.set()
        self.live.stop()
        super().tearDown()

    def test_start_is_refused_while_models_load_then_accepted(self):
        self.live.preload_models()
        cookie = self.login("hse1")
        status = json.loads(self.request("GET", "/api/live/status", cookie=cookie)[2])
        self.assertEqual(status["models"]["state"], "chargement")
        code, _, body = self.request("POST", "/api/live/start", {"source": "video:test.mp4"}, cookie=cookie)
        self.assertEqual(code, 409)
        self.assertIn("Chargement des modèles en cours", json.loads(body)["error"])
        self.assertIsNone(self.live.session)  # rien n'a démarré

        FakeDetector.release.set()
        self.assertTrue(wait_until(lambda: self.live.models_status()["state"] == "pret"))
        code, _, body = self.request("POST", "/api/live/start", {"source": "video:test.mp4"}, cookie=cookie)
        self.assertEqual(code, 200, body)
        self.assertEqual(self.live._detector.verified, 1)  # empreintes revérifiées au démarrage

    def test_models_are_loaded_once_and_verified_at_each_start(self):
        FakeDetector.release.set()
        self.live.preload_models()
        self.live.wait_for_models(5)
        detector = self.live._detector
        for _ in range(2):
            self.live.start("video:test.mp4")
        self.assertIs(self.live._detector, detector)  # pas de second chargement
        self.assertEqual(detector.verified, 2)

    def test_modified_weights_after_loading_block_the_start(self):
        FakeDetector.release.set()
        self.live.preload_models()
        self.live.wait_for_models(5)
        FakeDetector.integrity_error = "Modèle models/best.pt MODIFIÉ"
        with self.assertRaisesRegex(RuntimeError, "MODIFIÉ"):
            self.live.start("video:test.mp4")
        self.assertIsNone(self.live.session)
        self.assertIsNone(self.live._detector)  # modèle en mémoire écarté
        self.assertEqual(self.live.models_status()["state"], "erreur")

    def test_without_preload_models_load_at_first_start(self):
        FakeDetector.release.set()
        self.assertEqual(self.live.models_status()["state"], "non_demarre")
        self.live.start("video:test.mp4")
        self.assertEqual(self.live.models_status()["state"], "pret")


class VideoPreparationTests(LiveHttpTestCase):
    """Copie 1280 px faite hors du clic : au démarrage du serveur et à l'ajout d'une vidéo."""

    def setUp(self):
        from app.live import video_prep

        super().setUp()
        for item in (fake_detector_module(), patch.object(video_prep, "CACHE_PATH", self.root / "cache")):
            item.start()
            self.patches.append(item)
        self.conversions = []
        original = video_prep._convert

        def slow_convert(capture, path, *args):
            self.conversions.append(path.name)
            time.sleep(0.6)  # laisse le temps de cliquer pendant la conversion
            original(capture, path, *args)

        item = patch.object(video_prep, "_convert", slow_convert)
        item.start()
        self.patches.append(item)
        self.videos = self.root / "data" / "videos"
        self.videos.mkdir(parents=True)
        FakeDetector.release.set()

    def tearDown(self):
        self.live.stop()
        super().tearDown()

    def test_click_during_background_preparation_waits_without_converting_twice(self):
        write_video(self.videos / "vertical.mp4", 1080, 1920, frames=3)
        write_video(self.videos / "petite.mp4", 320, 240, frames=3)
        self.live.preload_models()
        thread = self.live.prepare_videos()  # comme au démarrage du serveur
        self.assertTrue(wait_until(lambda: "vertical.mp4" in self.live.status()["videos_en_preparation"]))
        self.live.wait_for_models(5)
        status = self.live.start("video:vertical.mp4")  # clic avant la fin : attend la conversion
        thread.join(5)
        self.assertEqual(self.conversions, ["vertical.mp4"])  # une seule conversion, petite vidéo inchangée
        self.assertEqual(Path(self.live.session.source).name, "vertical_720x1280.mp4")
        self.assertTrue(status["running"])
        self.assertEqual(self.live.status()["videos_en_preparation"], [])

    def test_uploaded_video_is_prepared_right_away(self):
        source = write_video(self.root / "envoi.mp4", 1080, 1920, frames=3)
        data = source.read_bytes()
        saved = self.live.add_uploaded_video("envoi.mp4", io.BytesIO(data), len(data))
        self.assertTrue(wait_until(lambda: (self.root / "cache" / "envoi_720x1280.mp4").is_file()))
        self.assertEqual(self.conversions, [saved])


class FirstImageTests(unittest.TestCase):
    def setUp(self):
        from app.live.session import LiveSession
        from app.sensors.machine_state_provider import MachineStateProvider

        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        FakeDetector.release.set()
        FakeDetector.integrity_error = None
        FakeDetector.analysis_s = 0.8  # première analyse lente, comme sur CPU
        self.session = LiveSession(
            str(write_video(root / "test.mp4", 640, 480, frames=40)), "test.mp4", detector=FakeDetector({}),
            machine_states=MachineStateProvider(root / "machines.json"), alerts_path=root / "alerts.csv",
            snapshot_dir=root / "captures", audit=None)

    def tearDown(self):
        self.session.stop()
        FakeDetector.analysis_s = 0.0
        self.directory.cleanup()

    def test_raw_image_with_banner_is_sent_before_the_first_analysis(self):
        from app.visualization.overlay import COLORS, LABEL_BACKGROUND

        started = time.perf_counter()
        self.session.start()
        seq, jpeg = self.session.wait_jpeg(0, timeout=2.0)
        self.assertLess(time.perf_counter() - started, 0.6)  # bien avant la fin de l'analyse (0,8 s)
        self.assertEqual(seq, 1)
        self.assertTrue(self.session.status()["stats"]["preview"])
        self.assertEqual(self.session.status()["stats"]["analysed"], 0)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        band = np.array(LABEL_BACKGROUND[COLORS["a_verifier"]], np.int16)
        middle = image[image.shape[0] // 2, image.shape[1] - 5].astype(np.int16)  # bord droit du bandeau
        self.assertLess(np.abs(middle - band).max(), 20)

        seq, _ = self.session.wait_jpeg(1, timeout=3.0)  # puis l'image annotée
        self.assertEqual(seq, 2)
        self.assertFalse(self.session.status()["stats"]["preview"])

    def test_banner_fits_a_vertical_video(self):
        from app.visualization.overlay import OverlayRenderer

        frame = np.full((1280, 720, 3), 40, np.uint8)
        image = OverlayRenderer().render_pending(frame.copy())
        self.assertEqual(image.shape, frame.shape)
        white = (image[600:680] > 235).all(axis=2)  # texte du bandeau, au centre de l'image
        self.assertTrue(white.any())
        self.assertFalse(white[:, -12:].any())  # marge droite intacte : le titre tient dans la largeur


class VideoCacheTests(unittest.TestCase):
    def setUp(self):
        from app.live import video_prep

        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cache = patch.object(video_prep, "CACHE_PATH", self.root / "cache")
        self.cache.start()
        self.prepare = video_prep.prepare_video

    def tearDown(self):
        self.cache.stop()
        self.directory.cleanup()

    def test_vertical_1080p_video_is_cached_at_1280(self):
        source = write_video(self.root / "vertical.mp4", 1080, 1920, frames=3)
        prepared = self.prepare(source, log=lambda *a: None)
        self.assertEqual(prepared.parent, self.root / "cache")
        capture = cv2.VideoCapture(str(prepared))
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        self.assertEqual(size, (720, 1280))
        self.assertEqual(self.prepare(source, log=lambda *a: None), prepared)  # réutilisée ensuite

    def test_video_up_to_1280_is_used_as_is(self):
        source = write_video(self.root / "hd.mp4", 1280, 720, frames=3)
        self.assertEqual(self.prepare(source, log=lambda *a: None), source)


if __name__ == "__main__":
    unittest.main()
