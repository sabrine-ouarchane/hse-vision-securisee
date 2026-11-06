import unittest

import cv2
import numpy as np

from app.config import load_detection_rules
from app.security.tamper_detector import TamperDetector, apply_prudence, dhash, hamming

RULES = load_detection_rules()
RNG = np.random.default_rng(7)
# Scène synthétique texturée, plus grande que l'image : permet de simuler un
# panoramique (scène en mouvement) en déplaçant la fenêtre de cadrage.
CANVAS = np.full((900, 1400, 3), 110, np.uint8)
for _ in range(220):
    x, y = int(RNG.integers(0, 1350)), int(RNG.integers(0, 850))
    color = tuple(int(c) for c in RNG.integers(20, 235, 3))
    if RNG.random() < 0.5:
        cv2.rectangle(CANVAS, (x, y), (x + int(RNG.integers(15, 90)), y + int(RNG.integers(15, 90))), color, -1)
    else:
        cv2.circle(CANVAS, (x, y), int(RNG.integers(8, 45)), color, -1)
WIDTH, HEIGHT = 640, 480
HZ = 2.0  # cadence d'analyse réaliste en temps réel sur CPU


def scene(t, pan=0.0, moving=True):
    """Image de la caméra au temps t : bruit de capteur et petit objet mobile."""
    x0 = int(100 + pan * t) % (CANVAS.shape[1] - WIDTH)
    frame = CANVAS[100:100 + HEIGHT, x0:x0 + WIDTH].copy()
    if moving:
        cv2.circle(frame, (int(80 + 40 * t) % WIDTH, 400), 18, (240, 240, 240), -1)
    noise = np.random.default_rng(int(t * 1000)).normal(0, 3, frame.shape)
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


def run(detector, frame_at, start, end):
    results = []
    t = start
    while t < end:
        results.append(detector.update(frame_at(t), t))
        t = round(t + 1 / HZ, 3)
    return results


def confirmed(results):
    return {event["type"] for result in results for event in result["active"]}


class TamperDetectorTests(unittest.TestCase):
    def detector(self, **profile):
        return TamperDetector(RULES["tamper"], profile)

    def test_clean_stream_stays_reliable(self):
        results = run(self.detector(), scene, 0, 20)
        self.assertEqual(confirmed(results), set())
        self.assertEqual(results[0]["state"], "initialisation")
        self.assertEqual(results[-1]["state"], "fiable")
        self.assertEqual(results[-1]["reference"], "session")

    def test_worker_passing_in_front_is_ignored(self):
        def frame_at(t):
            return np.full((HEIGHT, WIDTH, 3), 12, np.uint8) if 6 <= t < 7.5 else scene(t)
        self.assertEqual(confirmed(run(self.detector(), frame_at, 0, 15)), set())

    def test_lens_covered_is_confirmed_then_ended_with_duration(self):
        detector = self.detector()
        def frame_at(t):
            return np.full((HEIGHT, WIDTH, 3), 8, np.uint8) if 5 <= t < 15 else scene(t)
        results = run(detector, frame_at, 0, 20)
        self.assertEqual(confirmed(results), {"obstruction"})  # pas de « rejeu » parasite d'images noires
        first = next(r for r in results if r["compromised"])
        self.assertLessEqual(first["active"][0]["since"], 5.5)
        self.assertEqual(results[-1]["state"], "fiable")
        self.assertEqual(detector.events[0]["type"], "obstruction")
        self.assertAlmostEqual(detector.events[0]["duration_s"], 9.5, delta=1.0)

    def test_partial_mask_is_obstruction(self):
        def frame_at(t):
            frame = scene(t)
            if t >= 5:
                frame[:, :int(WIDTH * 0.65)] = 0  # rectangle noir sur 65 % de l'image
            return frame
        self.assertIn("obstruction", confirmed(run(self.detector(), frame_at, 0, 12)))

    def test_camera_moved_is_detected_only_for_fixed_camera(self):
        def frame_at(t):
            frame = scene(t)
            if t >= 5:
                matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 8, 1.0)
                matrix[0, 2] += WIDTH * 0.12
                frame = cv2.warpAffine(frame, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
            return frame
        self.assertIn("deplacement", confirmed(run(self.detector(), frame_at, 0, 12)))
        mobile = run(self.detector(camera_fixe=False), frame_at, 0, 12)
        self.assertNotIn("deplacement", confirmed(mobile))
        self.assertEqual(mobile[-1]["displacement_check"], "désactivé (caméra mobile)")

    def test_blur_and_glare_are_image_degradation(self):
        def blurred(t):
            return cv2.GaussianBlur(scene(t), (0, 0), 6) if t >= 5 else scene(t)

        def glare(t):
            frame = scene(t)
            if t >= 5:
                frame[:int(HEIGHT * 0.5), :] = 255
            return frame
        self.assertEqual(confirmed(run(self.detector(), blurred, 0, 12)), {"image_degradee"})
        self.assertIn("image_degradee", confirmed(run(self.detector(), glare, 0, 12)))

    def test_frozen_stream(self):
        frozen = scene(5.0)
        results = run(self.detector(), lambda t: frozen if t >= 5 else scene(t), 0, 22)
        self.assertEqual(confirmed(results), {"flux_fige"})

    def test_replay_of_moving_scene_is_detected(self):
        # Scène en mouvement ; à t = 10 s, on rejoue les images de t = 3 à 8 s
        def frame_at(t):
            return scene(t - 7, pan=40) if 10 <= t < 15 else scene(t, pan=40)
        self.assertIn("rejeu", confirmed(run(self.detector(camera_fixe=False), frame_at, 0, 15)))

    def test_static_scene_is_not_a_replay(self):
        # Scène immobile (seul le bruit du capteur change) : ressemble à tout son historique
        results = run(self.detector(), lambda t: scene(t, moving=False), 0, 20)
        self.assertEqual(confirmed(results), set())

    def test_loop_reset_prevents_false_replay(self):
        detector = self.detector(camera_fixe=False)
        def frame_at(t):
            return scene(t % 8, pan=40)  # vidéo de 8 s lue en boucle
        results, t = [], 0.0
        while t < 24:
            if t > 0 and t % 8 == 0:
                detector.reset_replay()  # comme tools/evaluate_tamper.py --loops entre deux tours
            results.append(detector.update(frame_at(t), t))
            t = round(t + 1 / HZ, 3)
        self.assertNotIn("rejeu", confirmed(results))

    def test_declared_reference_at_other_resolution(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            cv2.imwrite(str(path), cv2.resize(scene(0), (WIDTH * 3, HEIGHT * 3)))  # référence en haute résolution
            detector = self.detector(tamper_reference=str(path))
            def frame_at(t):
                frame = cv2.resize(scene(t), (WIDTH * 2, HEIGHT * 2))  # flux à une autre résolution
                if t >= 3:
                    frame = np.roll(frame, int(WIDTH * 2 * 0.2), axis=1)
                return frame
            results = run(detector, frame_at, 0, 8)
        self.assertEqual(results[0]["reference"], "profil")
        self.assertNotEqual(results[0]["state"], "initialisation")
        self.assertIn("deplacement", confirmed(results))

    def test_frame_by_frame_processing_reuses_last_check(self):
        detector = self.detector()
        results = [detector.update(scene(t / 60), t / 60) for t in range(0, 600)]  # 10 s à 60 img/s
        self.assertEqual(confirmed(results), set())  # images consécutives très proches : pas de « flux figé »
        self.assertLess(sum(r["ms"] > 0 for r in results), 40)

    def test_disabled_detector(self):
        detector = TamperDetector({**RULES["tamper"], "enabled": False})
        self.assertEqual(detector.update(scene(0), 0)["state"], "desactive")

    def test_perceptual_hash(self):
        a, b = cv2.cvtColor(scene(1), cv2.COLOR_BGR2GRAY), cv2.cvtColor(scene(2), cv2.COLOR_BGR2GRAY)
        self.assertLessEqual(hamming(dhash(a), dhash(b)), 6)
        self.assertGreater(hamming(dhash(a), dhash(np.fliplr(a).copy())), 15)


class PrudenceTests(unittest.TestCase):
    PERSONS = [
        {"track_id": 1, "status": "conforme", "reasons": [], "violations": []},
        {"track_id": 2, "status": "violation", "reasons": [], "violations": ["casque"]},
        {"track_id": 3, "status": "a_verifier", "reasons": [], "violations": []},
        {"track_id": 4, "status": "indetermine", "reasons": ["tête hors champ"], "violations": []},
    ]

    def persons(self):
        return [dict(p, reasons=list(p["reasons"])) for p in self.PERSONS]

    def test_compromised_camera_never_reports_compliance(self):
        persons = self.persons()
        camera = {"compromised": True, "active": [{"type": "obstruction", "label": "Obstruction de l'objectif"}]}
        self.assertEqual(apply_prudence(persons, camera), 1)
        self.assertEqual([p["status"] for p in persons], ["indetermine", "violation", "a_verifier", "indetermine"])
        self.assertIn("surveillance compromise", persons[0]["reasons"][0])
        self.assertEqual(persons[1]["violations"], ["casque"])  # violation prouvée conservée
        self.assertEqual(persons[3]["reasons"], ["tête hors champ"])

    def test_reliable_camera_changes_nothing(self):
        persons = self.persons()
        self.assertEqual(apply_prudence(persons, {"compromised": False, "active": []}), 0)
        self.assertEqual(persons, self.persons())


class CyberIncidentTests(unittest.TestCase):
    def test_one_cyber_incident_per_sabotage_logged_as_cyber(self):
        import csv
        import tempfile
        from pathlib import Path

        from app.alerts.alert_manager import AlertManager
        from app.alerts.incident_manager import IncidentManager

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "alerts.csv"
            manager = IncidentManager(alert_manager=AlertManager(str(log), cooldown_frames=0),
                                      snapshot_dir=Path(directory) / "captures", source_name="Caméra test")
            frame = scene(0)
            situation = {"kind": "obstruction", "track_id": "camera", "bbox": (0, 0, WIDTH, HEIGHT),
                         "severity": "critique", "confidence": None, "started_t": 5.0,
                         "message": "Obstruction de l'objectif (image sombre)"}
            for step in range(10):  # sabotage confirmé à t = 7,5 s, maintenu jusqu'à 12 s
                manager.update([situation], 7.5 + step * 0.5, step, frame)
            manager.update([], 16.0, 20, frame)
            incidents = manager.snapshot()
            with log.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue((Path(directory) / "captures" / incidents[0]["snapshot"]).is_file())
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]["label"], "CYBER · Obstruction de l'objectif")
        self.assertEqual(incidents[0]["status"], "terminee")
        self.assertEqual(incidents[0]["started_t"], 5.0)  # début réel, pas l'instant de confirmation
        self.assertAlmostEqual(incidents[0]["duration_s"], 7.0, delta=0.1)
        self.assertEqual([r["alert_type"] for r in rows], ["cyber"])

    def test_compromised_banner_is_drawn(self):
        from app.visualization.overlay import COLORS, OverlayRenderer

        integrity = {"compromised": True, "active": [
            {"type": "flux_fige", "label": "Flux figé", "detail": "image inchangée", "duration_s": 4.0, "since": 1.0}]}
        image = OverlayRenderer().render(scene(0), persons=[], machines=[], situations=[], raw_detections=[],
                                         scale=1.0, hud={"camera": "Test", "integrity": integrity})
        violet = np.all(np.abs(image[HEIGHT - 60, 0:3].astype(int) - COLORS["cyber"]) <= 2, axis=1)
        self.assertTrue(violet.any())  # cadre violet autour de l'image


def _contrast(a_bgr, b_bgr):
    from app.visualization.overlay import _luminance
    high, low = sorted((_luminance(a_bgr), _luminance(b_bgr)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class OverlayColorTests(unittest.TestCase):
    """Le dessin sur la vidéo utilise les couleurs d'état de dashboard/assets/hse.css, lisibles sur toute image."""

    STATUS_TOKENS = {"conforme": "--st-normal", "violation": "--st-danger", "a_verifier": "--st-vigil-fill",
                     "indetermine": "--st-off", "cyber": "--st-cyber", "arret": "--st-normal",
                     "marche": "--st-danger", "inconnu": "--st-vigil-fill", "danger": "--st-danger",
                     "vigilance": "--st-vigil-fill", "machine_arretee": "--st-off"}

    @classmethod
    def setUpClass(cls):
        import re
        from pathlib import Path
        css = (Path(__file__).resolve().parent / "dashboard" / "assets" / "hse.css").read_text(encoding="utf-8")
        root = css[css.index(":root{"):css.index("}", css.index(":root{"))]
        cls.tokens = {name: tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
                      for name, value in re.findall(r"(--st-[a-z-]+):(#[0-9a-fA-F]{6})", root)}

    def test_overlay_colors_are_the_state_colors_of_hse_css(self):
        from app.visualization.overlay import COLORS, label_background
        for status, token in self.STATUS_TOKENS.items():
            with self.subTest(status=status):
                self.assertEqual(tuple(reversed(COLORS[status])), self.tokens[token])
        # étiquette « à vérifier » : orange foncé, pour que le texte blanc reste lisible
        self.assertEqual(tuple(reversed(label_background(COLORS["a_verifier"]))), self.tokens["--st-vigil"])

    def test_labels_and_strokes_stay_readable_on_light_and_dark_images(self):
        from app.visualization.overlay import COLORS, halo, label_background
        dark, light = (24, 24, 24), (236, 236, 236)
        for status in ("conforme", "violation", "a_verifier", "indetermine", "cyber", "vigilance", "danger"):
            with self.subTest(status=status):
                color = COLORS[status]
                self.assertGreaterEqual(_contrast((255, 255, 255), label_background(color)), 4.5)  # texte blanc
                for background in (dark, light):  # trait ou liseré : au moins 3:1 (élément graphique)
                    self.assertGreaterEqual(max(_contrast(color, background), _contrast(halo(color), background)), 3)

    def test_drawn_boxes_use_state_colors_with_halo(self):
        from app.visualization.overlay import COLORS, OverlayRenderer, halo
        persons = [{"track_id": 1, "bbox": (60, 120, 200, 420), "status": "conforme", "items": {}, "reasons": []},
                   {"track_id": 2, "bbox": (300, 120, 440, 420), "status": "a_verifier", "items": {}, "reasons": []}]
        for value in (10, 245):  # image presque noire, puis presque blanche
            image = OverlayRenderer(show_raw=False).render(
                np.full((HEIGHT, WIDTH, 3), value, np.uint8), persons=persons, machines=[], situations=[],
                raw_detections=[], scale=1.0, hud={"camera": "Test"})
            pixels = {tuple(int(c) for c in p) for p in image.reshape(-1, 3)}
            for status in ("conforme", "a_verifier"):
                with self.subTest(fond=value, status=status):
                    self.assertIn(tuple(COLORS[status]), pixels)
                    self.assertIn(tuple(halo(COLORS[status])), pixels)


class TamperSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path

        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.video = cls.root / "propre.mp4"
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (WIDTH, HEIGHT))
        for index in range(100):  # 10 s à 10 img/s
            writer.write(scene(index / 10, pan=30))
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def simulate(self, attack, **kwargs):
        import json

        from tools.tamper_simulator import simulate

        path = simulate(self.video, attack, self.root / "attaques", start=4.0, duration=4.0, **kwargs)
        truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        capture = cv2.VideoCapture(str(path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return path, truth, frames

    def test_in_place_attacks_keep_duration_and_write_ground_truth(self):
        for attack, expected in (("obstruction", "obstruction"), ("flou", "image_degradee"),
                                 ("eblouissement", "image_degradee")):
            path, truth, frames = self.simulate(attack)
            self.assertEqual(path.name, f"propre__{attack}.mp4")
            self.assertEqual((truth["expected_type"], truth["start_s"], truth["end_s"]), (expected, 4.0, 8.0))
            self.assertEqual(frames, 100)

    def test_displacement_lasts_until_end(self):
        _, truth, _ = self.simulate("deplacement")
        self.assertEqual((truth["start_s"], truth["end_s"]), (4.0, 10.0))

    def test_frozen_and_replay_insert_frames(self):
        _, truth, frames = self.simulate("flux_fige")
        self.assertEqual((truth["start_s"], truth["end_s"], frames), (4.0, 19.0, 250))  # 15 s insérées
        _, truth, frames = self.simulate("rejeu", replay_s=4.0)
        # inséré à 2 × 4 s = 8 s (historique suffisant) : les images de 4 à 8 s sont rejouées
        self.assertEqual((truth["start_s"], truth["end_s"], frames), (8.0, 12.0, 140))
        with self.assertRaisesRegex(ValueError, "trop court"):
            self.simulate("rejeu", replay_s=3.0)

    def test_simulated_attacks_are_detected(self):
        detected = {}
        for attack in ("obstruction", "flux_fige", "rejeu"):
            path, truth, _ = self.simulate(attack, replay_s=3.5)
            detector = TamperDetector(RULES["tamper"], {"camera_fixe": False})
            capture = cv2.VideoCapture(str(path))
            index, types = 0, set()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % 5 == 0:  # 2 analyses/s
                    types |= {e["type"] for e in detector.update(frame, index / 10)["active"]}
                index += 1
            capture.release()
            detected[attack] = truth["expected_type"] in types
        self.assertEqual(detected, {"obstruction": True, "flux_fige": True, "rejeu": True})

    def test_attacked_video_inherits_original_camera_profile(self):
        from app.config import load_camera_profile

        profile = load_camera_profile("test1__obstruction.mp4")
        self.assertFalse(profile["camera_fixe"])
        self.assertIn("attaque simulée (obstruction)", profile["name"])


class AuditLogTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.log, self.key = root / "audit_log.jsonl", root / "secrets" / "audit.key"

    def tearDown(self):
        self.directory.cleanup()

    def write_entries(self, count=5):
        from app.security.audit_log import AuditLog

        audit = AuditLog(self.log, self.key)
        for index in range(count):
            audit.append("etat_machine_modifie", {"machine_id": "engin_1", "apres": "marche" if index % 2 else "arret"})
        return audit

    def lines(self):
        return self.log.read_text(encoding="utf-8").splitlines()

    def rewrite(self, lines):
        self.log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_intact_log_is_valid_and_chained(self):
        import json

        from app.security.audit_log import verify

        self.write_entries()
        entries = [json.loads(line) for line in self.lines()]
        self.assertEqual(verify(self.log, self.key), {"ok": True, "count": 5, "errors": [], "warnings": []})
        self.assertEqual(entries[1]["prev_hash"], entries[0]["hash"])
        self.assertTrue(self.key.is_file())

    def test_modified_line_is_detected(self):
        from app.security.audit_log import verify

        self.write_entries()
        lines = self.lines()
        lines[2] = lines[2].replace('"arret"', '"marche"')
        self.rewrite(lines)
        result = verify(self.log, self.key)
        self.assertFalse(result["ok"])
        self.assertTrue(any(e.startswith("Ligne 3 : contenu modifié") for e in result["errors"]))

    def test_deleted_middle_and_last_lines_are_detected(self):
        from app.security.audit_log import verify

        self.write_entries()
        lines = self.lines()
        self.rewrite(lines[:2] + lines[3:])
        self.assertTrue(any("chaînage rompu" in e for e in verify(self.log, self.key)["errors"]))
        self.rewrite(lines[:4])  # suppression de la dernière ligne : seul l'en-tête le révèle
        self.assertIn("1 dernière(s) ligne(s) supprimée(s) (en-tête : 5 entrées, fichier : 4).",
                      verify(self.log, self.key)["errors"])

    def test_chain_rewritten_without_key_is_detected(self):
        import hashlib
        import json

        from app.security.audit_log import GENESIS, _canonical, verify

        self.write_entries(3)
        previous, forged = GENESIS, []
        for line in self.lines():  # l'attaquant recalcule tout le chaînage SHA-256, mais n'a pas la clé
            entry = json.loads(line)
            entry["data"]["apres"] = "inconnu"
            entry["prev_hash"] = previous
            entry["hash"] = previous = hashlib.sha256(_canonical(entry)).hexdigest()
            forged.append(json.dumps(entry, ensure_ascii=False))
        self.rewrite(forged)
        errors = verify(self.log, self.key)["errors"]
        self.assertEqual(sum("signature HMAC invalide" in e for e in errors), 3)

    def test_tampering_is_recorded_at_next_start(self):
        import json

        from app.security.audit_log import AuditLog

        self.write_entries(3)
        self.rewrite(self.lines()[:2])
        audit = AuditLog(self.log, self.key)
        last = json.loads(self.lines()[-1])
        self.assertEqual(last["event"], "alerte_integrite_journal")
        self.assertEqual(audit._seq, 3)

    def test_incidents_are_audited_with_duration(self):
        import json

        from app.alerts.incident_manager import IncidentManager

        audit = self.write_entries(0)
        manager = IncidentManager(audit=audit, audit_context={"session_id": "20261029_100000"}, source_name="Caméra")
        situation = {"kind": "flux_fige", "track_id": "camera", "bbox": (0, 0, 10, 10), "severity": "critique",
                     "message": "Flux figé", "started_t": 2.0}
        for step in range(5):
            manager.update([situation], 4.5 + step, step)
        manager.update([], 20.0, 30)
        events = [json.loads(line) for line in self.lines()]
        self.assertEqual([e["event"] for e in events], ["incident_ouvert", "incident_cloture"])
        self.assertEqual(events[1]["data"]["duration_s"], 6.5)
        self.assertEqual(events[1]["data"]["categorie"], "cyber")
        self.assertEqual(events[0]["data"]["session_id"], "20261029_100000")

    def test_open_incidents_closed_at_end_of_session(self):
        import json

        from app.alerts.incident_manager import IncidentManager

        manager = IncidentManager(audit=self.write_entries(0))
        situation = {"kind": "rejeu", "track_id": "camera", "bbox": (0, 0, 10, 10), "severity": "critique",
                     "message": "Rejeu suspecté", "started_t": 3.0}
        manager.update([situation], 6.0, 1)
        manager.update([situation], 8.0, 2)
        manager.close_all()
        last = json.loads(self.lines()[-1])
        self.assertEqual((last["event"], last["data"]["duration_s"], last["data"]["motif"]),
                         ("incident_cloture", 5.0, "fin de session"))
        self.assertEqual(manager.snapshot()[0]["status"], "terminee")


class SabotageFreezeTests(unittest.TestCase):
    """Incidents de personnes pendant un sabotage confirmé : gelés, repris ou clôturés avec motif."""

    PHONE = {"kind": "telephone", "track_id": 1, "bbox": (100, 100, 200, 400), "severity": "critique",
             "confidence": 0.7, "message": "Personne #1 : usage du téléphone"}
    FROZEN = {"kind": "flux_fige", "track_id": "camera", "bbox": (0, 0, 640, 480), "severity": "critique",
              "message": "Flux figé", "started_t": 4.5}
    RELIABLE = {"compromised": False, "suspected": False}
    SUSPECTED = {"compromised": False, "suspected": True}
    COMPROMISED = {"compromised": True, "suspected": False}

    def setUp(self):
        import tempfile
        from pathlib import Path

        from app.alerts.incident_manager import IncidentManager
        from app.security.audit_log import AuditLog

        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.log = root / "audit_log.jsonl"
        self.manager = IncidentManager(audit=AuditLog(self.log, root / "audit.key"))

    def tearDown(self):
        self.directory.cleanup()

    def events(self):
        import json

        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def play(self, start, end, situations, integrity, step=0.5):
        t = start
        while t < end:
            self.manager.update(situations, t, int(t * 10), integrity=integrity)
            t = round(t + step, 3)

    def phone_incident(self):
        return next(i for i in self.manager.snapshot() if i["kind"] == "telephone")

    def test_frozen_stream_freezes_open_violation(self):
        self.play(1.0, 4.5, [self.PHONE], self.RELIABLE)          # téléphone confirmé, image fiable
        self.play(4.5, 7.0, [self.PHONE], self.SUSPECTED)         # flux figé en observation
        self.play(7.0, 19.0, [self.PHONE, self.FROZEN], self.COMPROMISED)  # le flux figé montre toujours le téléphone
        phone = self.phone_incident()
        self.assertEqual(phone["status"], "active")
        self.assertTrue(phone["frozen"])
        self.assertEqual(phone["duration_s"], 3.0)                # arrêtée à la dernière image fiable (t = 4,0 s)
        self.assertIn("non actualisée depuis", phone["message"])
        self.assertEqual(len([i for i in self.manager.snapshot() if i["kind"] == "telephone"]), 1)
        self.assertIn("incident_gele", [e["event"] for e in self.events()])

    def test_person_seen_again_resumes_without_counting_sabotage(self):
        self.play(1.0, 4.5, [self.PHONE], self.RELIABLE)
        self.play(4.5, 10.0, [self.FROZEN], self.COMPROMISED)
        self.play(10.0, 12.0, [dict(self.PHONE, track_id=9)], self.RELIABLE)  # revue, nouvel identifiant
        phone = self.phone_incident()
        self.assertFalse(phone["frozen"])
        self.assertNotIn("non actualisée", phone["message"])
        self.assertEqual(phone["duration_s"], 3.0 + 1.5)          # 1 → 4 s, puis 10 → 11,5 s
        self.assertIn("incident_reprise", [e["event"] for e in self.events()])

    def test_person_not_seen_again_is_closed_with_sabotage_reason(self):
        self.play(1.0, 4.5, [self.PHONE], self.RELIABLE)
        self.play(4.5, 10.0, [self.FROZEN], self.COMPROMISED)
        self.play(10.0, 15.0, [], self.RELIABLE)
        phone = self.phone_incident()
        self.assertEqual((phone["status"], phone["duration_s"]), ("terminee", 3.0))
        closing = [e for e in self.events() if e["event"] == "incident_cloture" and e["data"]["type"] == "telephone"]
        self.assertEqual(closing[0]["data"]["motif"], "interrompu par sabotage caméra")

    def test_no_closure_while_anomaly_is_being_confirmed(self):
        self.play(1.0, 3.0, [self.PHONE], self.RELIABLE)
        self.play(3.0, 8.0, [], self.SUSPECTED)                   # personne masquée, sabotage en observation
        self.assertEqual(self.phone_incident()["status"], "active")
        self.play(8.0, 12.5, [], self.RELIABLE)                   # fausse alerte levée : clôture normale
        phone = self.phone_incident()
        self.assertEqual(phone["status"], "terminee")
        self.assertNotIn("close_reason", phone)

    def test_no_new_person_incident_from_compromised_image(self):
        self.play(1.0, 6.0, [self.PHONE, self.FROZEN], self.COMPROMISED)
        self.assertEqual([i["kind"] for i in self.manager.snapshot()], ["flux_fige"])

    def test_end_of_session_during_sabotage(self):
        self.play(1.0, 4.5, [self.PHONE], self.RELIABLE)
        self.play(4.5, 9.0, [self.FROZEN], self.COMPROMISED)
        self.manager.close_all()
        closing = [e for e in self.events() if e["event"] == "incident_cloture" and e["data"]["type"] == "telephone"]
        self.assertEqual(closing[0]["data"]["motif"], "interrompu par sabotage caméra")


class ModelIntegrityTests(unittest.TestCase):
    def test_known_modified_and_unknown_weights(self):
        import tempfile
        from pathlib import Path

        from app.security.model_integrity import ModelIntegrityError, update, verify_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, integrity = root / "best.pt", root / "model_integrity.json"
            weights.write_bytes(b"poids d'origine")
            update([weights], integrity)
            self.assertEqual(len(verify_model(weights, integrity)), 64)
            weights.write_bytes(b"poids d'origine + charge malveillante")
            with self.assertRaisesRegex(ModelIntegrityError, "MODIFIÉ"):
                verify_model(weights, integrity)
            other = root / "inconnu.pt"
            other.write_bytes(b"x")
            with self.assertRaisesRegex(ModelIntegrityError, "aucune empreinte"):
                verify_model(other, integrity)

    def test_project_models_match_recorded_hashes(self):
        from app.security.model_integrity import configured_models, verify_model

        from app.config import resolve

        paths = configured_models()
        if not all(resolve(path).is_file() for path in paths):
            self.skipTest("poids non publiés (voir README, « Poids du modèle »)")
        for path in paths:
            verify_model(path)


class UploadValidationTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        import dashboard.live_api as api

        self.directory = tempfile.TemporaryDirectory()
        self.api, self.previous = api, api.TEST_VIDEOS_PATH
        api.TEST_VIDEOS_PATH = Path(self.directory.name)

    def tearDown(self):
        self.api.TEST_VIDEOS_PATH = self.previous
        self.directory.cleanup()

    def upload(self, name, data):
        import io

        return self.api.save_uploaded_video(name, io.BytesIO(data), len(data))

    def test_fake_video_is_refused_and_deleted(self):
        with self.assertRaisesRegex(ValueError, "pas une vidéo lisible"):
            self.upload("piege.mp4", b"MZ\x90\x00" + b"\x00" * 4000)  # exécutable renommé
        self.assertEqual(list(self.api.TEST_VIDEOS_PATH.iterdir()), [])

    def test_real_video_is_accepted(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ok.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (WIDTH, HEIGHT))
            for index in range(5):
                writer.write(scene(index / 10))
            writer.release()
            self.assertEqual(self.upload("vidéo test.mp4", path.read_bytes()), "video_test.mp4")


if __name__ == "__main__":
    unittest.main()
