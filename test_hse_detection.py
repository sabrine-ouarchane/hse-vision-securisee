import json
import tempfile
import unittest
from pathlib import Path

from app.alerts.incident_manager import IncidentManager
from app.config import load_detection_rules
from app.epi.behavior_tracker import BehaviorTracker
from app.epi.compliance_checker import EPIComplianceChecker
from app.sensors.machine_state_provider import MachineStateProvider
from app.situations.proximity_checker import MachineTracker, ProximityChecker
from dashboard.live_api import resolve_source

RULES = load_detection_rules()
FRAME = (1080, 1920)
PERSON = (800, 300, 1000, 900)  # 600 px de haut : bien visible
HEAD = (860, 300, 940, 380)
TORSO = (820, 420, 980, 650)


def det(class_name, confidence, bbox, track_id=None):
    return {"class_name": class_name, "confidence": confidence, "bbox": bbox, "track_id": track_id}


def run(tracker, frames, fps=10.0, harness=False):
    checker = EPIComplianceChecker(harness_required=harness, confidence_thresholds=RULES["violation_confidence_thresholds"])
    persons = []
    for index, detections in enumerate(frames, start=1):
        persons = tracker.update(checker.check(detections), index / fps, FRAME)
    return persons


class BehaviorTrackerTests(unittest.TestCase):
    def test_single_frame_missing_helmet_is_not_confirmed(self):
        persons = run(BehaviorTracker(RULES), [[det("Person", 0.9, PERSON, 1), det("NO-Hardhat", 0.9, HEAD)]])
        self.assertNotEqual(persons[0]["status"], "violation")
        self.assertEqual(persons[0]["items"]["casque"], "a_verifier")

    def test_stable_missing_helmet_is_confirmed_after_min_duration(self):
        frame = [det("Person", 0.9, PERSON, 1), det("NO-Hardhat", 0.9, HEAD)]
        persons = run(BehaviorTracker(RULES), [frame] * 15)
        self.assertEqual(persons[0]["status"], "violation")
        self.assertIn("casque", persons[0]["violations"])

    def test_low_confidence_absence_stays_to_verify(self):
        frame = [det("Person", 0.9, PERSON, 1), det("NO-Hardhat", 0.5, HEAD)]
        persons = run(BehaviorTracker(RULES), [frame] * 20)
        self.assertEqual(persons[0]["items"]["casque"], "a_verifier")
        self.assertEqual(persons[0]["violations"], [])

    def test_helmet_and_vest_confirmed_is_compliant(self):
        frame = [det("Person", 0.9, PERSON, 1), det("Hardhat", 0.9, HEAD), det("Safety Vest", 0.9, TORSO)]
        persons = run(BehaviorTracker(RULES), [frame] * 12)
        self.assertEqual(persons[0]["status"], "conforme")

    def test_far_person_is_undetermined_not_violation(self):
        small = (900, 500, 920, 540)
        frame = [det("Person", 0.9, small, 1), det("NO-Hardhat", 0.95, (905, 500, 915, 508))]
        persons = run(BehaviorTracker(RULES), [frame] * 20)
        self.assertEqual(persons[0]["status"], "indetermine")
        self.assertIn("personne trop petite / éloignée", persons[0]["reasons"])

    def test_head_out_of_frame_does_not_confirm_missing_helmet(self):
        cut = (800, 0, 1000, 700)
        frame = [det("Person", 0.9, cut, 1), det("NO-Hardhat", 0.95, (860, 0, 940, 60))]
        persons = run(BehaviorTracker(RULES), [frame] * 20)
        self.assertEqual(persons[0]["items"]["casque"], "non_visible")

    def test_helmet_on_the_ground_is_not_counted_as_worn(self):
        frame = [det("Person", 0.9, PERSON, 1), det("Hardhat", 0.9, (850, 850, 930, 900)), det("Safety Vest", 0.9, TORSO)]
        persons = run(BehaviorTracker(RULES), [frame] * 12)
        self.assertNotEqual(persons[0]["items"]["casque"], "present")

    def test_phone_needs_repeated_hits(self):
        phone = det("Phone", 0.7, (900, 500, 940, 560))
        once = [[det("Person", 0.9, PERSON, 1), phone]] + [[det("Person", 0.9, PERSON, 1)]] * 15
        self.assertNotEqual(run(BehaviorTracker(RULES), once)[0]["items"]["telephone"], "detecte")
        often = [[det("Person", 0.9, PERSON, 1), phone]] * 15
        self.assertEqual(run(BehaviorTracker(RULES), often)[0]["items"]["telephone"], "detecte")

    def test_phone_outside_active_area_is_not_a_violation(self):
        tracker = BehaviorTracker(RULES, phone_active_area=[[0, 0], [0.2, 0], [0.2, 1], [0, 1]])
        frames = [[det("Person", 0.9, PERSON, 1), det("Phone", 0.7, (900, 500, 940, 560))]] * 15
        person = run(tracker, frames)[0]
        self.assertEqual(person["items"]["telephone"], "hors_zone")
        self.assertNotIn("telephone", person["violations"])

    def test_harness_only_required_when_configured(self):
        frame = [det("Person", 0.9, PERSON, 1), det("NO-Harness", 0.9, TORSO)]
        self.assertEqual(run(BehaviorTracker(RULES), [frame] * 15)[0]["items"]["harnais"], "non_requis")
        persons = run(BehaviorTracker(RULES, harness_required=True), [frame] * 15, harness=True)
        self.assertIn("harnais", persons[0]["violations"])


class ProximityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RULES["proximity"]
        self.machine = {"id": 1, "bbox": (1000, 300, 1600, 900), "confidence": 0.8}

    def check_sequence(self, person_bbox, state, policy="prudent", frames=10, fps=10.0):
        checker = ProximityChecker(self.cfg)
        result = []
        for index in range(1, frames + 1):
            person = {"track_id": 1, "bbox": person_bbox, "confidence": 0.9}
            result = checker.check([person], [self.machine], index / fps, FRAME, state, policy)
        return result

    def test_person_next_to_running_machine_raises_critical_alert(self):
        situations = self.check_sequence((880, 400, 980, 900), "marche")
        self.assertEqual(situations[0]["status"], "danger")
        self.assertEqual(situations[0]["severity"], "critique")
        self.assertIsNone(situations[0]["distance_m"])

    def test_stopped_machine_does_not_raise_running_machine_alert(self):
        situations = self.check_sequence((880, 400, 980, 900), "arret")
        self.assertEqual(situations[0]["status"], "machine_arretee")
        self.assertIsNone(situations[0]["severity"])

    def test_unknown_state_follows_configured_policy(self):
        self.assertEqual(self.check_sequence((880, 400, 980, 900), "inconnu", "prudent")[0]["status"], "danger_etat_inconnu")
        cautious = self.check_sequence((880, 400, 980, 900), "inconnu", "surveillance")[0]
        self.assertEqual(cautious["status"], "a_verifier")
        self.assertEqual(cautious["severity"], "avertissement")

    def test_foreground_person_overlapping_in_image_is_not_adjacent(self):
        # La personne chevauche l'engin à l'écran mais ses pieds sont bien plus près de la caméra
        situations = self.check_sequence((1200, 700, 1260, 1075), "marche")
        self.assertTrue(not situations or situations[0]["status"] != "danger")

    def test_operator_inside_machine_cab_is_not_a_pedestrian_at_risk(self):
        situations = self.check_sequence((1300, 450, 1400, 750), "marche")
        self.assertEqual(situations, [])

    def test_operator_seen_waist_up_at_cab_bottom_is_excluded(self):
        # Boîte engin coupée au bas de la cabine : seul le buste est visible
        situations = self.check_sequence((1300, 760, 1420, 900), "marche")
        self.assertEqual(situations, [])

    def test_full_body_pedestrian_in_front_of_machine_is_evaluated(self):
        situations = self.check_sequence((1300, 560, 1400, 900), "marche")
        self.assertEqual(situations[0]["status"], "danger")

    def test_single_frame_contact_is_not_confirmed(self):
        situations = self.check_sequence((880, 400, 980, 900), "marche", frames=1)
        self.assertTrue(not situations or situations[0]["status"] != "danger")

    def test_calibrated_distance_is_reported_in_meters(self):
        calibration = {"image_points": [[0, 0], [1, 0], [1, 1], [0, 1]], "ground_points_m": [[0, 0], [20, 0], [20, 10], [0, 10]]}
        measure = ProximityChecker(self.cfg, calibration).measure((880, 400, 980, 900), self.machine["bbox"], FRAME)
        self.assertIsNotNone(measure["distance_m"])

    def test_machine_tracker_ignores_oversized_and_single_frame_boxes(self):
        tracker = MachineTracker(self.cfg)
        huge = det("Machine", 0.9, (0, 0, 1920, 1080))
        self.assertEqual(tracker.update([huge], 0.1, FRAME), [])
        machine = det("Machine", 0.8, (1000, 300, 1600, 900))
        self.assertEqual(tracker.update([machine], 0.2, FRAME), [])
        self.assertEqual(len(tracker.update([machine], 0.3, FRAME)), 1)
        self.assertEqual(len(tracker.update([], 1.0, FRAME)), 1)  # maintien bref
        self.assertEqual(tracker.update([], 5.0, FRAME), [])


class IncidentTests(unittest.TestCase):
    def situation(self, track_id, bbox=PERSON):
        return {"kind": "casque", "track_id": track_id, "bbox": bbox, "severity": "critique", "message": "m", "confidence": 0.9}

    def test_one_incident_per_persistent_situation(self):
        manager = IncidentManager()
        opened = [manager.update([self.situation(1)], t / 10, t) for t in range(1, 60)]
        self.assertEqual(sum(len(o) for o in opened), 1)

    def test_track_id_switch_reuses_incident(self):
        manager = IncidentManager()
        manager.update([self.situation(1)], 1.0, 10)
        manager.update([], 1.5, 15)
        opened = manager.update([self.situation(7, (805, 305, 1005, 905))], 2.0, 20)
        self.assertEqual(opened, [])
        self.assertEqual(manager.snapshot()[0]["track_ids"], [1, 7])

    def test_short_interruption_for_same_person_reopens_incident(self):
        manager = IncidentManager(close_after_s=1.0, merge_s=4.0)
        manager.update([self.situation(1)], 1.0, 10)
        manager.update([], 2.5, 25)
        opened = manager.update([self.situation(1)], 4.0, 40)
        self.assertEqual(opened, [])
        self.assertEqual(len(manager.snapshot()), 1)
        self.assertEqual(manager.snapshot()[0]["status"], "active")

    def test_incident_closes_after_absence(self):
        manager = IncidentManager(close_after_s=1.0)
        manager.update([self.situation(1)], 1.0, 10)
        manager.update([], 3.0, 30)
        self.assertEqual(manager.snapshot()[0]["status"], "terminee")


class MachineStateTests(unittest.TestCase):
    def test_state_is_persisted_and_invalid_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.json"
            path.write_text(json.dumps({"unknown_policy": "prudent", "machines": {}}), encoding="utf-8")
            provider = MachineStateProvider(path)
            self.assertEqual(provider.get("engin")["state"], "inconnu")
            provider.set_state("engin", "arret", unknown_policy="surveillance")
            reloaded = MachineStateProvider(path)
            self.assertEqual(reloaded.get("engin")["state"], "arret")
            self.assertEqual(reloaded.unknown_policy, "surveillance")
            with self.assertRaises(ValueError):
                provider.set_state("engin", "a_fond")

    def test_missing_state_file_is_created_from_example_and_never_overwritten(self):
        from app.config import MACHINE_STATES_EXAMPLE_PATH
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine_states.json"
            provider = MachineStateProvider(path, example=MACHINE_STATES_EXAMPLE_PATH)
            self.assertEqual(provider.initialized_from, "machine_states.example.json")
            self.assertEqual((provider.get("engin_demo_1")["state"], provider.unknown_policy), ("inconnu", "prudent"))
            provider.set_state("engin_demo_1", "arret", source="hse1")
            again = MachineStateProvider(path, example=MACHINE_STATES_EXAMPLE_PATH)
            self.assertIsNone(again.initialized_from)
            self.assertEqual(again.get("engin_demo_1")["state"], "arret")
            # fichier illisible : jamais remplacé par l'exemple
            path.write_text("{abîmé", encoding="utf-8")
            self.assertIsNone(MachineStateProvider(path, example=MACHINE_STATES_EXAMPLE_PATH).initialized_from)
            self.assertEqual(path.read_text(encoding="utf-8"), "{abîmé")
            # autre chemin sans exemple explicite : aucune copie
            other = Path(directory) / "autre.json"
            self.assertIsNone(MachineStateProvider(other).initialized_from)
            self.assertFalse(other.exists())

    def test_versioned_example_is_safe_default_and_state_file_is_ignored(self):
        import subprocess
        from app.config import MACHINE_STATES_EXAMPLE_PATH, ROOT
        example = json.loads(MACHINE_STATES_EXAMPLE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(example["unknown_policy"], "prudent")
        self.assertTrue(example["machines"])
        self.assertEqual({m["state"] for m in example["machines"].values()}, {"inconnu"})
        ignored = subprocess.run(["git", "check-ignore", "-q", "config/machine_states.json"], cwd=ROOT)
        self.assertEqual(ignored.returncode, 0)


class LiveSourceTests(unittest.TestCase):
    def test_only_known_sources_are_accepted(self):
        for bad in ("video:../README.md", "video:..\\x.mp4", "camera:99", "file:C:/x.mp4", None):
            with self.assertRaises(ValueError):
                resolve_source(bad)
        self.assertEqual(resolve_source("camera:0"), (0, "camera_0"))


if __name__ == "__main__":
    unittest.main()
