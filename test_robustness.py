"""
Tests légers de l'évaluation de robustesse (tools/robustness/) : sans
best.pt ni jeu de données, sur des images et des boîtes synthétiques.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "robustness"))

import common as C  # noqa: E402


def box(cls, x1, y1, x2, y2):
    return cls, np.array([x1, y1, x2, y2], dtype=float)


class CommonTests(unittest.TestCase):
    def test_local_yaml_uses_absolute_paths_without_touching_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = C.write_local_yaml(Path(directory) / "jeu", Path(directory) / "sorties")
            text = path.read_text(encoding="utf-8")
        self.assertIn("test: test/images", text)
        self.assertNotIn("../", text)
        self.assertIn('"NO-Harness"', text)

    def test_sample_is_deterministic_and_keeps_rare_classes(self):
        items = [{"name": f"img{i:03d}.jpg", "labels": np.array([[0, .5, .5, .1, .1]])} for i in range(50)]
        items[37]["labels"] = np.array([[C.CLASS_ID["NO-Harness"], .5, .5, .1, .1]])
        first, second = C.choose_sample(items, 10, seed=3), C.choose_sample(items, 10, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertIn("img037.jpg", first)
        self.assertNotEqual(first, C.choose_sample(items, 10, seed=4))

    def test_letterbox_keeps_ratio_and_centers_image(self):
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        out, ratio, (left, top) = C.letterbox_square(image, 640)
        self.assertEqual(out.shape, (640, 640, 3))
        self.assertEqual((ratio, left, top), (1.0, 0, 140))
        self.assertTrue((out[:140] == 114).all() and (out[140:500] == 0).all())

    def test_perfect_predictions_give_full_recall_and_ap(self):
        evaluator = C.Evaluator()
        true_cls = np.array([0, 2, 0])
        true_xyxy = np.array([[10, 10, 100, 200], [30, 10, 70, 50], [200, 20, 300, 220]], dtype=float)
        evaluator.add("a.jpg", true_xyxy.copy(), np.array([.9, .8, .95]), true_cls.copy(), true_cls, true_xyxy)
        metrics = evaluator.metrics()
        for name in ("Person", "NO-Hardhat"):
            self.assertAlmostEqual(metrics[name]["rappel"], 1.0, places=3)
            self.assertGreater(metrics[name]["map50"], 0.99)
            self.assertEqual(metrics[name]["rappel_seuil_prod"], 1.0)
        self.assertEqual(metrics["Person"]["instances"], 2)

    def test_operating_recall_ignores_low_confidence_predictions(self):
        evaluator = C.Evaluator()
        true_xyxy = np.array([[30, 10, 70, 50]], dtype=float)
        evaluator.add("a.jpg", true_xyxy.copy(), np.array([0.2]), np.array([2]), np.array([2]), true_xyxy)
        metrics = evaluator.metrics()
        self.assertEqual(metrics["NO-Hardhat"]["rappel_seuil_prod"], 0.0)  # 0,2 < seuil 0,35
        self.assertGreater(metrics["NO-Hardhat"]["map50"], 0.99)          # mais comptée pour la mAP

    def test_person_decisions_follow_production_thresholds(self):
        person = [0, 0, 100, 300]
        head = [30, 0, 70, 40]
        true_cls = np.array([0, 2])
        true_xyxy = np.array([person, head], dtype=float)
        pred_xyxy = np.array([person, head], dtype=float)
        # NO-Hardhat à 0,9 : violation ; à 0,5 : à vérifier ; absent : indéterminé ; Hardhat+gilet : conforme
        self.assertEqual(C.person_decisions(true_cls, true_xyxy, pred_xyxy, np.array([.9, .9]), np.array([0, 2])),
                         [("violation", "violation")])
        self.assertEqual(C.person_decisions(true_cls, true_xyxy, pred_xyxy, np.array([.9, .5]), np.array([0, 2])),
                         [("violation", "a_verifier")])
        self.assertEqual(C.person_decisions(true_cls, true_xyxy, pred_xyxy[:1], np.array([.9]), np.array([0])),
                         [("violation", "indetermine")])
        vest = [10, 80, 90, 200]
        flipped = np.array([person, head, vest], dtype=float)
        self.assertEqual(C.person_decisions(true_cls, true_xyxy, flipped, np.array([.9, .9, .9]), np.array([0, 1, 3])),
                         [("violation", "conforme")])
        self.assertEqual(C.person_decisions(true_cls, true_xyxy, np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)),
                         [("violation", "non_detectee")])


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        self.image[30:200, 50:400] = (40, 160, 220)  # zone uniforme : le flou et la poussière s'y voient

    def test_perturbations_return_valid_images_increasingly_degraded(self):
        import perturbations as P

        boxes = np.array([[40, 20, 120, 100]], dtype=float)
        for kind in P.SEVERITIES:
            self.assertTrue((P.apply(self.image, kind, 0) == self.image).all(), kind)
            gaps = []
            for severity in range(1, 6):
                out = P.apply(self.image, kind, severity, np.random.default_rng(0), boxes=boxes)
                self.assertEqual((out.shape, out.dtype), (self.image.shape, np.uint8), kind)
                gaps.append(np.abs(out.astype(int) - self.image.astype(int)).mean())
            self.assertGreater(gaps[0], 0, kind)
            self.assertGreater(gaps[-1], gaps[0], kind)

    def test_occlusion_only_touches_object_boxes(self):
        import perturbations as P

        boxes = np.array([[40, 20, 120, 100]], dtype=float)
        out = P.apply(self.image, "occlusion", 5, np.random.default_rng(0), boxes=boxes)
        outside = np.ones(self.image.shape[:2], dtype=bool)
        outside[20:100, 40:120] = False
        self.assertTrue((out[outside] == self.image[outside]).all())
        self.assertFalse((out == self.image).all())

    def test_unknown_perturbation_or_severity_is_refused(self):
        import perturbations as P

        with self.assertRaises(ValueError):
            P.apply(self.image, "pluie", 1)
        with self.assertRaises(ValueError):
            P.apply(self.image, "flou", 6)


if __name__ == "__main__":
    unittest.main()
