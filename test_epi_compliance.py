import unittest

from app.epi.compliance_checker import EPIComplianceChecker


def detection(class_name, confidence, bbox=(10, 10, 90, 100), track_id=1):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
        "track_id": track_id,
    }


class EPIComplianceTests(unittest.TestCase):
    def test_confirmed_helmet_and_vest_are_compliant_when_harness_is_not_required(self):
        checker = EPIComplianceChecker()
        statuses = checker.check([
            detection("Person", 0.95),
            detection("Hardhat", 0.9, (30, 15, 60, 35)),
            detection("Safety Vest", 0.9, (25, 40, 75, 75)),
        ])

        self.assertEqual(statuses[0]["status"], "compliant")
        self.assertEqual(statuses[0]["violations"], [])

    def test_uncertain_missing_vest_detection_is_not_a_violation(self):
        checker = EPIComplianceChecker()
        statuses = checker.check([
            detection("Person", 0.9),
            detection("Hardhat", 0.9, (30, 15, 60, 35)),
            detection("NO-Safety Vest", 0.796, (25, 40, 75, 75)),
        ])

        self.assertEqual(statuses[0]["status"], "undetermined")
        self.assertEqual(statuses[0]["violations"], [])
        self.assertEqual(statuses[0]["uncertain_violations"], ["NO-Safety Vest"])

    def test_confident_missing_vest_detection_remains_a_violation(self):
        checker = EPIComplianceChecker()
        statuses = checker.check([
            detection("Person", 0.9),
            detection("NO-Safety Vest", 0.9, (25, 40, 75, 75)),
        ])

        self.assertEqual(statuses[0]["status"], "violation")
        self.assertEqual(statuses[0]["violations"], ["NO-Safety Vest"])

    def test_phone_detection_uses_its_explicit_violation_threshold(self):
        checker = EPIComplianceChecker()
        statuses = checker.check([
            detection("Person", 0.9),
            detection("Phone", 0.55, (45, 40, 55, 55)),
        ])

        self.assertEqual(statuses[0]["status"], "violation")
        self.assertEqual(statuses[0]["violations"], ["Phone"])


    def test_torso_only_duplicate_box_is_merged_into_full_person(self):
        checker = EPIComplianceChecker()
        statuses = checker.check([
            detection("Person", 0.9, (100, 0, 500, 1000), track_id=1),
            detection("Person", 0.7, (110, 400, 490, 700), track_id=2),
        ])
        self.assertEqual([s["track_id"] for s in statuses], [1])

    def test_two_people_side_by_side_are_kept(self):
        checker = EPIComplianceChecker()
        statuses = checker.check([
            detection("Person", 0.9, (100, 100, 300, 700), track_id=1),
            detection("Person", 0.9, (220, 120, 420, 720), track_id=2),
        ])
        self.assertEqual(len(statuses), 2)

    def test_small_person_behind_is_kept(self):
        checker = EPIComplianceChecker()
        statuses = checker.check([
            detection("Person", 0.9, (100, 0, 500, 1000), track_id=1),
            detection("Person", 0.8, (300, 100, 360, 300), track_id=2),
        ])
        self.assertEqual(len(statuses), 2)


if __name__ == "__main__":
    unittest.main()
