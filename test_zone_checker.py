"""
Tests du module de zonage (app/zones/zone_checker.py), déplacés sans
changement depuis dashboard/test_dashboard.py lors de la suppression de
l'ancien tableau de bord du site. Les polygones caméra sont lus dans
config/camera_zones.json (config/zones.json décrit les zones du site).
"""

import unittest

from app.config import CAMERA_ZONES_PATH
from app.sensors.machine_state_simulator import MachineStateSimulator
from app.zones.zone_checker import ZoneChecker


class ZoneCheckerTests(unittest.TestCase):
    def test_running_machine_is_only_red_when_proximity_is_confirmed(self):
        class SafeGas:
            def is_out_of_threshold(self, zone_id, elapsed_seconds):
                return False, None, {"CO_ppm": 1}

            def get_readings(self, zone_id, elapsed_seconds):
                return {"CO_ppm": 1}

        level, reason = ZoneChecker(CAMERA_ZONES_PATH).get_zone_risk_level(
            "zone_pilote_1", 0, SafeGas(), MachineStateSimulator(default_state="marche")
        )
        self.assertEqual(level, "VERT")
        self.assertIn("aucune violation active", reason)

    def test_unknown_machine_state_is_orange(self):
        class SafeGas:
            def is_out_of_threshold(self, zone_id, elapsed_seconds):
                return False, None, {"CO_ppm": 1}

            def get_readings(self, zone_id, elapsed_seconds):
                return {"CO_ppm": 1}

        level, reason = ZoneChecker(CAMERA_ZONES_PATH).get_zone_risk_level(
            "zone_pilote_1", 0, SafeGas(), MachineStateSimulator(default_state="inconnu")
        )
        self.assertEqual(level, "ORANGE")
        self.assertIn("état machine inconnu", reason)

    def test_missing_gas_readings_are_not_green(self):
        class MissingGas:
            def is_out_of_threshold(self, zone_id, elapsed_seconds):
                return False, None, {}

            def get_readings(self, zone_id, elapsed_seconds):
                return None

        level, reason = ZoneChecker(CAMERA_ZONES_PATH).get_zone_risk_level(
            "zone_pilote_1", 0, MissingGas(), MachineStateSimulator(default_state="arret")
        )
        self.assertEqual(level, "ORANGE")
        self.assertIn("gaz indisponibles", reason)

    def test_confirmed_proximity_to_running_machine_is_red(self):
        class SafeGas:
            def is_out_of_threshold(self, zone_id, elapsed_seconds):
                return False, None, {"CO_ppm": 1}

            def get_readings(self, zone_id, elapsed_seconds):
                return {"CO_ppm": 1}

        level, reason = ZoneChecker(CAMERA_ZONES_PATH).get_zone_risk_level(
            "zone_pilote_1", 0, SafeGas(), MachineStateSimulator(default_state="marche"),
            has_machine_proximity=True,
        )
        self.assertEqual(level, "ROUGE")
        self.assertIn("proximité machine", reason)


if __name__ == "__main__":
    unittest.main()
