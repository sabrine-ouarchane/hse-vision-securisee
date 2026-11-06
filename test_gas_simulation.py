import unittest

from app.sensors.dashboard_gas_simulator import DashboardGasSimulator


class DashboardGasSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.simulator = DashboardGasSimulator()

    def test_normal_scenario_returns_available_demo_sensor_values(self):
        zone = self.simulator.snapshot(["zone_pilote_1"])["zones"]["zone_pilote_1"]

        self.assertEqual(len(zone["sensors"]), 7)
        self.assertTrue(all(sensor["status"] == "normal" for sensor in zone["sensors"]))
        self.assertTrue(all(sensor["sensor_id"] and sensor["timestamp"] for sensor in zone["sensors"]))

    def test_warning_and_critical_scenarios_cross_the_configured_levels(self):
        self.simulator.set_state(scenario="warning")
        warning_snapshot = self.simulator.snapshot(["zone_pilote_1"])
        warning = warning_snapshot["zones"]["zone_pilote_1"]
        self.assertIn("avertissement", {sensor["status"] for sensor in warning["sensors"]})
        self.assertTrue(warning_snapshot["alerts"])
        self.assertTrue(all(alert["type"] == "gas" for alert in warning_snapshot["alerts"]))
        self.assertEqual(len(warning_snapshot["alerts"]), len(
            self.simulator.snapshot(["zone_pilote_1"])["alerts"]
        ))

        self.simulator.set_state(scenario="critical")
        critical_snapshot = self.simulator.snapshot(["zone_pilote_1"])
        critical = critical_snapshot["zones"]["zone_pilote_1"]
        self.assertIn("critique", {sensor["status"] for sensor in critical["sensors"]})
        self.assertTrue(any(alert["severity"] == "critique" for alert in critical_snapshot["alerts"]))
        self.assertTrue(all(alert["status"] == "résolue" for alert in critical_snapshot["alerts"]
                            if alert["severity"] == "avertissement"))

    def test_return_to_normal_resolves_alert_and_status_can_be_updated(self):
        self.simulator.set_state(scenario="warning")
        warning = self.simulator.snapshot(["zone_pilote_1"])["alerts"][0]
        self.assertTrue(self.simulator.set_alert_status(warning["event_id"], "en cours"))
        self.assertEqual(warning["status"], "en cours")

        self.simulator.set_state(scenario="normal")
        resolved = self.simulator.snapshot(["zone_pilote_1"])["alerts"]
        self.assertTrue(all(alert["status"] == "résolue" for alert in resolved))
        self.assertFalse(self.simulator.set_alert_status("missing", "résolue"))

    def test_unavailable_scenario_never_returns_a_normal_value(self):
        self.simulator.set_state(scenario="unavailable")
        zone = self.simulator.snapshot(["zone_pilote_1"])["zones"]["zone_pilote_1"]

        self.assertTrue(all(sensor["value"] is None for sensor in zone["sensors"]))
        self.assertTrue(all(sensor["status"] == "indisponible" for sensor in zone["sensors"]))

    def test_unconfigured_zones_do_not_receive_fake_sensor_values(self):
        snapshot = self.simulator.snapshot(["poste_demo_pb1"])

        self.assertEqual(snapshot["zones"], {})


if __name__ == "__main__":
    unittest.main()
