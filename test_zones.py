"""
Tests des zones dangereuses (étape 9) : simulateur de capteurs, contrôles de
plausibilité, classement vert / orange / rouge, hystérésis, zones machine et
réglages modifiables depuis l'interface.
"""

import copy
import json
import math
import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.sensors.provider import Reading, SensorProvider
from app.sensors.simulated_provider import SimulatedSensorProvider
from app.zones.risk import ZoneRiskEvaluator, classify_value, instability, machine_level, orange_limits
from app.zones.settings import (ConfigError, PARAM_BOUNDS, apply_parameter_changes, check_parameter_changes,
                                load_zones_config, validate)

CONFIG = load_zones_config()
T0 = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)
NORMS, MEASURES = CONFIG["normes"], CONFIG["mesures"]


def at(seconds):
    return T0 + timedelta(seconds=seconds)


class Feeder:
    """Alimente un évaluateur seconde par seconde avec des séries synthétiques (sans simulateur)."""

    UNITS = {m: NORMS[m]["unite"] for m in NORMS if not m.startswith("_")}

    def __init__(self, config=CONFIG):
        self.config = config
        self.evaluator = ZoneRiskEvaluator(config, T0)
        self.t = 0
        self.sensors = {s["id"]: (z["id"], s["mesure"], s.get("normale", MEASURES[s["mesure"]]["normale"]))
                        for z in config["zones"] for s in z.get("capteurs", [])}
        self.machines = {f"{z['id']}-{m['id']}": z["id"] for z in config["zones"] for m in z.get("machines", [])}
        self.rng = random.Random(0)

    def step(self, overrides=None, silent=(), machine_state="arret", presence=None, lag=0.0):
        """Une seconde : chaque capteur émet sa valeur normale (avec bruit) sauf `overrides` ou `silent`."""
        self.t += 1
        now = at(self.t)
        readings = []
        for sensor_id, (zone, measure, normal) in self.sensors.items():
            if sensor_id in silent:
                continue
            if overrides and sensor_id in overrides:
                value = overrides[sensor_id]
            else:
                spec = MEASURES[measure]
                value = round(round((normal + self.rng.gauss(0, spec["bruit"])) / spec["resolution"])
                              * spec["resolution"], 6)
                value = max(value, spec["plage"][0])
            readings.append(Reading(now - timedelta(seconds=lag), zone, sensor_id, measure, value, self.UNITS[measure]))
        for sensor_id, zone in self.machines.items():
            if sensor_id not in silent:
                state = machine_state(sensor_id) if callable(machine_state) else machine_state
                readings.append(Reading(now, zone, sensor_id, "etat", state, ""))
        self.evaluator.ingest(readings, now)
        return self.evaluator.evaluate(now, presence)

    def run(self, seconds, **kwargs):
        state = None
        for _ in range(seconds):
            state = self.step(**kwargs)
        return state


# ---------------------------------------------------------------------------
class ValueRulesTests(unittest.TestCase):
    def test_toxic_gas_green_orange_red_at_70_percent(self):
        so2, spec = NORMS["SO2_ppm"], MEASURES["SO2_ppm"]
        self.assertEqual(classify_value(so2, spec, 1.3, 70)[0], "VERT")
        level, reason = classify_value(so2, spec, 1.5, 70)
        self.assertEqual(level, "ORANGE")
        self.assertIn("1,4 ppm", reason)
        level, reason = classify_value(so2, spec, 2.4, 70)
        self.assertEqual(level, "ROUGE")
        self.assertIn("SO2 = 2,4 ppm ≥ 2 ppm", reason)
        self.assertEqual(classify_value(so2, spec, 2.0, 70)[0], "ROUGE")  # S-HSE-23 : < 2 ppm

    def test_oxygen_uses_both_bounds_from_ambient_air(self):
        o2, spec = NORMS["O2_pct"], MEASURES["O2_pct"]
        limits = orange_limits(o2, 70)
        self.assertAlmostEqual(limits["bas"], 19.92)
        self.assertAlmostEqual(limits["haut"], 22.72)
        self.assertEqual(classify_value(o2, spec, 20.9, 70)[0], "VERT")
        self.assertEqual(classify_value(o2, spec, 19.8, 70)[0], "ORANGE")
        self.assertEqual(classify_value(o2, spec, 22.9, 70)[0], "ORANGE")
        level, reason = classify_value(o2, spec, 19.4, 70)
        self.assertEqual(level, "ROUGE")
        self.assertIn("< 19,5 %", reason)
        level, reason = classify_value(o2, spec, 23.6, 70)
        self.assertEqual(level, "ROUGE")
        self.assertIn("> 23,5 %", reason)

    def test_temperature_orange_from_20_degree_reference(self):
        temp, spec = NORMS["Temp_C"], MEASURES["Temp_C"]
        self.assertAlmostEqual(orange_limits(temp, 70)["haut"], 30.5)
        self.assertEqual(classify_value(temp, spec, 30.0, 70)[0], "VERT")
        self.assertEqual(classify_value(temp, spec, 31.0, 70)[0], "ORANGE")
        self.assertEqual(classify_value(temp, spec, 35.0, 70)[0], "ROUGE")

    def test_zero_threshold_gas_below_detection_limit_is_green(self):
        no2, spec = NORMS["NO2_ppm"], MEASURES["NO2_ppm"]
        self.assertEqual(classify_value(no2, spec, 0.1, 70)[0], "VERT")
        level, reason = classify_value(no2, spec, 0.3, 70)
        self.assertEqual(level, "ORANGE")
        self.assertIn("confirmation en cours", reason)

    def test_percentage_setting_moves_the_orange_limit(self):
        so2, spec = NORMS["SO2_ppm"], MEASURES["SO2_ppm"]
        self.assertEqual(classify_value(so2, spec, 1.5, 80)[0], "VERT")
        self.assertEqual(classify_value(so2, spec, 1.5, 50)[0], "ORANGE")


class InstabilityTests(unittest.TestCase):
    params = CONFIG["parametres"]

    def test_stable_noise_is_not_unstable(self):
        rng = random.Random(1)
        window = [(t, 0.3 + rng.gauss(0, 0.1)) for t in range(60)]
        self.assertIsNone(instability(NORMS["SO2_ppm"], MEASURES["SO2_ppm"], window, self.params))

    def test_slow_drift_towards_threshold_is_unstable(self):
        window = [(t, 0.3 + 0.01 * t) for t in range(60)]  # +0,6 ppm/min, seuil dans ≈ 2 min
        reason = instability(NORMS["SO2_ppm"], MEASURES["SO2_ppm"], window, self.params)
        self.assertIn("SO2 en hausse", reason)
        self.assertIn("2 ppm atteint dans", reason)

    def test_oxygen_drift_down_targets_lower_bound(self):
        window = [(t, 20.9 - 0.01 * t) for t in range(60)]
        reason = instability(NORMS["O2_pct"], MEASURES["O2_pct"], window, self.params)
        self.assertIn("O2 en baisse", reason)
        self.assertIn("19,5 %", reason)

    def test_abnormal_variability_is_unstable(self):
        rng = random.Random(2)
        window = [(t, 4 + rng.gauss(0, 3.0)) for t in range(60)]  # bruit normal du CO : 0,6 ppm
        reason = instability(NORMS["CO_ppm"], MEASURES["CO_ppm"], window, self.params)
        self.assertIn("CO instable", reason)

    def test_short_window_is_not_judged(self):
        window = [(t, 0.3 + 0.05 * t) for t in range(5)]
        self.assertIsNone(instability(NORMS["SO2_ppm"], MEASURES["SO2_ppm"], window, self.params))


# ---------------------------------------------------------------------------
class GasZoneTests(unittest.TestCase):
    def test_stable_series_is_green_with_readable_reason(self):
        state = Feeder().run(90)["sechage"]
        self.assertEqual(state["niveau"], "VERT")
        self.assertIn("sous 70 %", state["raison"])
        self.assertEqual(len(state["capteurs"]), 5)

    def test_drift_goes_green_orange_red(self):
        feeder = Feeder()
        feeder.run(60)
        levels = []
        for second in range(200):
            value = round(0.3 + 2.2 * min(second / 180, 1), 1)
            levels.append(feeder.step(overrides={"sechage-so2": value})["sechage"]["niveau_brut"])
        self.assertEqual(levels[0], "VERT")
        first_orange, first_red = levels.index("ORANGE"), levels.index("ROUGE")
        self.assertLess(first_orange, first_red)
        self.assertLess(first_orange, 120)  # la dérive est signalée avant le seuil orange (1,4 ppm à 99 s)

    def test_leak_is_red_with_value_in_reason(self):
        feeder = Feeder()
        feeder.run(60)
        state = feeder.run(3, overrides={"cuves-lie": 4.0})
        state = feeder.step(overrides={"cuves-lie": 7.5})["cuves"]
        self.assertEqual(state["niveau"], "ROUGE")
        self.assertIn("Explosivité = 7,5 % LIE ≥ 5 % LIE", state["raison"])
        self.assertEqual(state["causes"][0]["type"], "gaz")

    def test_return_to_normal_waits_for_hysteresis_one_level_at_a_time(self):
        feeder = Feeder()
        feeder.run(60)
        feeder.run(5, overrides={"hangars-co": 30.0})
        state = feeder.run(5)["hangars"]  # valeurs revenues à la normale
        self.assertEqual(state["niveau"], "ROUGE")
        self.assertIn("hystérésis", state["raison"])
        delay = CONFIG["parametres"]["hysteresis_s"]
        state = feeder.run(delay - 5)["hangars"]  # dernière seconde du délai d'hystérésis
        self.assertEqual(state["niveau"], "ROUGE")
        state = feeder.step()["hangars"]
        self.assertEqual(state["niveau"], "ORANGE")
        # le pic reste dans la fenêtre glissante (instabilité) puis l'hystérésis s'applique de nouveau
        self.assertIn("CO instable", state["raison_actuelle"])
        state = feeder.run(CONFIG["parametres"]["fenetre_s"] + delay)["hangars"]
        self.assertEqual(state["niveau"], "VERT")

    def test_rise_is_immediate_even_during_hysteresis(self):
        feeder = Feeder()
        feeder.run(60)
        feeder.run(3, overrides={"hangars-co": 20.0})
        state = feeder.step(overrides={"hangars-co": 26.0})["hangars"]
        self.assertEqual(state["niveau"], "ROUGE")

    def test_oxygen_low_and_high_in_a_zone(self):
        for value, sign in ((19.2, "<"), (23.8, ">")):
            feeder = Feeder()
            feeder.run(30)
            for step in range(1, 5):  # variation physiquement possible (< 1 %/s)
                feeder.step(overrides={"hangars-o2": round(20.9 + (value - 20.9) * step / 5, 2)})
            state = feeder.step(overrides={"hangars-o2": value})["hangars"]
            self.assertEqual(state["niveau"], "ROUGE")
            self.assertIn(f"O2 = {str(value).replace('.', ',')} % {sign}", state["raison"])

    def test_zero_threshold_gas_red_only_after_confirmation(self):
        feeder = Feeder()
        feeder.run(30)
        state = feeder.step(overrides={"sechage-no2": 0.4})["sechage"]
        self.assertEqual(state["niveau_brut"], "ORANGE")
        self.assertIn("confirmation en cours", state["raison_actuelle"])
        state = feeder.run(CONFIG["parametres"]["confirmation_zero_s"], overrides={"sechage-no2": 0.4})["sechage"]
        self.assertEqual(state["niveau"], "ROUGE")
        self.assertIn("NO2 = 0,4 ppm > 0 ppm", state["raison"])

    def test_isolated_zero_threshold_spike_does_not_open_red(self):
        feeder = Feeder()
        feeder.run(30)
        feeder.step(overrides={"sechage-no2": 0.4})
        state = feeder.step()["sechage"]
        self.assertNotEqual(state["niveau_brut"], "ROUGE")


# ---------------------------------------------------------------------------
class PlausibilityTests(unittest.TestCase):
    def assert_sensor_fault(self, state, sensor_id, fault, fragment):
        self.assertEqual(state["niveau"], "ROUGE")
        cause = state["causes"][0]
        self.assertEqual((cause["type"], cause["cle"], cause["defaut"]), ("capteur", sensor_id, fault))
        self.assertIn("donnée capteur non fiable", state["raison"])
        self.assertIn(fragment, state["raison"])

    def test_missing_data_is_red(self):
        feeder = Feeder()
        feeder.run(30)
        state = feeder.run(CONFIG["parametres"]["delai_absence_s"] + 1, silent={"sechage-so2"})["sechage"]
        self.assert_sensor_fault(state, "sechage-so2", "absence", "aucune donnée depuis")

    def test_sensor_never_seen_is_red(self):
        state = Feeder().run(CONFIG["parametres"]["delai_absence_s"] + 1, silent={"cuves-lie"})["cuves"]
        self.assert_sensor_fault(state, "cuves-lie", "absence", "aucune donnée reçue")

    def test_late_timestamp_is_red(self):
        feeder = Feeder()
        state = feeder.step(lag=CONFIG["parametres"]["delai_absence_s"] + 2)["hangars"]
        self.assertEqual(state["niveau"], "ROUGE")
        self.assertEqual(state["causes"][0]["defaut"], "retard")

    def test_frozen_value_is_red(self):
        feeder = Feeder()
        feeder.run(30)
        state = feeder.run(CONFIG["parametres"]["duree_figee_s"] + 1, overrides={"sechage-co": 6.5})["sechage"]
        self.assert_sensor_fault(state, "sechage-co", "figee", "figé à 6,5 ppm")

    def test_impossible_jump_is_red(self):
        feeder = Feeder()
        feeder.run(30)
        state = feeder.step(overrides={"sechage-so2": 60.0})["sechage"]
        self.assert_sensor_fault(state, "sechage-so2", "saut", "variation impossible")

    def test_out_of_range_is_red(self):
        feeder = Feeder()
        feeder.run(30)
        state = feeder.step(overrides={"cuves-lie": 140.0})["cuves"]
        self.assert_sensor_fault(state, "cuves-lie", "hors_plage", "hors de la plage du capteur")

    def test_non_numeric_value_is_red(self):
        feeder = Feeder()
        feeder.run(5)
        state = feeder.step(overrides={"hangars-o2": "NaN?"})["hangars"]
        self.assert_sensor_fault(state, "hangars-o2", "invalide", "valeur invalide")

    def test_forced_zero_on_oxygen_is_detected(self):
        feeder = Feeder()
        feeder.run(30)
        state = feeder.step(overrides={"hangars-o2": 0.0})["hangars"]
        self.assert_sensor_fault(state, "hangars-o2", "saut", "variation impossible")

    def test_forced_zero_on_gas_near_zero_is_a_known_limit(self):
        """Limite documentée : 0 forcé sur un gaz proche de 0 reste plausible (redondance nécessaire)."""
        feeder = Feeder()
        feeder.run(30)
        state = feeder.run(CONFIG["parametres"]["duree_figee_s"] + 5, overrides={"sechage-so2": 0.0})["sechage"]
        self.assertNotIn("capteur", {c["type"] for c in state["causes"]})

    def test_sensor_recovers_after_quarantine(self):
        feeder = Feeder()
        feeder.run(30)
        feeder.step(overrides={"sechage-so2": 60.0})
        feeder.run(5, overrides={"sechage-so2": 0.3})
        state = feeder.step()["sechage"]
        self.assertEqual(state["niveau_brut"], "ROUGE")  # toujours en quarantaine
        state = feeder.run(CONFIG["parametres"]["quarantaine_s"])["sechage"]
        self.assertNotIn("capteur", {c["type"] for c in state["causes"]})


# ---------------------------------------------------------------------------
class MachineZoneTests(unittest.TestCase):
    def test_rule_matrix(self):
        expected = {
            ("arret", "presente"): "VERT", ("arret", "absente"): "VERT", ("arret", "inconnue"): "VERT",
            ("marche", "presente"): "ROUGE", ("marche", "absente"): "ORANGE", ("marche", "inconnue"): "ORANGE",
            ("inconnu", "presente"): "ROUGE", ("inconnu", "absente"): "ORANGE", ("inconnu", "inconnue"): "ORANGE",
        }
        for (state, presence), level in expected.items():
            with self.subTest(state=state, presence=presence):
                self.assertEqual(machine_level(state, {"etat": presence})[0], level)

    def test_unknown_presence_reason_and_unknown_state_reason(self):
        level, reason = machine_level("inconnu", {"etat": "inconnue", "source": None})
        self.assertIn("S-HSE-22", reason)
        self.assertIn("présence non vérifiable (aucune caméra associée)", reason)

    def test_active_proximity_is_red(self):
        level, reason = machine_level("marche", {"etat": "absente", "proximite": True})
        self.assertEqual(level, "ROUGE")
        self.assertIn("proximité", reason)

    def test_compromised_camera_prevents_green(self):
        self.assertEqual(machine_level("arret", {"etat": "absente", "camera_compromise": True})[0], "ORANGE")
        self.assertEqual(machine_level("marche", {"etat": "absente", "camera_compromise": True})[0], "ROUGE")

    def test_zone_state_combines_machines(self):
        feeder = Feeder()
        state = feeder.step(machine_state="arret", presence={"parcs": {"etat": "presente"}})["parcs"]
        self.assertEqual(state["niveau"], "VERT")
        feeder = Feeder()
        one_running = lambda sensor_id: "marche" if sensor_id == "parcs-roue_pelle_1" else "arret"
        state = feeder.step(machine_state=one_running, presence={"parcs": {"etat": "presente"}})["parcs"]
        self.assertEqual(state["niveau"], "ROUGE")
        self.assertIn("personne détectée", state["raison"])
        self.assertEqual({m["machine"]: m["etat"] for m in state["machines"]},
                         {"gerbeur_1": "arret", "roue_pelle_1": "marche"})

    def test_silent_machine_sensor_is_red_and_state_unknown(self):
        feeder = Feeder()
        feeder.run(5)
        state = feeder.run(CONFIG["parametres"]["delai_absence_s"] + 1, silent={"convoyeurs-convoyeur_principal"})
        state = state["convoyeurs"]
        self.assertEqual(state["niveau"], "ROUGE")
        self.assertEqual(state["causes"][0]["type"], "capteur")
        self.assertEqual(state["machines"][0]["etat"], "inconnu")


class GreyZoneTests(unittest.TestCase):
    def test_out_of_scope_zone_is_grey(self):
        state = Feeder().step()["poste_elec"]
        self.assertEqual((state["niveau"], state["raison"]), ("GRIS", "hors périmètre (non évalué)"))

    def test_stopped_monitoring_is_grey_without_causes(self):
        evaluator = ZoneRiskEvaluator(CONFIG, T0)
        states = evaluator.evaluate(at(60), active=False)
        self.assertTrue(all(s["niveau"] == "GRIS" and not s["causes"] for s in states.values()))
        self.assertEqual(states["sechage"]["raison"], "surveillance arrêtée")


# ---------------------------------------------------------------------------
class SimulatorTests(unittest.TestCase):
    def series(self, seed, seconds=30, scenario=None):
        provider = SimulatedSensorProvider(CONFIG, seed=seed)
        provider.start(T0)
        if scenario:
            provider.set_scenario(*scenario, now=T0)
        return [r for s in range(1, seconds + 1) for r in provider.poll(at(s))]

    def test_is_a_sensor_provider_emitting_1hz_readings(self):
        provider = SimulatedSensorProvider(CONFIG)
        self.assertIsInstance(provider, SensorProvider)
        provider.start(T0)
        readings = provider.poll(at(10))
        self.assertEqual(len(readings), 11 * len(provider.sensors()))
        first = readings[0].as_dict()
        self.assertEqual(set(first), {"timestamp", "zone", "sensor_id", "measure", "value", "unit"})
        self.assertEqual(provider.poll(at(10)), [])  # rien de nouveau
        self.assertEqual(len(provider.poll(at(11))), len(provider.sensors()))

    def test_same_seed_same_series_regardless_of_polling(self):
        provider = SimulatedSensorProvider(CONFIG, seed=7)
        provider.start(T0)
        bulk = provider.poll(at(30))
        self.assertEqual(bulk, self.series(7))
        self.assertNotEqual(self.series(7), self.series(8))

    def test_not_running_emits_nothing(self):
        provider = SimulatedSensorProvider(CONFIG)
        self.assertEqual(provider.poll(at(5)), [])
        provider.start(T0)
        provider.stop()
        self.assertEqual(provider.poll(at(5)), [])
        self.assertFalse(provider.status()["actif"])

    def test_silent_scenario_mutes_only_the_target_sensor(self):
        readings = self.series(0, 10, ("sechage", "muet", "sechage-co"))
        sensors = {r.sensor_id for r in readings if r.timestamp > at(1)}
        self.assertNotIn("sechage-co", sensors)
        self.assertIn("sechage-so2", sensors)

    def test_drift_crosses_threshold_and_leak_is_physically_plausible(self):
        drift = [r.value for r in self.series(0, 200, ("sechage", "derive")) if r.sensor_id == "sechage-so2"]
        self.assertLess(drift[5], 1.0)
        self.assertGreater(drift[-1], NORMS["SO2_ppm"]["max"])
        leak = [r.value for r in self.series(0, 20, ("cuves", "fuite")) if r.sensor_id == "cuves-lie"]
        self.assertGreater(max(leak), NORMS["LIE_pct"]["max"])
        steps = [abs(b - a) for a, b in zip(leak, leak[1:])]
        self.assertLessEqual(max(steps), MEASURES["LIE_pct"]["vitesse_max_par_s"])

    def test_machine_scenarios_and_declared_state(self):
        provider = SimulatedSensorProvider(CONFIG, declared_state=lambda m: "marche" if m == "gerbeur_1" else "arret")
        provider.start(T0)
        states = {r.sensor_id: r.value for r in provider.poll(at(0)) if r.measure == "etat"}
        self.assertEqual(states["parcs-gerbeur_1"], "marche")
        self.assertEqual(states["parcs-roue_pelle_1"], "arret")
        provider.set_scenario("parcs", "arret", now=at(0))
        states = {r.sensor_id: r.value for r in provider.poll(at(1)) if r.measure == "etat"}
        self.assertEqual(states["parcs-gerbeur_1"], "arret")

    def test_invalid_scenarios_are_refused(self):
        provider = SimulatedSensorProvider(CONFIG)
        with self.assertRaises(ValueError):
            provider.set_scenario("sechage", "marche")
        with self.assertRaises(ValueError):
            provider.set_scenario("parcs", "fuite")
        with self.assertRaises(ValueError):
            provider.set_scenario("poste_elec", "stable")
        with self.assertRaises(ValueError):
            provider.set_scenario("sechage", "fuite", "cuves-lie")

    def test_simulator_drives_evaluator_end_to_end(self):
        provider = SimulatedSensorProvider(CONFIG)
        provider.start(T0)
        evaluator = ZoneRiskEvaluator(CONFIG, T0)
        for second in range(1, 31):
            evaluator.ingest(provider.poll(at(second)), at(second))
        self.assertEqual(evaluator.evaluate(at(30))["sechage"]["niveau"], "VERT")
        provider.set_scenario("sechage", "hors_plage", now=at(30))
        evaluator.ingest(provider.poll(at(31)), at(31))
        state = evaluator.evaluate(at(31))["sechage"]
        self.assertEqual(state["niveau"], "ROUGE")
        self.assertEqual(state["causes"][0]["defaut"], "hors_plage")


# ---------------------------------------------------------------------------
class SettingsTests(unittest.TestCase):
    def test_bounds_are_safe(self):
        self.assertEqual(PARAM_BOUNDS["pourcentage_orange"][:2], (50, 90))
        self.assertEqual(PARAM_BOUNDS["hysteresis_s"][:2], (5, 120))
        self.assertEqual(PARAM_BOUNDS["delai_absence_s"][:2], (1, 10))

    def test_valid_change_returns_diff(self):
        diff = check_parameter_changes(CONFIG, {"pourcentage_orange": 60, "hysteresis_s": 30})
        self.assertEqual(diff, {"pourcentage_orange": (70, 60)})

    def test_out_of_bounds_and_non_numeric_values_are_refused(self):
        for change in ({"pourcentage_orange": 100}, {"pourcentage_orange": 49}, {"hysteresis_s": 3600},
                       {"delai_absence_s": 0}, {"delai_absence_s": math.nan}, {"fenetre_s": "60"},
                       {"quarantaine_s": True}, {}):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                check_parameter_changes(CONFIG, change)

    def test_normative_thresholds_and_zones_are_not_editable_from_interface(self):
        for change in ({"normes": {"SO2_ppm": {"max": 50}}}, {"SO2_ppm": 50}, {"zones": []},
                       {"vitesse_max_par_s": 1000}):
            with self.subTest(change=change), self.assertRaises(ConfigError) as error:
                check_parameter_changes(CONFIG, change)
            self.assertIn("n'est pas modifiable", str(error.exception))

    def test_apply_writes_file_atomically_and_keeps_norms(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zones.json"
            path.write_text(json.dumps(CONFIG), encoding="utf-8")
            config, diff = apply_parameter_changes({"hysteresis_s": 45}, path)
            self.assertEqual(diff, {"hysteresis_s": (30, 45)})
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["parametres"]["hysteresis_s"], 45)
            self.assertEqual(stored["normes"], CONFIG["normes"])
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["zones.json"])

    def test_invalid_file_is_rejected(self):
        broken = copy.deepcopy(CONFIG)
        broken["parametres"]["pourcentage_orange"] = 99
        with self.assertRaises(ConfigError):
            validate(broken)
        broken = copy.deepcopy(CONFIG)
        broken["zones"][0]["capteurs"].append({"id": "x", "mesure": "CH4_ppm"})
        with self.assertRaises(ConfigError):
            validate(broken)

    def test_invalid_plan_is_rejected(self):
        broken = copy.deepcopy(CONFIG)
        broken["zones"][0]["plan"]["points"] = [[0, 0], [2000, 0], [0, 10]]
        with self.assertRaises(ConfigError):
            validate(broken)

    def test_project_config_has_the_seven_poc_zones(self):
        methods = {z["id"]: z["methode"] for z in CONFIG["zones"]}
        self.assertEqual(methods, {"sechage": "gaz", "cuves": "gaz", "hangars": "gaz", "convoyeurs": "machine",
                                   "parcs": "machine", "ferroviaire": "machine", "poste_elec": "hors_perimetre"})


# ---------------------------------------------------------------------------
# Étape 9b : moniteur, incidents, audit, cohérence, API
# ---------------------------------------------------------------------------
class Clock:
    def __init__(self, start=T0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds=1):
        self.now += timedelta(seconds=seconds)
        return self.now


HSE = {"username": "hse1", "role": "responsable_hse"}
ADMIN = {"username": "admin1", "role": "admin"}


class MonitorTestCase(unittest.TestCase):
    def setUp(self):
        from app.security.audit_log import AuditLog
        from app.zones.monitor import ZoneMonitor

        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config_path = self.root / "zones.json"
        self.config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
        self.audit = AuditLog(self.root / "audit_log.jsonl", self.root / "secrets" / "audit.key")
        self.alerts_path = self.root / "alerts_log.csv"
        self.clock = Clock()
        self.presence = {}
        self.monitor = ZoneMonitor(self.audit, self.alerts_path, self.root / "zones", self.config_path,
                                   presence=lambda: self.presence, clock=self.clock)

    def tearDown(self):
        self.directory.cleanup()

    def events(self, name=None):
        from app.security.audit_log import read_entries
        return [e for e in read_entries(self.audit.path) if name is None or e["event"] == name]

    def run_seconds(self, seconds):
        for _ in range(seconds):
            self.monitor.tick(self.clock.advance())
        return self.monitor.states

    def csv_rows(self):
        import csv
        with self.alerts_path.open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


class MonitorTests(MonitorTestCase):
    def test_starts_stopped_grey_with_who_and_when(self):
        status = self.monitor.status()
        self.assertFalse(status["actif"])
        self.assertEqual(status["arret"]["par"], "systeme")
        self.assertTrue(all(z["niveau"] == "GRIS" for z in status["zones"]))
        self.monitor.start(HSE)
        self.run_seconds(5)
        self.monitor.stop(HSE)
        status = self.monitor.status()
        self.assertEqual((status["arret"]["par"], status["arret"]["a"]),
                         ("hse1", self.clock.now.isoformat(timespec="seconds")))
        self.assertTrue(all(z["niveau"] == "GRIS" for z in status["zones"]))
        self.assertEqual(status["zones"][0]["raison"], "surveillance arrêtée")
        self.assertEqual([e["utilisateur"] for e in self.events("surveillance_zones_demarree")], ["hse1"])
        self.assertEqual([e["utilisateur"] for e in self.events("surveillance_zones_arretee")], ["hse1"])

    def test_double_start_or_stop_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.monitor.stop(HSE)
        self.monitor.start(HSE)
        with self.assertRaises(RuntimeError):
            self.monitor.start(HSE)

    def test_gas_leak_opens_one_incident_in_csv_audit_and_history_then_closes(self):
        self.monitor.start(HSE)
        self.run_seconds(30)
        self.monitor.set_scenario("cuves", "fuite", user=HSE)
        self.run_seconds(20)
        opened = self.events("incident_ouvert")
        self.assertEqual(len(opened), 1)
        data = opened[0]["data"]
        self.assertEqual((data["session_id"], data["type"], data["categorie"], data["zone"]),
                         ("zones", "gaz", "zone_gaz", "cuves"))
        row = next(r for r in self.csv_rows() if r["event_id"] == data["csv_event_id"])
        self.assertEqual(row["alert_type"], "zone_gaz")
        self.assertTrue(row["message"].startswith("[Zone Cuves de combustible] Explosivité"))
        self.assertEqual(self.monitor.open_incidents()[0]["mesure"], "LIE_pct")
        self.monitor.set_scenario("cuves", "retour", user=HSE)
        self.run_seconds(200)
        closed = self.events("incident_cloture")
        self.assertEqual(len(closed), 1)
        self.assertGreater(closed[0]["data"]["duree_s"], 10)
        stored = self.monitor.history.incidents()
        self.assertEqual(len(stored), 1)
        self.assertIsNotNone(stored[0]["fin"])
        scenario = self.events("scenario_simulation_modifie")[0]
        self.assertEqual((scenario["utilisateur"], scenario["data"]["ancien"], scenario["data"]["nouveau"]),
                         ("hse1", "stable", "fuite"))

    def test_unreliable_sensor_opens_cyber_incident(self):
        self.monitor.start(HSE)
        self.run_seconds(10)
        self.monitor.set_scenario("sechage", "muet", "sechage-o2", user=HSE)
        self.run_seconds(6)
        data = self.events("incident_ouvert")[0]["data"]
        self.assertEqual((data["type"], data["categorie"], data["cle"]), ("capteur", "cyber", "sechage-o2"))
        self.assertIn("donnée capteur non fiable", data["message"])
        self.assertEqual(self.csv_rows()[-1]["alert_type"], "cyber")
        self.assertEqual(self.monitor.states["sechage"]["niveau"], "ROUGE")

    def test_stop_closes_open_incidents(self):
        self.monitor.start(HSE)
        self.run_seconds(5)
        self.monitor.set_scenario("cuves", "hors_plage", user=HSE)
        self.run_seconds(3)
        self.monitor.stop(HSE)
        closed = self.events("incident_cloture")
        self.assertEqual([c["data"]["motif"] for c in closed], ["surveillance arrêtée"])
        self.assertEqual(self.monitor.open_incidents(), [])

    def test_flapping_around_threshold_keeps_a_single_incident(self):
        self.monitor.start(HSE)
        self.run_seconds(5)
        self.monitor.set_scenario("sechage", "derive", user=HSE)
        self.run_seconds(260)  # SO2 oscille autour de 2 ppm puis se stabilise au-dessus
        self.assertEqual(len(self.events("incident_ouvert")), 1)

    def test_machine_zone_uses_presence_from_live_with_hold(self):
        self.monitor.start(HSE)
        self.monitor.set_scenario("parcs", "marche", user=HSE)
        self.presence = {"parcs": {"etat": "presente", "source": "Engin mobile"}}
        self.run_seconds(2)
        self.assertEqual(self.monitor.states["parcs"]["niveau"], "ROUGE")
        self.assertEqual(self.events("incident_ouvert")[0]["data"]["type"], "machine")
        self.presence = {"parcs": {"etat": "absente", "source": "Engin mobile"}}
        self.run_seconds(2)
        self.assertEqual(self.monitor.states["parcs"]["niveau_brut"], "ROUGE")  # maintien de présence
        self.run_seconds(CONFIG["parametres"]["presence_maintien_s"])
        self.assertEqual(self.monitor.states["parcs"]["niveau_brut"], "ORANGE")
        self.assertIn("aucune personne détectée", self.monitor.states["parcs"]["raison_actuelle"])

    def test_compromised_camera_keeps_stopped_machine_zone_out_of_green(self):
        self.monitor.start(HSE)
        self.monitor.set_scenario("parcs", "arret", user=HSE)
        self.presence = {"parcs": {"etat": "absente", "camera_compromise": True, "source": "Engin mobile"}}
        state = self.run_seconds(2)["parcs"]
        self.assertEqual(state["niveau"], "ORANGE")
        self.assertIn("caméra compromise", state["raison"])

    def test_parameters_within_bounds_are_applied_and_audited(self):
        self.monitor.start(HSE)
        result = self.monitor.update_parameters({"pourcentage_orange": 60}, ADMIN)
        self.assertEqual(result["modifies"], {"pourcentage_orange": 60})
        stored = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["parametres"]["pourcentage_orange"], 60)
        self.assertEqual(self.monitor.evaluator.params["pourcentage_orange"], 60)
        entry = self.events("zones_parametres_modifies")[0]
        self.assertEqual((entry["utilisateur"], entry["data"]["champs"]["pourcentage_orange"]),
                         ("admin1", {"avant": 70, "apres": 60}))

    def test_out_of_bounds_or_normative_changes_are_refused_and_audited(self):
        from app.zones.settings import ConfigError
        for change in ({"pourcentage_orange": 100}, {"hysteresis_s": 3600}, {"SO2_ppm": 50}):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                self.monitor.update_parameters(change, ADMIN)
        refused = self.events("zones_parametres_refuses")
        self.assertEqual(len(refused), 3)
        self.assertEqual(refused[0]["utilisateur"], "admin1")
        stored = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["parametres"], CONFIG["parametres"])

    def test_invalid_scenario_is_refused(self):
        for args in (("sechage", "marche"), ("inconnue", "stable"), ("sechage", 3), ("sechage", "fuite", "x" * 100)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.monitor.set_scenario(*args, user=HSE)
        self.assertEqual(self.events("scenario_simulation_modifie"), [])

    def test_statistics_and_series(self):
        self.monitor.start(HSE)
        self.run_seconds(30)
        self.monitor.set_scenario("hangars", "fuite", user=HSE)
        self.run_seconds(30)
        now = self.clock.now.isoformat()
        video = [{"kind": "casque", "video": "homme_machine2.mp4", "wall_time": now},
                 {"kind": "proximite", "video": "homme_machine2.mp4", "wall_time": now},
                 {"kind": "obstruction", "video": "homme_machine2.mp4", "wall_time": now},
                 {"kind": "casque", "video": "test1.mp4", "wall_time": now}]
        stats = self.monitor.statistics(self.clock.now - timedelta(hours=1), self.clock.now, video)
        hangars, parcs = stats["zones"]["hangars"], stats["zones"]["parcs"]
        self.assertEqual(hangars["incidents"]["gaz"], 1)
        self.assertEqual(hangars["depassements"], {"CO_ppm": 1})
        self.assertGreater(hangars["temps"]["ROUGE"], 0)
        self.assertAlmostEqual(sum(hangars["temps"].values()), 61, delta=1)
        self.assertEqual((parcs["incidents"]["epi"], parcs["incidents"]["proximite"], parcs["incidents"]["camera"]),
                         (1, 1, 1))
        self.assertEqual(stats["classement"][0], "hangars")
        self.assertNotIn("poste_elec", stats["classement"])
        series = self.monitor.series("hangars-co", self.clock.now - timedelta(hours=1), self.clock.now)
        self.assertEqual((series["seuil_max"], series["orange"]["haut"], series["pas"]), (25, 17.5, "seconde"))
        self.assertGreaterEqual(len(series["points"]), 60)
        with self.assertRaises(LookupError):
            self.monitor.series("inconnu", self.clock.now, self.clock.now)

    def test_all_series_share_an_axis_starting_at_first_measure(self):
        before_start = self.clock.now
        self.monitor.start(HSE)
        self.run_seconds(90)
        hour = self.clock.now - timedelta(hours=1)
        curves = self.monitor.all_series(hour, self.clock.now)
        self.assertEqual([c["capteur"] for c in curves["capteurs"]],
                         ["sechage-so2", "sechage-co", "sechage-no2", "sechage-o2", "sechage-temp",
                          "cuves-lie", "cuves-temp", "hangars-o2", "hangars-co"])
        first = min(datetime.fromisoformat(c["points"][0][0]) for c in curves["capteurs"])
        self.assertEqual(datetime.fromisoformat(curves["debut"]), first)
        self.assertGreater(first, before_start - timedelta(seconds=5))
        self.assertEqual(curves["fin"], self.clock.now.isoformat(timespec="seconds"))
        lie = curves["capteurs"][5]
        self.assertEqual((lie["libelle"], lie["unite"], lie["seuil_max"], lie["zone_nom"]),
                         ("Explosivité", "% LIE", 5, "Cuves de combustible"))
        reduced = self.monitor.all_series(hour, self.clock.now, max_points=30)
        so2 = reduced["capteurs"][0]
        self.assertLessEqual(len(so2["points"]), 31)
        self.assertEqual(so2["regroupement_s"], 3)
        self.assertEqual(len(so2["points"][0]), 4)

    def test_all_series_without_data_falls_back_to_period_start(self):
        start = self.clock.now - timedelta(hours=1)
        curves = self.monitor.all_series(start, self.clock.now)
        self.assertEqual(curves["debut"], start.isoformat(timespec="seconds"))
        self.assertTrue(all(c["points"] == [] for c in curves["capteurs"]))

    def test_downsample_keeps_brief_peak(self):
        from app.zones.history import downsample
        base = datetime(2026, 1, 1, 12, 0, 0)
        points = [[(base + timedelta(seconds=s)).isoformat(), 9.0 if s == 37 else 1.0] for s in range(100)]
        self.assertEqual(downsample(points, 200), (points, None))
        reduced, step = downsample(points, 10)
        self.assertEqual((len(reduced), step), (10, 10))
        self.assertEqual(max(p[3] for p in reduced), 9.0)
        self.assertEqual(min(p[1] for p in reduced), 1.0)


class ZoneConsistencyTests(MonitorTestCase):
    def make_incident(self):
        self.monitor.start(HSE)
        self.run_seconds(5)
        self.monitor.set_scenario("cuves", "hors_plage", user=HSE)
        self.run_seconds(3)

    def check(self):
        from app.security.consistency import check_incidents
        return check_incidents(self.events(), self.alerts_path, self.root / "live",
                               self.monitor.history.incidents_path)

    def test_zone_incidents_are_consistent(self):
        self.make_incident()
        self.assertEqual(self.check(), [])

    def test_deleted_csv_line_is_detected(self):
        self.make_incident()
        lines = self.alerts_path.read_text(encoding="utf-8").splitlines()
        self.alerts_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        self.assertIn("ligne supprimée de alerts_log.csv", [a["detail"] for a in self.check()])

    def test_modified_or_deleted_history_is_detected(self):
        self.make_incident()
        path = self.monitor.history.incidents_path
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["kind"] = "machine"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.assertIn("incident modifié dans zones/incidents.jsonl", [a["detail"] for a in self.check()])
        path.unlink()
        self.assertIn("fichier supprimé ou vide", [a["detail"] for a in self.check()])


class LivePresenceTests(unittest.TestCase):
    def test_zone_of_associated_camera_is_unknown_without_analysis(self):
        from dashboard.live_api import LiveController
        controller = LiveController.__new__(LiveController)  # sans charger l'historique ni les modèles
        controller.session = None
        presence = controller.zone_presence()
        self.assertEqual(presence["parcs"]["etat"], "inconnue")
        self.assertEqual(presence["parcs"]["source"], "Engin mobile (démonstration)")


class ZonesHttpTests(unittest.TestCase):
    """API /api/zones/* sur un vrai serveur (rôles, origine, audit)."""

    def setUp(self):
        from unittest.mock import patch

        import dashboard.server as server
        from app.zones.monitor import ZoneMonitor
        from test_auth import HttpTestCase

        self.http = HttpTestCase("run")
        self.http.setUp()
        self.addCleanup(self.http.tearDown)
        config_path = self.http.root / "zones.json"
        config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
        self.monitor = ZoneMonitor(self.http.audit, self.http.root / "alerts_log.csv", self.http.root / "zones",
                                   config_path, presence=dict)
        item = patch.object(server, "ZONES", self.monitor)
        item.start()
        self.addCleanup(item.stop)

    def call(self, method, path, body=None, user=None):
        cookie = self.http.login(user) if user else None
        status, _, data = self.http.request(method, path, body, cookie=cookie)
        return status, json.loads(data) if data else None

    def test_status_requires_session_and_reports_rights(self):
        self.assertEqual(self.call("GET", "/api/zones/status")[0], 401)
        status, body = self.call("GET", "/api/zones/status", user="op1")
        self.assertEqual(status, 200)
        self.assertEqual((body["peut_simuler"], body["peut_regler"], body["actif"]), (False, False, False))
        self.assertEqual(len(body["zones"]), 7)
        body = self.call("GET", "/api/zones/status", user="hse1")[1]
        self.assertEqual((body["peut_simuler"], body["peut_regler"]), (True, False))

    def test_operator_cannot_drive_simulation_through_api(self):
        for body in ({"action": "demarrer"}, {"action": "scenario", "zone": "cuves", "scenario": "fuite"}):
            self.assertEqual(self.call("POST", "/api/zones/simulation", body, "op1")[0], 403)
        self.assertFalse(self.monitor.provider.running)

    def test_hse_drives_simulation_and_is_audited(self):
        status, body = self.call("POST", "/api/zones/simulation", {"action": "demarrer"}, "hse1")
        self.assertEqual((status, body["actif"]), (200, True))
        status, body = self.call("POST", "/api/zones/simulation",
                                 {"action": "scenario", "zone": "cuves", "scenario": "fuite"}, "hse1")
        self.assertEqual((status, body["nouveau"]), (200, "fuite"))
        self.assertEqual(self.call("POST", "/api/zones/simulation", {"action": "demarrer"}, "hse1")[0], 409)
        self.assertEqual(self.call("POST", "/api/zones/simulation",
                                   {"action": "scenario", "zone": "cuves", "scenario": "marche"}, "hse1")[0], 400)
        status, body = self.call("POST", "/api/zones/simulation", {"action": "arreter"}, "hse1")
        self.assertEqual((status, body["arret"]["par"]), (200, "hse1"))

    def test_parameters_are_admin_only_and_bounded(self):
        body = {"parametres": {"hysteresis_s": 20}}
        self.assertEqual(self.call("POST", "/api/zones/parametres", body, "op1")[0], 403)
        self.assertEqual(self.call("POST", "/api/zones/parametres", body, "hse1")[0], 403)
        status, result = self.call("POST", "/api/zones/parametres", body, "admin1")
        self.assertEqual((status, result["modifies"]), (200, {"hysteresis_s": 20}))
        status, result = self.call("POST", "/api/zones/parametres", {"parametres": {"pourcentage_orange": 95}},
                                   "admin1")
        self.assertEqual(status, 400)
        self.assertIn("bornes sûres", result["error"])
        from app.security.audit_log import read_entries
        events = [e["event"] for e in read_entries(self.http.audit.path)]
        self.assertIn("zones_parametres_modifies", events)
        self.assertIn("zones_parametres_refuses", events)

    def test_simultaneous_sessions_share_state_and_audit_each_author(self):
        from app.security.audit_log import read_entries
        hse, admin, op = (self.http.login(user) for user in ("hse1", "admin1", "op1"))
        hse_bis = self.http.login("hse1")  # même compte ouvert dans un second navigateur

        def post(cookie, path, body):
            return self.http.request("POST", path, body, cookie=cookie)[0]

        def view(cookie):
            body = json.loads(self.http.request("GET", "/api/zones/status", cookie=cookie)[2])
            return (body["actif"], body["demarrage"], body["incidents_ouverts"],
                    [(z["zone"], z["niveau"], z.get("scenario")) for z in body["zones"]])

        self.assertEqual(post(hse, "/api/zones/simulation", {"action": "demarrer"}), 200)
        self.assertEqual(post(admin, "/api/zones/simulation",
                              {"action": "scenario", "zone": "cuves", "scenario": "hors_plage"}), 200)
        self.assertEqual(post(hse_bis, "/api/zones/simulation",
                              {"action": "scenario", "zone": "sechage", "scenario": "derive"}), 200)
        self.assertEqual(post(admin, "/api/zones/parametres", {"parametres": {"hysteresis_s": 20}}), 200)
        self.assertEqual(post(op, "/api/zones/simulation", {"action": "arreter"}), 403)
        now = datetime.now().astimezone()
        for second in range(1, 6):
            self.monitor.tick(now + timedelta(seconds=second))
        states = [view(cookie) for cookie in (hse, admin, op, hse_bis)]
        self.assertTrue(all(state == states[0] for state in states))
        self.assertEqual((states[0][0], states[0][1]["par"]), (True, "hse1"))
        self.assertTrue(states[0][2])  # incident ouvert, vu par les quatre sessions
        # se déconnecter dans un navigateur ne ferme pas l'autre session du même compte
        self.assertEqual(post(hse, "/api/logout", {}), 200)
        self.assertEqual(self.http.request("GET", "/api/zones/status", cookie=hse)[0], 401)
        self.assertEqual(post(hse_bis, "/api/zones/simulation", {"action": "arreter"}), 200)
        self.assertEqual(view(op)[0], False)
        authors = [(e["event"], e["utilisateur"], e["role"]) for e in read_entries(self.http.audit.path)
                   if e["event"] in ("surveillance_zones_demarree", "scenario_simulation_modifie",
                                     "zones_parametres_modifies", "deconnexion", "surveillance_zones_arretee")]
        self.assertEqual(authors, [
            ("surveillance_zones_demarree", "hse1", "responsable_hse"),
            ("scenario_simulation_modifie", "admin1", "admin"),
            ("scenario_simulation_modifie", "hse1", "responsable_hse"),
            ("zones_parametres_modifies", "admin1", "admin"),
            ("deconnexion", "hse1", "responsable_hse"),
            ("surveillance_zones_arretee", "hse1", "responsable_hse"),
        ])

    def test_cross_origin_post_is_refused(self):
        cookie = self.http.login("hse1")
        status, _, _ = self.http.request("POST", "/api/zones/simulation", {"action": "demarrer"}, cookie=cookie,
                                         headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertFalse(self.monitor.provider.running)

    def test_zones_page_requires_session_and_uses_nonce_csp_without_external_resources(self):
        status, headers, _ = self.http.request("GET", "/zones")
        self.assertEqual((status, headers["Location"]), (303, "/login?next=/zones"))
        cookie = self.http.login("op1")
        status, headers, body = self.http.request("GET", "/zones", cookie=cookie)
        self.assertEqual(status, 200)
        csp = headers["Content-Security-Policy"]
        self.assertIn("script-src 'self' 'nonce-", csp)
        self.assertNotIn("unsafe-inline' 'nonce", csp)
        html = body.decode("utf-8")
        nonce = csp.split("'nonce-")[1].split("'")[0]
        self.assertIn(f'<script nonce="{nonce}">', html)
        for forbidden in ("http://", "https://", "onclick=", "<script src"):
            self.assertNotIn(forbidden, html.replace("http://www.w3.org/2000/svg", ""))
        self.assertIn('href="/live"', html)

    def test_live_page_links_to_zones(self):
        status, _, body = self.http.request("GET", "/live", cookie=self.http.login("op1"))
        self.assertEqual(status, 200)
        self.assertIn('href="/zones"', body.decode("utf-8"))

    def test_status_exposes_plan_and_start_info(self):
        self.call("POST", "/api/zones/simulation", {"action": "demarrer"}, "hse1")
        body = self.call("GET", "/api/zones/status", user="op1")[1]
        self.assertEqual(body["demarrage"]["par"], "hse1")
        self.assertIsNone(body["arret"])
        sechage = next(z for z in body["zones"] if z["zone"] == "sechage")
        self.assertGreaterEqual(len(sechage["plan"]["points"]), 3)
        self.assertEqual([c["id"] for c in sechage["liste_capteurs"]][:2], ["sechage-so2", "sechage-co"])

    def test_stats_and_series_endpoints(self):
        status, body = self.call("GET", "/api/zones/stats?periode=jour", user="op1")
        self.assertEqual((status, body["periode"]), (200, "jour"))
        self.assertIn("classement", body)
        self.assertEqual(self.call("GET", "/api/zones/stats?periode=annee", user="op1")[0], 400)
        status, body = self.call("GET", "/api/zones/series?capteur=sechage-so2&periode=heure", user="op1")
        self.assertEqual((status, body["seuil_max"]), (200, 2))
        self.assertEqual(self.call("GET", "/api/zones/series?capteur=x&periode=heure", user="op1")[0], 404)
        status, body = self.call("GET", "/api/zones/courbes?periode=jour", user="op1")
        self.assertEqual((status, body["periode"], len(body["capteurs"])), (200, "jour", 9))
        self.assertEqual(self.call("GET", "/api/zones/courbes?periode=annee", user="op1")[0], 400)
        self.assertEqual(self.http.request("GET", "/api/zones/courbes?periode=heure")[0], 401)


if __name__ == "__main__":
    unittest.main()
