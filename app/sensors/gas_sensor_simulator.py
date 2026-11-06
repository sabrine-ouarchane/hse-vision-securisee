"""
GasSensorSimulator (v1)

Simule des lectures de capteurs gaz (O2, SO2, H2S, CO, NH3, explosivité, T°)
par zone, en l'absence de capteurs réels sur site (cf. cahier des charges
section 5/12). Les valeurs de base respectent les seuils réels S-HSE-23 ;
un motif d'excursion (pic de CO) est intégré pour permettre de démontrer le
déclenchement d'alerte, et se répète en boucle quelle que soit la durée de
la vidéo traitée.

Même logique de conception que MachineStateSimulator : une classe simple,
appelée à chaque frame avec un temps écoulé (secondes), pas de datetime réel.
"""

import pandas as pd

THRESHOLDS = {
    "O2_pct":  {"min": 19.5, "max": 23.5},
    "SO2_ppm": {"max": 2},
    "H2S_ppm": {"max": 10},
    "CO_ppm":  {"max": 25},
    "NH3_ppm": {"max": 25},
    "Explosivity_pct_LIE": {"max": 5},
    "Temp_C":  {"max": 35},
}


class GasSensorSimulator:
    def __init__(self, csv_path="data/sensors/gas_sensors_simulated.csv"):
        self.df = pd.read_csv(csv_path)
        self.loop_duration_s = self.df["Elapsed_s"].max() + (
            self.df["Elapsed_s"].iloc[1] - self.df["Elapsed_s"].iloc[0]
        )

    def get_readings(self, zone_id: str, elapsed_seconds: float) -> dict:
        """Retourne les valeurs gaz les plus proches du temps écoulé donné, pour une zone.
        Le motif de données boucle automatiquement (modulo) si la vidéo est plus longue
        que la série simulée."""
        t = elapsed_seconds % self.loop_duration_s
        zone_data = self.df[self.df.Zone_ID == zone_id]
        if zone_data.empty:
            return None
        idx = (zone_data.Elapsed_s - t).abs().idxmin()
        row = zone_data.loc[idx]
        return row.drop(["Elapsed_s", "Zone_ID"]).to_dict()

    def is_out_of_threshold(self, zone_id: str, elapsed_seconds: float):
        """Retourne (bool, raison, lectures) : True si un seuil S-HSE-23 est dépassé."""
        readings = self.get_readings(zone_id, elapsed_seconds)
        if readings is None:
            return False, None, {}

        if not (THRESHOLDS["O2_pct"]["min"] <= readings["O2_pct"] <= THRESHOLDS["O2_pct"]["max"]):
            return True, f"O2 hors plage ({readings['O2_pct']}%)", readings
        for gas in ("SO2_ppm", "H2S_ppm", "CO_ppm", "NH3_ppm"):
            if readings[gas] > THRESHOLDS[gas]["max"]:
                return True, f"{gas} au-dessus du seuil ({readings[gas]})", readings
        if readings["Explosivity_pct_LIE"] > THRESHOLDS["Explosivity_pct_LIE"]["max"]:
            return True, f"Explosivité au-dessus du seuil ({readings['Explosivity_pct_LIE']}%)", readings
        if readings["Temp_C"] > THRESHOLDS["Temp_C"]["max"]:
            return True, f"Température au-dessus du seuil ({readings['Temp_C']}°C)", readings

        return False, None, readings
