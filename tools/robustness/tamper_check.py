"""
tools/robustness/tamper_check.py — Complément de la partie 2.

Question : quand une perturbation naturelle fait perdre ses détections au
modèle, le détecteur de sabotage (app/security/tamper_detector.py)
prévient-il que la surveillance n'est plus fiable ?

Pour chaque image de l'échantillon : le détecteur apprend sa référence sur
l'image propre (comme les 2 premières secondes d'une session), puis reçoit
l'image perturbée. On relève les indices levés sur CETTE image (« suspecté »),
sans la confirmation temporelle de 2,5 s : c'est la borne haute de ce que le
système pourrait signaler. Témoin : l'image propre elle-même (fausses alertes).
Le contrôle de déplacement est désactivé (caméra déclarée mobile), le flux
figé est ignoré (la même image est répétée volontairement).

Usage : python tools/robustness/tamper_check.py
Sortie : data/outputs/robustness/partie2_sabotage.csv
"""

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import perturbations as P  # noqa: E402
from app.security.tamper_detector import TamperDetector  # noqa: E402

KINDS = ["bruit_gaussien", "flou", "poussiere", "contraste"]
SEVERITIES = [3, 4, 5]
WATCHED = ("obstruction", "image_degradee")


def main():
    timer = C.Timer()
    items = C.load_items("test")
    duplicates = C.near_duplicates(items)
    sample = set(C.fixed_sample(items)["echantillon"])
    subsets = C.subset_names(sorted(sample), duplicates)
    cfg = C.RULES["tamper"]
    flags = {}  # (condition, sévérité) -> {image: set(types)}
    for item in (i for i in items if i["name"] in sample):
        image, _, true_xyxy = (lambda im: (im, *C.labels_xyxy(item["labels"], im.shape[1], im.shape[0])))(
            C.read_image(item["path"]))
        conditions = [("propre", 0, image)]
        for kind in KINDS:
            for severity in SEVERITIES:
                rng = np.random.default_rng([C.SEED, zlib.crc32(f"{item['name']}|{kind}|{severity}".encode())])
                conditions.append((kind, severity, P.apply(image, kind, severity, rng, boxes=true_xyxy)))
        for kind, severity, perturbed in conditions:
            probe = TamperDetector(cfg, {"camera_fixe": False, "tamper_reference": None})
            for t in (0.0, 0.5, 1.0, 1.5):
                probe.update(image, t)  # référence apprise sur l'image propre
            result = probe.update(perturbed, 2.5)
            flags.setdefault((kind, severity), {})[item["name"]] = set(result["suspected"]) & set(WATCHED)
    rows = []
    for (kind, severity), per_image in flags.items():
        for subset, names in subsets.items():
            selected = [per_image[n] for n in names if n in per_image]
            count = len(selected)
            rows.append({
                "condition": kind, "severite": severity, "sous_ensemble": subset, "images": count,
                "signalees": sum(1 for s in selected if s),
                "taux_signalees": round(sum(1 for s in selected if s) / count, 3) if count else "",
                "obstruction": sum(1 for s in selected if "obstruction" in s),
                "image_degradee": sum(1 for s in selected if "image_degradee" in s),
            })
    C.write_csv(C.OUT / "partie2_sabotage.csv", rows)
    for row in rows:
        if row["sous_ensemble"] == "test_complet":
            print(f"{row['condition']:15s} s{row['severite']} : {row['signalees']}/{row['images']} signalées "
                  f"(obstruction {row['obstruction']}, image dégradée {row['image_degradee']})")
    print(f"Terminé en {timer}.")


if __name__ == "__main__":
    main()
