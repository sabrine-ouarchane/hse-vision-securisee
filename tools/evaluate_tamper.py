"""
tools/evaluate_tamper.py

Évalue le détecteur de sabotage caméra (app/security/tamper_detector.py),
sans YOLO, à la cadence d'analyse du temps réel (≈ 2 images/s par défaut,
avec une gigue aléatoire comme sur un poste CPU) :

- vidéos attaquées (data/tamper_videos, vérité terrain JSON) : détection du
  type attendu oui/non, délai de détection (s), autres alertes émises ;
- vidéos propres (data/test_videos) : fausses alertes, y compris en lecture
  en boucle (historique de rejeu remis à zéro à chaque tour, comme en direct).

Résultats : tableau dans le terminal + CSV (data/outputs/tamper_evaluation.csv).

Usage :
    python tools/evaluate_tamper.py
    python tools/evaluate_tamper.py --hz 1.5 --seeds 3 --loops 3
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import ROOT, load_camera_profile, load_detection_rules  # noqa: E402
from app.security.tamper_detector import TamperDetector  # noqa: E402

TAMPER_DIR = ROOT / "data" / "tamper_videos"
CLEAN_DIR = ROOT / "data" / "test_videos"
DEFAULT_CSV = ROOT / "data" / "outputs" / "tamper_evaluation.csv"


def analyse(path, profile, cfg, hz=2.0, seed=0, loops=1):
    """
    Lit la vidéo comme la session temps réel : une image analysée toutes les
    ~1/hz secondes (gigue ±20 %). Renvoie la liste des sabotages confirmés
    (type, instant de confirmation, début estimé) et le temps moyen par image.
    """
    rng = random.Random(seed)
    detector = TamperDetector(cfg, profile)
    confirmed, seen, durations = [], set(), []
    offset = 0.0
    for loop in range(loops):
        if loop:
            detector.reset_replay()  # nouveau tour : la relecture volontaire n'est pas un rejeu
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        index, next_t, length = 0, rng.uniform(0, 1 / hz), 0.0
        while True:
            target = int(next_t * fps)
            ok = True
            while index < target and ok:
                ok = capture.grab()
                index += 1
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            t = offset + target / fps
            started = time.perf_counter()
            result = detector.update(frame, t)
            durations.append((time.perf_counter() - started) * 1000)
            for event in result["active"]:
                key = (event["type"], event["since"])
                if key not in seen:
                    seen.add(key)
                    confirmed.append({"type": event["type"], "confirmed_t": round(t, 2), "since": event["since"],
                                      "detail": event["detail"]})
            length = target / fps
            next_t += (1 / hz) * rng.uniform(0.8, 1.2)
        capture.release()
        offset += length + 1 / hz
    return confirmed, (sum(durations) / len(durations) if durations else 0.0), offset


def evaluate_attacks(cfg, hz, seeds, attacks=None):
    rows = []
    for truth_path in sorted(TAMPER_DIR.glob("*.json")):
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if attacks and truth["attack"] not in attacks:
            continue
        video = truth_path.with_suffix(".mp4")
        if not video.is_file():
            continue
        profile = load_camera_profile(truth["source_file"])
        mobile = not profile.get("camera_fixe", True)
        for seed in range(seeds):
            events, ms, _ = analyse(video, profile, cfg, hz, seed)
            expected = truth["expected_type"]
            # Détection valable : type attendu confirmé pendant l'attaque (ou dans la marge de confirmation)
            hits = [e for e in events if e["type"] == expected
                    and truth["start_s"] - 0.5 <= e["confirmed_t"] <= truth["end_s"] + cfg.get("confirm_s", 2.5)]
            others = [e for e in events if e not in hits]
            # Non évaluable : déplacement sur caméra mobile, ou attaque plus courte que confirmation + 1 s
            too_short = truth["end_s"] - truth["start_s"] < cfg.get("confirm_s", 2.5) + 1.0
            evaluable = not (expected == "deplacement" and mobile) and not too_short
            rows.append({
                "jeu": "attaque", "video": video.name, "source": truth["source_file"], "attaque": truth["attack"],
                "type_attendu": expected, "graine": seed,
                "debut_s": truth["start_s"], "fin_s": truth["end_s"],
                "detecte": ("oui" if hits else "non") if evaluable else
                           ("non évaluable (attaque trop courte)" if too_short else "non évaluable (caméra mobile)"),
                "delai_s": round(hits[0]["confirmed_t"] - truth["start_s"], 2) if hits else "",
                "autres_alertes": "; ".join(f"{e['type']}@{e['confirmed_t']}s" for e in others),
                "ms_par_image": round(ms, 2),
            })
            print(f"  {video.name:38s} graine {seed} : {rows[-1]['detecte']:4s} "
                  f"délai {rows[-1]['delai_s'] or '—'} s · autres : {rows[-1]['autres_alertes'] or 'aucune'}", flush=True)
    return rows


def evaluate_clean(cfg, hz, seeds, loops):
    rows = []
    for video in sorted(CLEAN_DIR.glob("*.mp4")):
        if "__" in video.stem:
            continue
        profile = load_camera_profile(video.name)
        for seed in range(seeds):
            events, ms, duration = analyse(video, profile, cfg, hz, seed, loops)
            rows.append({
                "jeu": "propre", "video": video.name, "source": video.name, "attaque": "aucune",
                "type_attendu": "aucun", "graine": seed, "debut_s": "", "fin_s": round(duration, 1),
                "detecte": f"{len(events)} fausse(s) alerte(s)", "delai_s": "",
                "autres_alertes": "; ".join(f"{e['type']}@{e['confirmed_t']}s ({e['detail']})" for e in events),
                "ms_par_image": round(ms, 2),
                "camera_fixe": profile.get("camera_fixe", True),
            })
            print(f"  {video.name:38s} graine {seed} ({duration:.0f} s, {loops} tour(s), caméra "
                  f"{'fixe' if profile.get('camera_fixe', True) else 'mobile'}) : {rows[-1]['detecte']} "
                  f"{rows[-1]['autres_alertes']}", flush=True)
    return rows


def summary(attack_rows, clean_rows):
    lines = ["", "Taux de détection par type d'attaque", "-" * 78,
             f"{'Attaque':16s}{'Type attendu':16s}{'Détectées':>12s}{'Taux':>8s}{'Délai moy.':>12s}{'Délai max':>11s}"]
    for attack in sorted({r["attaque"] for r in attack_rows}):
        rows = [r for r in attack_rows if r["attaque"] == attack and r["detecte"] in ("oui", "non")]
        excluded = sum(1 for r in attack_rows if r["attaque"] == attack and r["detecte"] not in ("oui", "non"))
        detected = [r for r in rows if r["detecte"] == "oui"]
        delays = [r["delai_s"] for r in detected]
        rate = f"{len(detected) / len(rows):.0%}" if rows else "—"
        mean = f"{sum(delays) / len(delays):.1f} s" if delays else "—"
        worst = f"{max(delays):.1f} s" if delays else "—"
        note = f"  ({excluded} non évaluable(s) : caméra mobile ou vidéo trop courte)" if excluded else ""
        expected = attack_rows[[r["attaque"] for r in attack_rows].index(attack)]["type_attendu"]
        lines.append(f"{attack:16s}{expected:16s}{f'{len(detected)}/{len(rows)}':>12s}{rate:>8s}{mean:>12s}{worst:>11s}{note}")
    other = sum(1 for r in attack_rows if r["autres_alertes"])
    lines.append(f"Vidéos attaquées avec une alerte d'un autre type : {other}/{len(attack_rows)}")
    if not clean_rows:
        return "\n".join(lines)
    total_minutes = sum(r["fin_s"] for r in clean_rows) / 60
    false_alarms = sum(int(r["detecte"].split()[0]) for r in clean_rows)
    lines += ["", "Fausses alertes sur vidéos propres", "-" * 78,
              f"{len(clean_rows)} lecture(s), {total_minutes:.1f} min analysées : {false_alarms} fausse(s) alerte(s) "
              f"({false_alarms / total_minutes if total_minutes else 0:.2f} / min)"]
    for row in clean_rows:
        if row["autres_alertes"]:
            lines.append(f"  {row['video']} (graine {row['graine']}) : {row['autres_alertes']}")
    all_ms = [r["ms_par_image"] for r in attack_rows + clean_rows]
    if all_ms:
        lines.append(f"\nCoût moyen du contrôle d'intégrité : {sum(all_ms) / len(all_ms):.1f} ms par image analysée")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Évaluation du détecteur de sabotage caméra.")
    parser.add_argument("--hz", type=float, default=2.0, help="Cadence d'analyse simulée (images/s).")
    parser.add_argument("--seeds", type=int, default=2, help="Nombre de tirages de la gigue d'échantillonnage.")
    parser.add_argument("--loops", type=int, default=3, help="Tours de lecture des vidéos propres (mode boucle).")
    parser.add_argument("--attacks", nargs="+", help="N'évaluer que ces attaques (ex. rejeu deplacement).")
    parser.add_argument("--skip-clean", action="store_true", help="Ne pas rejouer les vidéos propres.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    cfg = load_detection_rules()["tamper"]
    print(f"Vidéos attaquées ({TAMPER_DIR.relative_to(ROOT)}) :")
    attack_rows = evaluate_attacks(cfg, args.hz, args.seeds, args.attacks)
    clean_rows = []
    if not args.skip_clean:
        print(f"Vidéos propres ({CLEAN_DIR.relative_to(ROOT)}, {args.loops} tour(s)) :")
        clean_rows = evaluate_clean(cfg, args.hz, args.seeds, args.loops)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["jeu", "video", "source", "attaque", "type_attendu", "graine", "debut_s", "fin_s", "detecte",
              "delai_s", "autres_alertes", "ms_par_image", "camera_fixe"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", delimiter=";")
        writer.writeheader()
        writer.writerows(attack_rows + clean_rows)
    print(summary(attack_rows, clean_rows))
    print(f"\nCSV : {csv_path}")


if __name__ == "__main__":
    main()
