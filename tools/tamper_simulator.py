"""
tools/tamper_simulator.py

Fabrique, à partir d'une vidéo propre, une version « attaquée » par type de
sabotage caméra, et écrit à côté la vérité terrain (type, début, fin) en JSON.

Attaques produites (à l'instant --start, en secondes) :
- obstruction   : rectangle noir (ou zone très floue avec --obstruction-style flou)
                  sur 70 % de l'image pendant --duration ;
- deplacement   : caméra décalée (--shift, 20 % de la largeur) et tournée
                  (--angle, 12°) jusqu'à la fin ;
- flou          : flou progressif (3 s) puis maintenu pendant --duration ;
- eblouissement : zone saturée (reflet / lampe) sur ~45 % de l'image ;
- flux_fige     : une image répétée pendant 15 s (insérée), puis reprise ;
- rejeu         : les --replay-s dernières secondes sont rejouées (insérées) ;
                  inséré au plus tôt à 2 × --replay-s pour disposer d'un historique.

Les vidéos sont écrites à 1280 px de côté max (comme la copie analysée en
direct) dans data/tamper_videos/ ; --out data/test_videos les rend
sélectionnables dans le dashboard pour une démonstration.

Usage :
    python tools/tamper_simulator.py                                   # toutes les vidéos de test
    python tools/tamper_simulator.py --input data/test_videos/test1.mp4 --types obstruction rejeu --start 4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import ROOT  # noqa: E402

ATTACKS = ("obstruction", "deplacement", "flou", "eblouissement", "flux_fige", "rejeu")
# Type de sabotage attendu du détecteur pour chaque attaque simulée
EXPECTED = {
    "obstruction": "obstruction", "deplacement": "deplacement", "flou": "image_degradee",
    "eblouissement": "image_degradee", "flux_fige": "flux_fige", "rejeu": "rejeu",
}
DEFAULT_OUT = ROOT / "data" / "tamper_videos"
FREEZE_S = 15.0


def output_size(width, height, max_side=1280):
    scale = min(1.0, max_side / max(width, height))
    return int(width * scale) // 2 * 2, int(height * scale) // 2 * 2


def obstruct(frame, style="noir"):
    height, width = frame.shape[:2]
    x2 = int(width * 0.7)
    if style == "flou":  # objectif couvert d'un film translucide : plus aucune texture
        small = cv2.resize(frame[:, :x2], (6, 6), interpolation=cv2.INTER_AREA)
        frame[:, :x2] = cv2.resize(small, (x2, height), interpolation=cv2.INTER_LINEAR)
    else:
        frame[:, :x2] = 0
    return frame


def displace(frame, shift=0.20, angle=12.0):
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    matrix[0, 2] += width * shift
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def blur(frame, progress):
    sigma = 0.5 + 11.5 * min(1.0, progress)
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def glare(frame):
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, (int(width * 0.55), int(height * 0.4)), (int(width * 0.42), int(height * 0.38)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(3, width / 60)).astype(np.float32)[..., None] / 255
    return (frame * (1 - mask) + 255 * mask).clip(0, 255).astype(np.uint8)


def simulate(input_path, attack, out_dir=DEFAULT_OUT, start=3.0, duration=6.0, replay_s=5.0,
             obstruction_style="noir", max_side=1280, shift=0.20, angle=12.0) -> Path:
    """Écrit <vidéo>__<attaque>.mp4 et sa vérité terrain <vidéo>__<attaque>.json ; renvoie le chemin vidéo."""
    if attack not in ATTACKS:
        raise ValueError(f"Attaque inconnue : {attack} ({', '.join(ATTACKS)})")
    input_path = Path(input_path)
    probe = cv2.VideoCapture(str(input_path))
    try:
        if not probe.isOpened():
            raise FileNotFoundError(f"Vidéo illisible : {input_path}")
        fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        size = output_size(int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)), max_side)
    finally:
        probe.release()
    length_s = total / fps

    # L'attaque doit laisser le temps d'établir la référence (début) et de confirmer (fin)
    if attack == "flux_fige":
        start = min(start, length_s)
    elif attack == "rejeu":
        start = min(max(start, 2 * replay_s), length_s - 0.5)  # inséré avant la dernière image
        replay_s = min(replay_s, start)
        if replay_s < 3.5:
            raise ValueError(f"{input_path.name} : trop court pour rejouer au moins 3,5 s (confirmation 2,5 s).")
    else:
        start = max(2.5, min(start, length_s - 3.0))
        if length_s - start < 4.0:
            print(f"[ATTENTION] {input_path.name} : attaque de {length_s - start:.1f} s seulement "
                  f"(vidéo courte) — à peine plus que le délai de confirmation.")
    start_frame = int(round(start * fps))

    capture = cv2.VideoCapture(str(input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{input_path.stem}__{attack}.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    written = 0

    def write(frame):
        nonlocal written
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    end_s = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            t = index / fps
            if index == start_frame and attack == "flux_fige":
                for _ in range(int(round(FREEZE_S * fps))):
                    write(frame)
                end_s = written / fps
            elif index == start_frame and attack == "rejeu":
                # Seconde lecture de la vidéo pour rejouer le segment [start - replay_s, start)
                replay = cv2.VideoCapture(str(input_path))
                replay.set(cv2.CAP_PROP_POS_FRAMES, start_frame - int(round(replay_s * fps)))
                for _ in range(int(round(replay_s * fps))):
                    ok_replay, old = replay.read()
                    if not ok_replay:
                        break
                    write(old)
                replay.release()
                end_s = written / fps
            in_window = start <= t < start + duration
            if attack == "obstruction" and in_window:
                frame = obstruct(frame, obstruction_style)
            elif attack == "deplacement" and t >= start:
                frame = displace(frame, shift, angle)
            elif attack == "flou" and in_window:
                frame = blur(frame, (t - start) / 3.0)
            elif attack == "eblouissement" and in_window:
                frame = glare(frame)
            write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()

    total_s = written / fps
    if end_s is None:
        end_s = total_s if attack == "deplacement" else min(start + duration, total_s)
    truth = {
        "source_file": input_path.name,
        "attack": attack,
        "expected_type": EXPECTED[attack],
        "start_s": round(start, 2),
        "end_s": round(end_s, 2),
        "video_duration_s": round(total_s, 2),
        "fps": round(fps, 2),
        "size": list(size),
        "params": {
            "duration_s": duration, "replay_s": replay_s if attack == "rejeu" else None,
            "freeze_s": FREEZE_S if attack == "flux_fige" else None,
            "obstruction_style": obstruction_style if attack == "obstruction" else None,
            "shift": shift if attack == "deplacement" else None,
            "angle_deg": angle if attack == "deplacement" else None,
        },
    }
    target.with_suffix(".json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description="Simulateur d'attaques visuelles sur caméra (POC local).")
    parser.add_argument("--input", nargs="+", help="Vidéo(s) propre(s) ; par défaut toutes celles de data/test_videos.")
    parser.add_argument("--types", nargs="+", default=list(ATTACKS), choices=ATTACKS)
    parser.add_argument("--start", type=float, default=3.0, help="Début de l'attaque (s).")
    parser.add_argument("--duration", type=float, default=6.0, help="Durée des attaques en place (s).")
    parser.add_argument("--replay-s", type=float, default=5.0, help="Longueur du segment rejoué (s).")
    parser.add_argument("--obstruction-style", choices=("noir", "flou"), default="noir")
    parser.add_argument("--shift", type=float, default=0.20, help="Déplacement : décalage (fraction de la largeur).")
    parser.add_argument("--angle", type=float, default=12.0, help="Déplacement : rotation (degrés).")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Dossier de sortie (data/test_videos pour la démo).")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input] if args.input else sorted((ROOT / "data" / "test_videos").glob("*.mp4"))
    inputs = [p for p in inputs if "__" not in p.stem]  # pas de vidéo déjà attaquée en entrée
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    for path in inputs:
        for attack in args.types:
            try:
                target = simulate(path, attack, out_dir, args.start, args.duration, args.replay_s,
                                  args.obstruction_style, shift=args.shift, angle=args.angle)
            except ValueError as error:
                print(f"[IGNORÉ] {error}")
                continue
            truth = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
            print(f"[OK] {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target} : "
                  f"{attack} de {truth['start_s']} s à {truth['end_s']} s")


if __name__ == "__main__":
    main()
