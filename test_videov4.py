"""
test_videov4.py

Détection HSE sur une vidéo ou une caméra : comportements dangereux
(casque, gilet, harnais, téléphone) et situations homme-machine.

Deux modes :
- par défaut, chaque image est analysée puis écrite dans une vidéo annotée
  (évaluation reproductible, indépendante de la vitesse du PC) ;
- --realtime : la source est lue à sa cadence réelle et seule la dernière
  image disponible est analysée (comme une caméra en direct sur un PC sans
  GPU) ; les images non analysées sont ignorées et comptées.

Le zonage général du site (gaz, polygones de zones) n'intervient pas ici.

Exemples :
    python test_videov4.py --input data/test_videos/just_phone.mp4
    python test_videov4.py --input data/test_videos/homme_machine2.mp4 --machine-state marche
    python test_videov4.py --input 0 --realtime --show          # webcam
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from app.config import ROOT
from app.live.pipeline import HSEPipeline
from app.sensors.machine_state_provider import MachineStateProvider

DEFAULT_INPUT = "data/test_videos/homme_machine2.mp4"


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    """H.264 (lisible dans un navigateur) si disponible, sinon MPEG-4."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Impossible de créer la vidéo de sortie : {path}")


def main(args):
    source = int(args.input) if args.input.isdigit() else args.input
    is_camera = isinstance(source, int) or str(source).startswith(("rtsp://", "http://", "https://"))
    if not is_camera and not Path(source).is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    source_name = f"camera_{source}" if isinstance(source, int) else Path(str(source)).name
    stem = Path(source_name).stem

    output = Path(args.output or ROOT / "data" / "outputs" / f"{stem}_hse.mp4")
    alerts = Path(args.alerts or output.with_name(f"{stem}_alerts.csv"))
    report = output.with_name(f"{stem}_rapport.json")
    snapshots = output.with_name(f"{stem}_incidents")
    output.parent.mkdir(parents=True, exist_ok=True)
    if alerts.exists():
        alerts.unlink()  # un journal par exécution

    machine_states = MachineStateProvider()
    pipeline = HSEPipeline(
        source_name,
        alerts_path=alerts,
        snapshot_dir=snapshots,
        machine_states=machine_states,
        harness_required=True if args.harness_required else None,
        show_raw=not args.hide_raw,
    )
    if args.machine_state:
        machine_states.set_state(pipeline.machine_id, args.machine_state, source="ligne de commande",
                                 unknown_policy=args.unknown_machine_policy)
    elif args.unknown_machine_policy:
        state = machine_states.get(pipeline.machine_id)["state"]
        machine_states.set_state(pipeline.machine_id, state, source="ligne de commande",
                                 unknown_policy=args.unknown_machine_policy)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source : {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h, _ = pipeline.display_size(width, height)
    print(f"[INFO] Source : {source_name} — {width}x{height} @ {fps:.1f} img/s, {total or '∞'} images")
    print(f"[INFO] Capacités du modèle : {pipeline.detector.capabilities()}")
    print(f"[INFO] Engin '{pipeline.machine_id}' : {machine_states.get(pipeline.machine_id)}, "
          f"politique état inconnu = {machine_states.unknown_policy}")

    # En temps réel, la vidéo de sortie contient les images analysées à leur cadence effective.
    writer, codec = open_writer(output, fps if not args.realtime else 4.0, (out_w, out_h))
    if codec != "avc1":
        print("[ATTENTION] H.264 indisponible : vidéo en MPEG-4 (lecture navigateur impossible, utiliser VLC).")

    frame_idx = analysed = skipped = 0
    started = time.perf_counter()
    processing_times = []
    analysis_fps = 0.0
    try:
        while True:
            if args.realtime and not is_camera:
                # Saute les images qu'une caméra aurait produites pendant l'analyse précédente
                target = int((time.perf_counter() - started) * fps)
                if target > frame_idx + 1:
                    skip = target - frame_idx - 1
                    for _ in range(skip):
                        if not capture.grab():
                            break
                    frame_idx += skip
                    skipped += skip
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and analysed >= args.max_frames:
                break
            now = time.perf_counter() - started if is_camera else frame_idx / fps
            result = pipeline.process(frame, now, frame_idx, fps=analysis_fps,
                                      timestamp=f"t = {now:.1f} s" if not is_camera else None)
            analysed += 1
            processing_times.append(result["total_ms"])
            recent = processing_times[-10:]
            analysis_fps = 1000 / (sum(recent) / len(recent))
            writer.write(result["annotated"])

            for incident in result["opened"]:
                print(f"[INCIDENT #{incident['id']}] t={now:.1f}s {incident['label']} — {incident['message']}")
            if args.show:
                cv2.imshow("HSE — détection temps réel (q pour quitter)", result["annotated"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if analysed % 25 == 0:
                print(f"[INFO] image {frame_idx}/{total or '?'} — {result['total_ms']:.0f} ms — "
                      f"{result['counts']}")
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    incidents = pipeline.incidents.snapshot()
    summary = {
        "source": source_name,
        "mode": "temps reel (images ignorees)" if args.realtime else "toutes les images",
        "images_source": frame_idx,
        "images_analysees": analysed,
        "images_ignorees": skipped,
        "duree_traitement_s": round(elapsed, 1),
        "latence_moyenne_ms": round(sum(processing_times) / max(1, len(processing_times)), 1),
        "cadence_analyse_img_s": round(analysed / elapsed, 2) if elapsed else None,
        "capacites_modele": pipeline.detector.capabilities(),
        "harnais_exige": pipeline.harness_required,
        "etat_machine": machine_states.get(pipeline.machine_id),
        "politique_etat_inconnu": machine_states.unknown_policy,
        "incidents": [{k: v for k, v in i.items() if k != "bbox"} for i in incidents],
        "video_annotee": str(output),
        "journal_alertes": str(alerts),
    }
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {analysed} image(s) analysée(s) en {elapsed:.0f} s "
          f"({summary['latence_moyenne_ms']:.0f} ms/image), {skipped} ignorée(s)")
    print(f"[OK] {len(incidents)} incident(s) — vidéo : {output} ({codec})")
    print(f"[OK] Rapport : {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection HSE (comportements + homme-machine) sur vidéo ou caméra.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Fichier vidéo, index de caméra (0) ou URL RTSP")
    parser.add_argument("--output", help="Vidéo annotée (défaut : data/outputs/<nom>_hse.mp4)")
    parser.add_argument("--alerts", help="Journal CSV des alertes (défaut : à côté de la vidéo)")
    parser.add_argument("--harness-required", action="store_true", help="Exiger le harnais (travail en hauteur)")
    parser.add_argument("--machine-state", choices=("marche", "arret", "inconnu"),
                        help="Déclare l'état de l'engin de cette caméra (écrit dans config/machine_states.json)")
    parser.add_argument("--unknown-machine-policy", choices=("prudent", "surveillance"))
    parser.add_argument("--realtime", action="store_true", help="Simule un flux en direct (saut d'images)")
    parser.add_argument("--show", action="store_true", help="Affiche l'analyse dans une fenêtre")
    parser.add_argument("--hide-raw", action="store_true", help="Masque les détections brutes du modèle")
    parser.add_argument("--max-frames", type=int, default=0, help="Limite le nombre d'images analysées")
    main(parser.parse_args())
