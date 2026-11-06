"""
tools/robustness/run_baseline.py — Partie 1 : référence sur le jeu propre.

- Contrôle du jeu : quasi-doublons test ↔ train/valid, instances par classe.
- Précision, rappel, mAP50 par classe sur l'échantillon fixe du test, en 640
  (résolution d'entraînement) et 960 (résolution de production), sur le test
  complet et sur le sous-ensemble indépendant (sans quasi-doublon).
- Décision du pipeline par personne annotée.
- Vérification de l'évaluateur maison contre `model.val` d'ultralytics
  (split test complet, 640).

Usage :
    python tools/robustness/run_baseline.py [--skip-val-check]
Sorties : data/outputs/robustness/baseline_*.csv
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def evaluate_clean(model, items, imgsz, subsets):
    evaluator, decisions = C.Evaluator(), C.DecisionCounter()
    for item in items:
        image = C.read_image(item["path"])
        true_cls, true_xyxy = C.labels_xyxy(item["labels"], image.shape[1], image.shape[0])
        pred = C.predict(model, image, imgsz)
        evaluator.add(item["name"], *pred, true_cls, true_xyxy)
        decisions.add(item["name"], C.person_decisions(true_cls, true_xyxy, *pred))
    return evaluator, decisions


def ultralytics_check(model, items, evaluator_all):
    """Écart entre l'évaluateur maison et model.val sur le split test complet (640)."""
    yaml_path = C.write_local_yaml()
    temp = C.OUT / "_val_tmp"
    try:
        metrics = model.val(data=str(yaml_path), split="test", imgsz=640, batch=8, plots=False, save_json=False,
                            project=str(temp), name="val", verbose=False, device="cpu")
        theirs = {C.NAMES[int(k)]: float(v) for k, v in zip(metrics.box.ap_class_index, metrics.box.ap50)}
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    ours = evaluator_all.metrics()
    rows = []
    for name in C.NAMES:
        if name in theirs:
            rows.append({"classe": name, "map50_ultralytics": round(theirs[name], 4),
                         "map50_evaluateur": round(ours[name]["map50"], 4),
                         "ecart": round(ours[name]["map50"] - theirs[name], 4)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-val-check", action="store_true", help="Ne pas comparer à model.val.")
    args = parser.parse_args()
    C.seed_everything()
    timer = C.Timer()

    C.write_local_yaml()
    items = C.load_items("test")
    duplicates = C.near_duplicates(items)
    sample = C.fixed_sample(items)
    in_sample = [i for i in items if i["name"] in set(sample["echantillon"])]
    subsets = C.subset_names(sample["echantillon"], duplicates)

    # Composition du jeu
    composition = []
    for subset, names in {"test_204": {i["name"] for i in items}, **subsets}.items():
        chosen = [i for i in items if i["name"] in names]
        for k, name in enumerate(C.NAMES):
            classes = [i["labels"][:, 0].astype(int) for i in chosen]
            composition.append({"sous_ensemble": subset, "images": len(chosen), "classe": name,
                                "instances": int(sum((c == k).sum() for c in classes)),
                                "images_avec_classe": int(sum((c == k).any() for c in classes))})
    C.write_csv(C.OUT / "baseline_composition.csv", composition)
    print(f"Test : {len(items)} images, {sum(duplicates.values())} quasi-doublons de train/valid ; "
          f"échantillon {len(in_sample)} images dont {len(subsets['test_independant'])} indépendantes.")

    model = C.load_model()
    metric_rows, decision_rows = [], []
    for imgsz in (640, 960):
        evaluator, decisions = evaluate_clean(model, in_sample, imgsz, subsets)
        metric_rows += C.metric_rows("propre", evaluator, subsets, imgsz=imgsz)
        decision_rows += C.decision_rows("propre", decisions, subsets, imgsz=imgsz)
        print(f"  {imgsz} px évalué ({timer})")
    C.write_csv(C.OUT / "baseline_metrics.csv", metric_rows)
    C.write_csv(C.OUT / "baseline_decisions.csv", decision_rows)

    if not args.skip_val_check:
        evaluator_all, _ = evaluate_clean(model, items, 640, None)
        check = ultralytics_check(model, items, evaluator_all)
        C.write_csv(C.OUT / "baseline_verification_val.csv", check)
        print("Vérification contre model.val (test complet, 640) : écart mAP50 max "
              f"{max(abs(r['ecart']) for r in check):.3f}")

    print(f"Terminé en {timer}. Résultats : {C.OUT}")


if __name__ == "__main__":
    main()
