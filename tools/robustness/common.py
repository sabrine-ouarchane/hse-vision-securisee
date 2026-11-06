"""
tools/robustness/common.py

Outils communs de l'évaluation de robustesse de models/best.pt (étape 8) :

- jeu d'évaluation : split TEST du dataset data_ocp (data/datasets/data_ocp),
  avec un data.yaml local (chemins Kaggle de l'original non modifiés) ;
- repérage des quasi-doublons test ↔ train/valid (images consécutives d'une
  même vidéo, variantes Roboflow d'une même photo) : les résultats sont
  donnés sur le test complet ET sur le sous-ensemble indépendant ;
- échantillon fixe (graine) partagé par toutes les parties ;
- évaluation par classe : mAP50, précision et rappel (comme ultralytics val),
  plus le rappel au seuil de production (confiance 0,35) ;
- décision du pipeline par personne annotée : violation / à vérifier /
  indéterminé / conforme / non détectée, avec les seuils de production.

Les images perturbées restent en mémoire : seuls des CSV, des graphiques et
quelques exemples sont écrits dans data/outputs/robustness/.
"""

import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.epi.compliance_checker import EPIComplianceChecker  # noqa: E402

DATASET = ROOT / "data" / "datasets" / "data_ocp"
OUT = ROOT / "data" / "outputs" / "robustness"
MODEL_PATH = ROOT / "models" / "best.pt"
SAMPLE_PATH = OUT / "echantillon.json"
DUPLICATES_PATH = OUT / "quasi_doublons.csv"

NAMES = ["Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Harness", "NO-Harness", "Phone",
         "Machine"]
CLASS_ID = {name: i for i, name in enumerate(NAMES)}
# Classes dont la disparition est dangereuse (faux négatif sur une violation ou une personne)
VANISH_TARGETS = ["Person", "NO-Hardhat", "NO-Safety Vest", "NO-Harness"]
# Bascule : la violation devient l'EPI porté
FLIP_PAIRS = {"NO-Hardhat": "Hardhat", "NO-Safety Vest": "Safety Vest"}

SEED = 0
SAMPLE_SIZE = 200
PGD_SIZE = 100
# Classes rares : leurs images sont toujours gardées dans les échantillons
RARE_CLASSES = ("NO-Harness",)
# Quasi-doublon : dHash 256 bits à ≤ 16 bits ET écart moyen < 10 niveaux sur une vignette 32×32
DUP_MAX_BITS = 16
DUP_MAX_PIXEL_GAP = 10.0

with (ROOT / "config" / "detection_rules.json").open(encoding="utf-8") as _rules:
    RULES = json.load(_rules)
PROD_CONF = RULES["inference"]["confidence"]            # 0,35 : seuil d'affichage / d'usage
PERSON_CONF = RULES["inference"]["track_confidence"]    # 0,15 : personnes suivies
VIOLATION_THRESHOLDS = RULES["violation_confidence_thresholds"]

DECISIONS = ["violation", "a_verifier", "indetermine", "conforme", "non_detectee"]
SUBSETS = ("test_complet", "test_independant")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def __str__(self):
        seconds = time.perf_counter() - self.start
        return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


# ---------------------------------------------------------------------------
# Jeu de données
# ---------------------------------------------------------------------------
def write_local_yaml(dataset=DATASET, out=OUT) -> Path:
    """data.yaml local en chemins absolus ; l'original (chemins Kaggle ../train/images) reste intact."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "data_ocp_local.yaml"
    path.write_text(
        f"path: {Path(dataset).resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames: {json.dumps(NAMES)}\n",
        encoding="utf-8",
    )
    return path


def load_items(split="test", dataset=DATASET) -> list[dict]:
    """Images d'un split avec leurs annotations YOLO (classe, cx, cy, w, h normalisés)."""
    items = []
    for image in sorted((Path(dataset) / split / "images").iterdir()):
        label = Path(dataset) / split / "labels" / f"{image.stem}.txt"
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.is_file() else []
        items.append({
            "name": image.name,
            "path": image,
            "labels": np.array([[float(v) for v in row] for row in rows], dtype=np.float64).reshape(-1, 5),
        })
    return items


def labels_xyxy(labels, width, height):
    """(n,5) normalisé -> classes (n,) et boîtes xyxy en pixels (n,4)."""
    cls = labels[:, 0].astype(int)
    cx, cy, w, h = labels[:, 1] * width, labels[:, 2] * height, labels[:, 3] * width, labels[:, 4] * height
    return cls, np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _fingerprint(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small[:, 1:] > small[:, :-1]).flatten(), thumb


def near_duplicates(test_items, dataset=DATASET, cache=DUPLICATES_PATH) -> dict:
    """Pour chaque image du test : distance au plus proche voisin de train+valid et drapeau quasi-doublon."""
    if cache.is_file():
        with cache.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if {r["image"] for r in rows} == {i["name"] for i in test_items}:
            return {r["image"]: r["quasi_doublon"] == "1" for r in rows}
    reference = [(p, *_fingerprint(p)) for split in ("train", "valid")
                 for p in sorted((Path(dataset) / split / "images").iterdir())]
    hashes = np.array([r[1] for r in reference])
    thumbs = np.array([r[2] for r in reference])
    flags, rows = {}, []
    for item in test_items:
        bits, thumb = _fingerprint(item["path"])
        distances = (hashes != bits).sum(1)
        nearest = int(distances.argmin())
        gap = float(np.abs(thumbs[nearest] - thumb).mean())
        flag = distances[nearest] <= DUP_MAX_BITS and gap < DUP_MAX_PIXEL_GAP
        flags[item["name"]] = bool(flag)
        rows.append({"image": item["name"], "plus_proche": f"{reference[nearest][0].parent.parent.name}/"
                     f"{reference[nearest][0].name}", "distance_bits": int(distances[nearest]),
                     "ecart_pixels": round(gap, 2), "quasi_doublon": int(flag)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return flags


def choose_sample(items, size, seed=SEED, rare=RARE_CLASSES) -> list[str]:
    """Échantillon déterministe ; toutes les images contenant une classe rare sont gardées."""
    rare_ids = {CLASS_ID[c] for c in rare}
    forced = [i["name"] for i in items if rare_ids & set(i["labels"][:, 0].astype(int).tolist())]
    others = sorted(i["name"] for i in items if i["name"] not in forced)
    rng = random.Random(seed)
    chosen = forced + rng.sample(others, max(0, min(size - len(forced), len(others))))
    return sorted(chosen)


def fixed_sample(items, seed=SEED) -> dict:
    """Échantillon commun à toutes les parties, enregistré dans echantillon.json (relu ensuite)."""
    names = {i["name"] for i in items}
    if SAMPLE_PATH.is_file():
        saved = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        if set(saved["echantillon"]) <= names and saved["graine"] == seed:
            return saved
    sample = choose_sample(items, SAMPLE_SIZE, seed)
    in_sample = [i for i in items if i["name"] in set(sample)]
    pgd = choose_sample(in_sample, PGD_SIZE, seed + 1)
    saved = {"graine": seed, "split": "test", "echantillon": sample, "pgd": pgd}
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
    return saved


def letterbox_square(image, size=640, pad_value=114):
    """Image carrée size×size (comme ultralytics : redimensionnement + bandes grises centrées)."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (new_w, new_h) != (width, height) else image
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    out[top:top + new_h, left:left + new_w] = resized
    return out, ratio, (left, top)


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
def load_model(path=MODEL_PATH):
    """best.pt, après contrôle de son empreinte (comme le pipeline de production)."""
    from ultralytics import YOLO

    from app.security.model_integrity import verify_model

    verify_model(path)
    return YOLO(str(path))


def predict(model, image_bgr, imgsz):
    """Prédictions brutes (confiance ≥ 0,001, comme ultralytics val) : boîtes xyxy, confiances, classes."""
    result = model.predict(image_bgr, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
    boxes = result.boxes
    return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------
IOUV = np.linspace(0.5, 0.95, 10)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_predictions(pred_cls, true_cls, iou, iouv=IOUV):
    """Appariement d'ultralytics (DetectionValidator.match_predictions) : (n_pred, n_seuils) booléens."""
    correct = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
    if len(pred_cls) == 0 or len(true_cls) == 0:
        return correct
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    for k, threshold in enumerate(iouv):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), k] = True
    return correct


def operating_thresholds(pred_cls):
    return np.where(pred_cls == CLASS_ID["Person"], PERSON_CONF, PROD_CONF)


class Evaluator:
    """Accumule les prédictions image par image ; métriques par classe et par sous-ensemble."""

    def __init__(self):
        self.images = []

    def add(self, name, pred_xyxy, pred_conf, pred_cls, true_cls, true_xyxy):
        iou = box_iou(true_xyxy, pred_xyxy)
        correct = match_predictions(pred_cls, true_cls, iou)
        # Rappel « production » : appariement refait avec les seules prédictions au-dessus du seuil d'usage
        keep = pred_conf >= operating_thresholds(pred_cls)
        correct_op = match_predictions(pred_cls[keep], true_cls, iou[:, keep], IOUV[:1])[:, 0]
        self.images.append({
            "name": name, "correct": correct, "conf": pred_conf, "pred_cls": pred_cls, "true_cls": true_cls,
            "op_tp": np.bincount(pred_cls[keep][correct_op], minlength=len(NAMES)),
            "op_pred": np.bincount(pred_cls[keep], minlength=len(NAMES)),
        })

    def metrics(self, names=None) -> dict:
        """{classe: {...}} pour les images dont le nom est dans `names` (toutes si None)."""
        from ultralytics.utils.metrics import ap_per_class

        selected = [i for i in self.images if names is None or i["name"] in names]
        result = {}
        if not selected:
            return result
        correct = np.concatenate([i["correct"] for i in selected])
        conf = np.concatenate([i["conf"] for i in selected])
        pred_cls = np.concatenate([i["pred_cls"] for i in selected])
        true_cls = np.concatenate([i["true_cls"] for i in selected])
        op_tp = sum(i["op_tp"] for i in selected)
        op_pred = sum(i["op_pred"] for i in selected)
        instances = np.bincount(true_cls, minlength=len(NAMES))
        images_with = np.zeros(len(NAMES), dtype=int)
        for i in selected:
            images_with[np.unique(i["true_cls"])] += 1
        per_class = {}
        if len(correct) and len(true_cls):
            _, _, p, r, _, ap, classes, *_ = ap_per_class(correct, conf, pred_cls, true_cls)
            per_class = {int(c): (p[k], r[k], ap[k, 0], ap[k].mean()) for k, c in enumerate(classes)}
        for k, name in enumerate(NAMES):
            p, r, ap50, ap = per_class.get(k, (0.0, 0.0, 0.0, 0.0))
            result[name] = {
                "images": int(len(selected)), "images_avec_classe": int(images_with[k]),
                "instances": int(instances[k]),
                "precision": float(p), "rappel": float(r), "map50": float(ap50), "map50_95": float(ap),
                "rappel_seuil_prod": float(op_tp[k] / instances[k]) if instances[k] else float("nan"),
                "precision_seuil_prod": float(op_tp[k] / op_pred[k]) if op_pred[k] else float("nan"),
            }
        return result


# ---------------------------------------------------------------------------
# Décision du pipeline (image isolée, seuils de production)
# ---------------------------------------------------------------------------
_CHECKER = EPIComplianceChecker(harness_required=False, confidence_thresholds=VIOLATION_THRESHOLDS)


def _detections(xyxy, conf, cls, operational=True):
    detections = []
    for box, c, k in zip(xyxy, conf, cls):
        if operational and c < (PERSON_CONF if k == CLASS_ID["Person"] else PROD_CONF):
            continue
        detections.append({"class_name": NAMES[int(k)], "confidence": float(c), "bbox": tuple(float(v) for v in box),
                           "track_id": None})
    return detections


def _label(status):
    if status["status"] == "violation":
        return "violation"
    if status["status"] == "compliant":
        return "conforme"
    return "a_verifier" if status["uncertain_violations"] else "indetermine"


def person_decisions(true_cls, true_xyxy, pred_xyxy, pred_conf, pred_cls) -> list[tuple[str, str]]:
    """(décision attendue d'après les annotations, décision obtenue) pour chaque personne annotée.

    La décision est celle d'EPIComplianceChecker (une image, sans confirmation
    temporelle) : violation / à vérifier (absence sous le seuil de violation) /
    indéterminé (EPI non identifiés) / conforme (casque ET gilet détectés).
    """
    truth = _CHECKER.check(_detections(true_xyxy, np.ones(len(true_cls)), true_cls, operational=False))
    obtained = _CHECKER.check(_detections(pred_xyxy, pred_conf, pred_cls))
    if not truth:
        return []
    iou = box_iou(np.array([t["bbox"] for t in truth]),
                  np.array([o["bbox"] for o in obtained]).reshape(-1, 4))
    pairs, used = [], set()
    for t_index in np.argsort(-iou.max(1) if iou.size else np.zeros(len(truth))):
        candidates = [(iou[t_index, k], k) for k in range(iou.shape[1]) if k not in used and iou[t_index, k] >= 0.5]
        if candidates:
            _, best = max(candidates)
            used.add(best)
            pairs.append((_label(truth[t_index]), _label(obtained[best])))
        else:
            pairs.append((_label(truth[t_index]), "non_detectee"))
    return pairs


class DecisionCounter:
    def __init__(self):
        self.rows = []  # (image, attendu, obtenu)

    def add(self, name, pairs):
        self.rows.extend((name, expected, got) for expected, got in pairs)

    def matrix(self, names=None) -> dict:
        counts = {e: {g: 0 for g in DECISIONS} for e in DECISIONS[:-1]}
        for name, expected, got in self.rows:
            if names is None or name in names:
                counts[expected][got] += 1
        return counts


# ---------------------------------------------------------------------------
# Sorties
# ---------------------------------------------------------------------------
def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset_names(sample_names, duplicates):
    """{"test_complet": noms, "test_independant": noms sans quasi-doublon}."""
    return {
        "test_complet": set(sample_names),
        "test_independant": {n for n in sample_names if not duplicates.get(n, False)},
    }


def metric_rows(condition, evaluator, subsets, **extra):
    rows = []
    for subset, names in subsets.items():
        for name, m in evaluator.metrics(names).items():
            rows.append({"condition": condition, **extra, "sous_ensemble": subset, "classe": name,
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
    return rows


def decision_rows(condition, counter, subsets, **extra):
    rows = []
    for subset, names in subsets.items():
        matrix = counter.matrix(names)
        n_images = len({r[0] for r in counter.rows if r[0] in names})
        for expected, row in matrix.items():
            rows.append({"condition": condition, **extra, "sous_ensemble": subset, "images": n_images,
                         "attendu": expected, "personnes": sum(row.values()), **row})
    return rows


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Image illisible : {path}")
    return image


def clean_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
