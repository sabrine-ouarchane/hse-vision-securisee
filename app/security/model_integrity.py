"""
app/security/model_integrity.py

Vérification d'intégrité des poids des modèles AVANT leur chargement.

Les fichiers .pt (PyTorch / Ultralytics) sont désérialisés avec pickle :
charger un fichier modifié peut exécuter du code arbitraire. On compare donc
leur empreinte SHA-256 à celle enregistrée dans config/model_integrity.json,
et on refuse de charger tout fichier inconnu ou modifié.

Après un réentraînement (ou l'ajout d'un modèle de provenance vérifiée),
enregistrer les nouvelles empreintes :
    python -m app.security.model_integrity --update
Vérifier sans charger :
    python -m app.security.model_integrity
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.config import CONFIG_DIR, ROOT, load_detection_rules, resolve

INTEGRITY_PATH = CONFIG_DIR / "model_integrity.json"


class ModelIntegrityError(RuntimeError):
    """Poids absents de la liste de confiance ou modifiés : chargement refusé."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(path) -> str:
    path = resolve(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def load_expected(integrity_path=INTEGRITY_PATH) -> dict:
    integrity_path = Path(integrity_path)
    if not integrity_path.is_file():
        return {}
    return json.loads(integrity_path.read_text(encoding="utf-8")).get("models", {})


def verify_model(path, integrity_path=INTEGRITY_PATH) -> str:
    """Retourne l'empreinte si elle est conforme ; lève ModelIntegrityError sinon."""
    key = _key(path)
    expected = load_expected(integrity_path).get(key)
    if expected is None:
        raise ModelIntegrityError(
            f"Modèle {key} : aucune empreinte SHA-256 de confiance dans {Path(integrity_path).name}. "
            f"Chargement refusé (fichier pickle). Après vérification de sa provenance, l'enregistrer avec : "
            f"python -m app.security.model_integrity --update"
        )
    actual = sha256_file(resolve(path))
    if actual != expected["sha256"]:
        raise ModelIntegrityError(
            f"Modèle {key} MODIFIÉ : empreinte SHA-256 {actual[:16]}… au lieu de {expected['sha256'][:16]}… "
            f"(enregistrée le {expected.get('recorded_at', '?')}). Chargement refusé : le fichier a pu être "
            f"altéré. Restaurer la version d'origine, ou, si le changement est voulu (réentraînement), "
            f"mettre à jour l'empreinte avec python -m app.security.model_integrity --update"
        )
    return actual


def update(paths, integrity_path=INTEGRITY_PATH) -> dict:
    """Enregistre l'empreinte actuelle des fichiers donnés (à n'utiliser que sur des poids de confiance)."""
    integrity_path = Path(integrity_path)
    data = json.loads(integrity_path.read_text(encoding="utf-8")) if integrity_path.is_file() else {}
    data.setdefault("_comment", "Empreintes SHA-256 des poids autorisés, vérifiées avant chaque chargement "
                                "(app/security/model_integrity.py). Mettre à jour uniquement pour des poids de "
                                "provenance vérifiée : python -m app.security.model_integrity --update")
    models = data.setdefault("models", {})
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    for path in paths:
        models[_key(path)] = {"sha256": sha256_file(resolve(path)), "size": resolve(path).stat().st_size,
                              "recorded_at": now}
    integrity_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return models


def configured_models() -> list[str]:
    return [path for path in load_detection_rules()["models"].values() if path]


def main():
    parser = argparse.ArgumentParser(description="Intégrité des poids des modèles (SHA-256).")
    parser.add_argument("--update", action="store_true", help="Enregistrer les empreintes actuelles comme référence.")
    parser.add_argument("paths", nargs="*", help="Fichiers (par défaut : modèles de config/detection_rules.json).")
    args = parser.parse_args()
    paths = args.paths or configured_models()
    if args.update:
        for key, entry in update(paths).items():
            print(f"[OK] {key} : {entry['sha256']}")
        return
    failed = False
    for path in paths:
        try:
            print(f"[OK] {_key(path)} : {verify_model(path)}")
        except (ModelIntegrityError, OSError) as error:
            failed = True
            print(f"[REFUS] {error}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
