"""
tools/verify_audit.py

Vérifie le journal d'audit chaîné (data/outputs/audit_log.jsonl) : toute
ligne modifiée, supprimée, insérée ou réécrite sans la clé est signalée.

Usage :
    python tools/verify_audit.py
    python tools/verify_audit.py --log chemin/audit_log.jsonl --key config/secrets/audit.key
Code de sortie : 0 si le journal est intact, 1 sinon.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.security.audit_log import AUDIT_PATH, KEY_PATH, verify  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Vérification du journal d'audit chaîné et signé.")
    parser.add_argument("--log", default=str(AUDIT_PATH))
    parser.add_argument("--key", default=str(KEY_PATH))
    args = parser.parse_args()

    if not Path(args.log).is_file():
        print(f"[INFO] Aucun journal d'audit : {args.log}")
        return 0
    result = verify(args.log, args.key)
    for warning in result["warnings"]:
        print(f"[ATTENTION] {warning}")
    if result["ok"]:
        print(f"[OK] Journal intact : {result['count']} entrée(s), chaînage SHA-256 et signatures HMAC valides.")
        return 0
    print(f"[ALTÉRÉ] {len(result['errors'])} anomalie(s) sur {result['count']} entrée(s) :")
    for error in result["errors"][:30]:
        print(f"  - {error}")
    if len(result["errors"]) > 30:
        print(f"  … et {len(result['errors']) - 30} autre(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
