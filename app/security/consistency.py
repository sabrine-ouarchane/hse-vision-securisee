"""
app/security/consistency.py

Cohérence entre le journal d'audit (chaîné et signé, donc la référence) et
les fichiers que l'application écrit à côté, plus faciles à modifier :

- incidents : chaque `incident_ouvert` du journal doit se retrouver dans
  alerts_log.csv (même ligne, même type, même gravité, même message), dans
  live_incidents/<session>/incidents.json une fois la session terminée, et
  sa capture JPEG doit avoir la même empreinte SHA-256 ;
- incidents de zone (session « zones », étape 9) : même ligne dans
  alerts_log.csv, et ouverture présente dans data/outputs/zones/incidents.jsonl
  (même type, même début) ;
- comptes : l'état de config/users.json (comptes, rôle, actif, haché du mot
  de passe) doit correspondre à la relecture des événements compte_cree /
  compte_desactive / compte_reactive / mot_de_passe_change du journal.

Toute différence est une anomalie : fichier modifié, ligne ou compte
supprimé, compte ajouté hors de l'application.
"""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.security.auth import password_fingerprint


def _anomaly(category, subject, detail):
    return {"categorie": category, "objet": subject, "detail": detail}


def _read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _zone_incidents(path):
    """Ouvertures enregistrées dans zones/incidents.jsonl : {id: ligne}, ou None si le fichier est absent."""
    path = Path(path) if path else None
    if path is None or not path.is_file():
        return None
    opened = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("evenement") == "ouvert":
            opened[row.get("id")] = row
    return opened


def check_incidents(entries, alerts_path, snapshots_dir, zone_incidents_path=None) -> list[dict]:
    anomalies = []
    rows = _read_csv_rows(alerts_path)
    by_event = {}
    for row in rows:
        by_event.setdefault(row.get("event_id"), []).append(row)
    closed_sessions = {e["data"].get("session_id") for e in entries if e.get("event") == "session_arretee"}
    # Les lignes écrites avant la correction de l'event_id (suffixe unique) peuvent partager un même
    # identifiant (ex. casque et gilet d'une même personne sur la même image) : une ligne n'est
    # « modifiée » que si aucune autre entrée du journal ne la revendique.
    claimed = {}
    for entry in entries:
        data = entry.get("data", {})
        if entry.get("event") == "incident_ouvert" and data.get("csv_event_id"):
            message = f"[{data['source']}] {data['message']}" if data.get("source") else data.get("message")
            claimed.setdefault(data["csv_event_id"], set()).add(message)
    session_files = {}
    zone_records = None

    for entry in entries:
        if entry.get("event") != "incident_ouvert":
            continue
        data = entry.get("data", {})
        session_id, incident_id = data.get("session_id"), data.get("incident_id")
        if not session_id:
            continue
        subject = f"incident #{incident_id} · session {session_id} · {data.get('type')}"
        expected_message = f"[{data['source']}] {data['message']}" if data.get("source") else data.get("message")

        # 1) alerts_log.csv
        if data.get("csv_event_id"):
            candidates = by_event.get(data["csv_event_id"], [])
            same = [r for r in candidates if r.get("message") == expected_message]
            unclaimed = [r for r in candidates if r.get("message") not in claimed[data["csv_event_id"]]]
            if not same and not unclaimed:
                anomalies.append(_anomaly("incident", subject, "ligne supprimée de alerts_log.csv"))
            elif not same:
                anomalies.append(_anomaly("incident", subject, "message modifié dans alerts_log.csv"))
            elif (same[0].get("alert_type") != data.get("categorie")
                  or (data.get("severity") and same[0].get("severity") != data.get("severity"))):
                anomalies.append(_anomaly("incident", subject, "type ou gravité modifiés dans alerts_log.csv"))
        elif data.get("message") and not any(
                r.get("message") == expected_message and r.get("source") == "vision-temps-reel" for r in rows):
            # Incident journalisé avant l'ajout de csv_event_id : recherche par message
            anomalies.append(_anomaly("incident", subject, "ligne absente de alerts_log.csv"))

        # 2) incidents de zone : zones/incidents.jsonl (écrit à chaque ouverture)
        if session_id == "zones":
            if zone_records is None:
                zone_records = _zone_incidents(zone_incidents_path) or {}
                if not zone_records:
                    anomalies.append(_anomaly("incident", "zones/incidents.jsonl", "fichier supprimé ou vide"))
                    zone_records["__signale__"] = True
            stored = zone_records.get(incident_id)
            if stored is None and "__signale__" not in zone_records:
                anomalies.append(_anomaly("incident", subject, "incident supprimé de zones/incidents.jsonl"))
            elif stored is not None and (stored.get("kind") != data.get("type")
                                         or stored.get("debut") != data.get("wall_time")):
                anomalies.append(_anomaly("incident", subject, "incident modifié dans zones/incidents.jsonl"))
            continue

        # 2) incidents.json (écrit à la fin de la session)
        if session_id in closed_sessions:
            if session_id not in session_files:
                path = Path(snapshots_dir) / session_id / "incidents.json"
                try:
                    session_files[session_id] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
                except (OSError, json.JSONDecodeError):
                    session_files[session_id] = "illisible"
                if session_files[session_id] is None:
                    anomalies.append(_anomaly("incident", f"session {session_id}", "incidents.json supprimé"))
                elif session_files[session_id] == "illisible":
                    anomalies.append(_anomaly("incident", f"session {session_id}", "incidents.json illisible"))
            stored = session_files[session_id]
            if isinstance(stored, list):
                match = next((i for i in stored if i.get("id") == incident_id), None)
                if match is None:
                    anomalies.append(_anomaly("incident", subject, "incident supprimé de incidents.json"))
                elif match.get("kind") != data.get("type") or match.get("wall_time") != data.get("wall_time"):
                    anomalies.append(_anomaly("incident", subject, "incident modifié dans incidents.json"))

        # 3) capture JPEG
        if data.get("snapshot") and data.get("snapshot_sha256"):
            path = Path(snapshots_dir) / session_id / data["snapshot"]
            if not path.is_file():
                anomalies.append(_anomaly("incident", subject, f"capture {data['snapshot']} supprimée"))
            elif hashlib.sha256(path.read_bytes()).hexdigest() != data["snapshot_sha256"]:
                anomalies.append(_anomaly("incident", subject, f"capture {data['snapshot']} modifiée"))
    return anomalies


def replay_accounts(entries) -> dict:
    """État des comptes reconstitué à partir du journal d'audit."""
    accounts = {}
    for entry in entries:
        data = entry.get("data", {})
        name = data.get("compte")
        event = entry.get("event")
        if event == "compte_cree":
            accounts[name] = {"role": data.get("role_compte"), "active": bool(data.get("actif", True)),
                              "fingerprint": data.get("empreinte_mot_de_passe")}
        elif name in accounts:
            if event == "mot_de_passe_change":
                accounts[name]["fingerprint"] = data.get("empreinte_mot_de_passe")
            elif event == "compte_desactive":
                accounts[name]["active"] = False
            elif event == "compte_reactive":
                accounts[name]["active"] = True
    return accounts


def check_accounts(entries, users: dict) -> list[dict]:
    anomalies = []
    expected = replay_accounts(entries)
    for name, user in users.items():
        subject = f"compte {name}"
        if name not in expected:
            anomalies.append(_anomaly("compte", subject, "présent dans users.json sans création journalisée"))
            continue
        reference = expected[name]
        if user.get("role") != reference["role"]:
            anomalies.append(_anomaly("compte", subject,
                                      f"rôle « {user.get('role')} » au lieu de « {reference['role']} » (journal)"))
        if bool(user.get("active")) != reference["active"]:
            state = "actif" if user.get("active") else "désactivé"
            anomalies.append(_anomaly("compte", subject, f"{state} dans users.json, contrairement au journal"))
        if reference["fingerprint"] and password_fingerprint(user.get("password", "")) != reference["fingerprint"]:
            anomalies.append(_anomaly("compte", subject, "mot de passe modifié hors de l'application"))
    for name in expected:
        if name not in users:
            anomalies.append(_anomaly("compte", f"compte {name}", "supprimé de users.json"))
    return anomalies


def run_checks(entries, alerts_path, snapshots_dir, users: dict, zone_incidents_path=None) -> dict:
    anomalies = (check_incidents(entries, alerts_path, snapshots_dir, zone_incidents_path)
                 + check_accounts(entries, users))
    return {
        "ok": not anomalies,
        "count": len(anomalies),
        "anomalies": anomalies,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entries": len(entries),
    }
