"""
tools/manage_users.py

Gestion des comptes du dashboard en console (config/users.json).
Aucun compte n'existe par défaut : le premier administrateur se crée ici.
Les mots de passe sont saisis de façon masquée, jamais en argument.
Chaque opération est consignée dans le journal d'audit (auteur
« console:<utilisateur Windows> »).

Usage :
    python tools/manage_users.py create <identifiant> --role admin|responsable_hse|operateur
    python tools/manage_users.py passwd <identifiant>
    python tools/manage_users.py disable <identifiant>
    python tools/manage_users.py enable <identifiant>
    python tools/manage_users.py unlock <identifiant>
    python tools/manage_users.py list
Les comptes ne sont jamais supprimés : ils sont désactivés.
"""

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.security.audit_log import AuditLog  # noqa: E402
from app.security.auth import MIN_PASSWORD_LENGTH, ROLES, AuthError, AuthService  # noqa: E402


def ask_password() -> str:
    first = getpass.getpass(f"Mot de passe ({MIN_PASSWORD_LENGTH} caractères minimum) : ")
    if getpass.getpass("Confirmer le mot de passe : ") != first:
        raise AuthError("Les deux saisies sont différentes.")
    return first


def main(argv=None, auth=None, password_prompt=ask_password):
    parser = argparse.ArgumentParser(description="Comptes du dashboard HSE.")
    parser.add_argument("action", choices=("create", "passwd", "disable", "enable", "unlock", "list"))
    parser.add_argument("username", nargs="?")
    parser.add_argument("--role", choices=ROLES)
    args = parser.parse_args(argv)

    auth = auth or AuthService(audit=AuditLog())
    actor = {"username": f"console:{getpass.getuser()}", "role": "console"}
    try:
        if args.action == "list":
            for user in auth.list_users():
                state = "actif" if user["active"] else "désactivé"
                print(f"{user['username']:24s} {user['role']:16s} {state}{' · bloqué' if user['locked'] else ''}"
                      f" · dernière connexion {user['last_login'] or '—'}")
            return 0
        if not args.username:
            raise AuthError("Identifiant manquant.")
        if args.action == "create":
            if not args.role:
                raise AuthError("Préciser --role (admin, responsable_hse ou operateur).")
            auth.create_user(args.username, password_prompt(), args.role, actor)
            print(f"[OK] Compte « {args.username} » créé ({args.role}).")
        elif args.action == "passwd":
            auth.set_password(args.username, password_prompt(), actor)
            print(f"[OK] Mot de passe de « {args.username} » changé ; ses sessions ouvertes sont fermées.")
        elif args.action in ("disable", "enable"):
            # En console, l'auteur n'est pas un compte du dashboard : seule la règle du dernier admin s'applique.
            auth.set_active(args.username, args.action == "enable", actor)
            print(f"[OK] Compte « {args.username} » {'réactivé' if args.action == 'enable' else 'désactivé'}.")
        else:
            auth.unlock(args.username, actor)
            print(f"[OK] Compte « {args.username} » débloqué.")
    except AuthError as error:
        print(f"[REFUS] {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
