"""
tools/robustness/summarize.py — Synthèse des parties 2 à 5 (tableau + graphiques).

Lit les CSV partieN_*.csv et produit :
- synthese.csv / synthese.md : par attaque et par classe, rappel au seuil de
  production (propre → attaqué, chute), et décision du pipeline pour les
  personnes annotées en violation. La ligne « violation → conforme » (bascule
  qui fait passer une personne sans casque pour conforme) est le cas le plus
  grave ; elle est mise en évidence.
- partie2_rappel_severite.png, partie3_rappel_eps.png,
  partie3_decisions_violations.png, partie5_temporel.png.

Usage : python tools/robustness/summarize.py [dossier]
"""

import csv
import sys
from pathlib import Path

CLASSES = ["Person", "NO-Hardhat", "NO-Safety Vest", "NO-Harness", "Hardhat", "Safety Vest", "Harness"]
DECISIONS = ["violation", "a_verifier", "indetermine", "conforme", "non_detectee"]


def _read(path):
    if not Path(path).is_file():
        return []
    with open(path, encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _key(row, fields):
    return tuple(row.get(f, "") for f in fields)


def _synthesis_rows(metrics, decisions, part, label_fields, reference):
    """Une ligne par condition × sous-ensemble ; `reference(row)` donne la clé de la condition propre."""
    by_condition = {}
    for row in metrics:
        key = _key(row, label_fields + ["sous_ensemble"])
        by_condition.setdefault(key, {})[row["classe"]] = row
    violations = {_key(r, label_fields + ["sous_ensemble"]): r for r in decisions if r.get("attendu") == "violation"}
    rows = []
    for key, classes in by_condition.items():
        sample = next(iter(classes.values()))
        clean_key = reference(sample)
        if clean_key is None:
            continue
        clean = by_condition.get(clean_key, {})
        out = {"partie": part, **dict(zip(label_fields + ["sous_ensemble"], key)), "images": sample["images"]}
        for name in CLASSES:
            before = _num(clean.get(name, {}).get("rappel_seuil_prod"))
            after = _num(classes.get(name, {}).get("rappel_seuil_prod"))
            out[f"{name}_rappel_propre"] = round(before, 3)
            out[f"{name}_rappel"] = round(after, 3)
            out[f"{name}_chute"] = round(before - after, 3)
        decision = violations.get(key)
        if decision:
            total = int(decision["personnes"]) or 1
            out["personnes_en_violation"] = decision["personnes"]
            for d in DECISIONS:
                out[f"violation_vers_{d}"] = f"{int(decision[d])} ({100 * int(decision[d]) / total:.0f} %)"
        rows.append(out)
    return rows


def build_synthesis(folder):
    folder = Path(folder)
    rows = []
    p2m, p2d = _read(folder / "partie2_metrics.csv"), _read(folder / "partie2_decisions.csv")
    rows += _synthesis_rows(p2m, p2d, 2, ["condition", "severite", "imgsz"],
                            lambda r: ("propre", "0", r["imgsz"], r["sous_ensemble"]))
    p3m, p3d = _read(folder / "partie3_metrics.csv"), _read(folder / "partie3_decisions.csv")
    rows += _synthesis_rows(p3m, p3d, 3, ["condition", "objectif", "eps", "evaluation"],
                            lambda r: ("aucune", "", "0", r["evaluation"], r["sous_ensemble"]))
    p4m, p4d = _read(folder / "partie4_metrics.csv"), _read(folder / "partie4_decisions.csv")
    rows += _synthesis_rows(p4m, p4d, 4, ["condition", "patch", "evaluation"],
                            lambda r: ("propre", r["patch"], r["evaluation"].split("+")[0], r["sous_ensemble"]))
    p5m, p5d = _read(folder / "partie5_metrics.csv"), _read(folder / "partie5_decisions.csv")
    rows += _synthesis_rows(p5m, p5d, 5, ["condition", "defense"],
                            lambda r: ("propre", r["defense"], r["sous_ensemble"]))
    return rows


def write_markdown(rows, path, temporal):
    lines = ["# Synthèse — robustesse de best.pt", "",
             "Rappel au seuil de production (confiance 0,35 ; personnes 0,15), propre → attaqué. "
             "Décision : personnes annotées en violation, décision obtenue sur une image isolée. "
             "**violation → conforme** est le cas le plus grave (bascule).", ""]
    header = ["partie", "condition", "détail", "sous-ensemble", "images", "Person", "NO-Hardhat", "NO-Safety Vest",
              "violation → violation", "→ à vérifier", "→ indéterminé", "**→ conforme**", "→ non détectée"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        subset = r["sous_ensemble"]
        if subset.endswith("independant") or not (subset in ("test_complet", "pgd_complet", "adaptatif")
                                                  or subset.startswith("patch_")):
            continue
        detail = " ".join(str(r.get(f, "")) for f in ("objectif", "eps", "severite", "imgsz", "patch", "evaluation",
                                                       "defense") if r.get(f, "") not in ("", None))
        cells = [str(r["partie"]), str(r["condition"]), detail, r["sous_ensemble"], str(r["images"])]
        for name in ("Person", "NO-Hardhat", "NO-Safety Vest"):
            cells.append(f"{r[f'{name}_rappel_propre']:.2f} → {r[f'{name}_rappel']:.2f}")
        for d in DECISIONS:
            value = r.get(f"violation_vers_{d}", "")
            cells.append(f"**{value}**" if d == "conforme" and value and not value.startswith("0 ") else value)
        lines.append("| " + " | ".join(cells) + " |")
    if temporal:
        lines += ["", "## Confirmation temporelle (pipeline complet, PGD 960 ε=8/255)", "",
                  "| vidéo | objectif | mode | fraction attaquée | graine | incidents casque | incidents gilet | "
                  "durée casque (s) | personnes conformes (cumul) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in temporal:
            lines.append(f"| {r['video']} | {r['objectif']} | {r['mode']} | {r['fraction']} | {r['graine']} | "
                         f"{r['incidents_casque']} | {r['incidents_gilet']} | {r['duree_casque_s']} | "
                         f"{r['statuts_conforme']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plots(folder, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    shown = ["Person", "NO-Hardhat", "NO-Safety Vest", "Hardhat", "Safety Vest", "Harness"]

    p2 = [r for r in _read(folder / "partie2_metrics.csv") if r["sous_ensemble"] == "test_complet"]
    if p2:
        kinds = [k for k in dict.fromkeys(r["condition"] for r in p2) if k != "propre"]
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
        for ax, kind in zip(axes.flat, kinds):
            for name in shown:
                points = sorted((int(r["severite"]), _num(r["rappel_seuil_prod"])) for r in p2
                                if r["classe"] == name and r["imgsz"] == "640" and r["condition"] in (kind, "propre"))
                if points:
                    ax.plot(*zip(*points), marker="o", label=name)
            ax.set_title(kind)
            ax.set_xlabel("sévérité (640 px)")
            ax.grid(alpha=0.3)
        axes.flat[0].set_ylabel("rappel au seuil de production")
        axes.flat[-1].axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "partie2_rappel_severite.png", dpi=90)
        plt.close(fig)

    p3 = _read(folder / "partie3_metrics.csv")
    if p3:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
        panels = [("fgsm", "640", "test_complet"), ("pgd", "640", "pgd_complet"),
                  ("fgsm", "960", "test_complet"), ("pgd", "960", "pgd_complet")]
        for ax, (method, evaluation, subset) in zip(axes, panels):
            for name in ["Person", "NO-Hardhat", "NO-Safety Vest"]:
                points = [(0, _num(r["rappel_seuil_prod"])) for r in p3 if r["condition"] == "aucune"
                          and r["evaluation"] == evaluation and r["sous_ensemble"] == subset and r["classe"] == name]
                points += sorted((int(r["eps"]), _num(r["rappel_seuil_prod"])) for r in p3
                                 if r["condition"] == method and r["objectif"] == "disparition"
                                 and r["evaluation"] == evaluation and r["sous_ensemble"] == subset
                                 and r["classe"] == name)
                if points:
                    ax.plot(*zip(*points), marker="o", label=name)
            ax.set_title(f"{method.upper()} disparition, évalué en {evaluation}")
            ax.set_xlabel("ε (/255)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("rappel au seuil de production")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "partie3_rappel_eps.png", dpi=90)
        plt.close(fig)

    decisions = [r for r in _read(folder / "partie3_decisions.csv") if r.get("attendu") == "violation"
                 and r["evaluation"] == "640" and r["sous_ensemble"] in ("test_complet", "pgd_complet")]
    if decisions:
        labels = [f"{r['condition']} {r['objectif']} {r['eps']}\n{r['sous_ensemble']}" for r in decisions]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
        bottom = [0.0] * len(decisions)
        colors = {"violation": "#2e7d32", "a_verifier": "#f9a825", "indetermine": "#9e9e9e", "conforme": "#c62828",
                  "non_detectee": "#5d4037"}
        for d in DECISIONS:
            values = [int(r[d]) / max(1, int(r["personnes"])) for r in decisions]
            ax.bar(range(len(decisions)), values, bottom=bottom, label=d, color=colors[d])
            bottom = [b + v for b, v in zip(bottom, values)]
        ax.set_xticks(range(len(decisions)), labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("part des personnes annotées en violation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "partie3_decisions_violations.png", dpi=90)
        plt.close(fig)

    temporal = _read(folder / "partie5_temporel.csv")
    if temporal:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, objective in zip(axes, ("disparition", "bascule")):
            for video in dict.fromkeys(r["video"] for r in temporal):
                for mode in ("aleatoire", "rafales"):
                    points = {}
                    for r in temporal:
                        if r["video"] == video and r["objectif"] == objective and r["mode"] in (mode, "", "toutes"):
                            points.setdefault(_num(r["fraction"]), []).append(_num(r["duree_casque_s"]) +
                                                                               _num(r["duree_gilet_s"]))
                    if points:
                        xs = sorted(points)
                        ax.plot(xs, [sum(points[x]) / len(points[x]) for x in xs], marker="o", label=f"{video} {mode}")
            ax.set_title(f"PGD 960 ε=8/255 — {objective}")
            ax.set_xlabel("fraction des images attaquées")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("durée d'incidents casque + gilet (s)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder / "partie5_temporel.png", dpi=90)
        plt.close(fig)
    log("Graphiques écrits.")


def summarize(folder, log=print):
    rows = build_synthesis(folder)
    if rows:
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with open(Path(folder) / "synthese.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows([{c: r.get(c, "") for c in columns} for r in rows])
    write_markdown(rows, Path(folder) / "synthese.md", _read(Path(folder) / "partie5_temporel.csv"))
    try:
        plots(folder, log)
    except Exception as error:  # noqa: BLE001 — la synthèse CSV reste utilisable
        log(f"Graphiques impossibles : {error}")
    log(f"Synthèse : {len(rows)} ligne(s) dans synthese.csv / synthese.md")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as C
    summarize(sys.argv[1] if len(sys.argv) > 1 else C.OUT)
