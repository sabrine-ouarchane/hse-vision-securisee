"""
app/security/tamper_detector.py

Détection des EFFETS VISUELS d'un sabotage de caméra, image par image, avant
l'inférence YOLO. OpenCV seul, quelques millisecondes par image sur CPU.

Cinq types :
- obstruction    : objectif couvert (image sombre / uniforme, ou grande
                   partie de la scène qui perd toute texture) ;
- deplacement    : caméra tournée ou décalée (ORB + homographie RANSAC par
                   rapport à une image de référence) — caméras fixes seulement ;
- image_degradee : flou (variance du Laplacien) ou éblouissement (pixels
                   saturés), relatifs à la scène de référence ;
- flux_fige      : image identique d'une analyse à l'autre ;
- rejeu          : images déjà vues réapparaissant avec un décalage constant
                   (hash perceptuel + historique récent).

Chaque indice doit être maintenu pendant confirm_s (2–3 s) pour être
confirmé : un ouvrier qui passe devant l'objectif ne déclenche rien.

Toutes les comparaisons se font sur une image de travail en niveaux de gris
de taille fixe (work_side_px), quelle que soit la résolution reçue (source
d'origine, copie réduite à 1280 px, image de référence déclarée).

Ce module ne fait AUCUNE prévention réseau : il constate seulement les
effets d'une attaque sur l'image reçue.
"""

import time
from collections import deque

import cv2
import numpy as np

from app.config import resolve

TAMPER_TYPES = ("obstruction", "deplacement", "image_degradee", "flux_fige", "rejeu")
TAMPER_LABELS = {
    "obstruction": "Obstruction de l'objectif",
    "deplacement": "Déplacement de la caméra",
    "image_degradee": "Image dégradée",
    "flux_fige": "Flux figé",
    "rejeu": "Rejeu suspecté",
}

DEFAULTS = {
    "enabled": True,
    "work_side_px": 480,
    "min_interval_s": 0.3,
    "reference_s": 2.0,
    "confirm_s": 2.5,
    "gap_tolerance_s": 1.0,
    "release_s": 2.0,
    "baseline_tau_s": 20.0,
    "grid": 6,
    "dark_mean": 20.0,
    "uniform_std": 8.0,
    "cell_uniform_std": 6.0,
    "cell_texture_min": 20.0,
    "cell_lost_ratio": 0.12,
    "obstruction_cells_ratio": 0.45,
    "blur_ratio": 0.05,
    "blur_min_reference": 30.0,
    "saturation_level": 250,
    "glare_ratio": 0.25,
    "freeze_diff": 0.6,
    "displacement_interval_s": 1.0,
    "orb_features": 600,
    "min_inliers": 25,
    "displacement_shift_pct": 10.0,
    "displacement_rotation_deg": 6.5,
    "replay_history_s": 60.0,
    "replay_min_lag_s": 2.0,
    "replay_match_bits": 8,
    "replay_unique_bits": 4,
    "replay_lag_tolerance_s": 0.75,
    "replay_min_matches": 3,
    "replay_window_s": 4.0,
}


def apply_prudence(persons: list[dict], camera: dict) -> int:
    """
    Règle de prudence (logique S-HSE-22) : tant qu'un sabotage est confirmé,
    l'image ne prouve plus la conformité. Les personnes « conforme » passent
    en « indéterminé » ; les violations déjà prouvées et les « à vérifier »
    sont conservées. Retourne le nombre de personnes déclassées.
    """
    if not camera.get("compromised"):
        return 0
    reason = "surveillance compromise (" + ", ".join(e["label"].lower() for e in camera["active"]) + ")"
    changed = 0
    for person in persons:
        if person["status"] == "conforme":
            person["status"] = "indetermine"
            person["reasons"] = [reason] + list(person.get("reasons", []))
            changed += 1
    return changed


def dhash(gray) -> int:
    """Hash perceptuel 64 bits (différence horizontale sur une vignette 9x8)."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class TamperDetector:
    def __init__(self, cfg: dict | None = None, profile: dict | None = None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        profile = profile or {}
        self.enabled = bool(self.cfg["enabled"])
        # Une caméra tenue à la main ou panoramique bouge par nature : le
        # contrôle de déplacement n'a de sens que pour une caméra fixe.
        self.camera_fixed = bool(profile.get("camera_fixe", True))
        self.reference_path = profile.get("tamper_reference")
        self._orb = cv2.ORB_create(self.cfg["orb_features"])
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self._started = None
        self._reference = None        # image de travail de référence (déplacement)
        self._reference_source = None  # "profil" ou "session"
        self._reference_features = None
        self._candidates = []          # images des premières secondes (choix de la plus nette)
        self._baseline = None          # statistiques de texture de la scène saine
        self._last_t = None
        self._last_result = None
        self._previous = None
        self._last_orb_t = None
        self._displacement = (None, "")
        self._types = {name: {"onset": None, "last_true": None, "since": None, "detail": ""} for name in TAMPER_TYPES}
        self.events = []               # sabotages terminés : type, début, fin, durée
        self.reset_replay()

    def reset_replay(self):
        """À appeler à chaque tour d'une vidéo lue en boucle (sinon rejeu garanti)."""
        self._hashes = deque()
        self._replay_run = []
        self._replay_samples = []
        if self._types["rejeu"]["since"] is None:
            self._types["rejeu"]["onset"] = None

    # ------------------------------------------------------------------
    def _work_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height, width = gray.shape
        scale = self.cfg["work_side_px"] / max(height, width)
        if scale < 1:
            gray = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))),
                              interpolation=cv2.INTER_AREA)
        return gray

    def _load_declared_reference(self, shape):
        """Image de référence déclarée dans config/camera_profiles.json (tamper_reference)."""
        path = resolve(self.reference_path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        if image is None:
            print(f"[TamperDetector] Référence introuvable ou illisible : {path} — référence prise en début de session.")
            return None
        height, width = shape
        if abs(image.shape[1] / image.shape[0] - width / height) > 0.02:
            print(f"[TamperDetector] Référence {path.name} : proportions différentes du flux — ignorée.")
            return None
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def _texture(self, gray):
        """Statistiques globales et par case de la grille."""
        grid = self.cfg["grid"]
        values = gray.astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        mean = cv2.resize(values, (grid, grid), interpolation=cv2.INTER_AREA)
        square = cv2.resize(values * values, (grid, grid), interpolation=cv2.INTER_AREA)
        cell_lap = cv2.resize(laplacian * laplacian, (grid, grid), interpolation=cv2.INTER_AREA)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "lap": float(laplacian.var()),
            "saturated": float((gray >= self.cfg["saturation_level"]).mean()),
            "cell_mean": mean,
            "cell_std": np.sqrt(np.maximum(square - mean * mean, 0)),
            "cell_lap": cell_lap,
        }

    def _features(self, gray):
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    # ------------------------------------------------------------------
    def _check_obstruction(self, texture):
        cfg = self.cfg
        if texture["mean"] < cfg["dark_mean"]:
            return True, f"image sombre (luminosité {texture['mean']:.0f}/255)"
        if texture["std"] < cfg["uniform_std"]:
            return True, f"image uniforme (écart-type {texture['std']:.1f})"
        if self._baseline is None:
            return None, ""
        textured = self._baseline["cell_lap"] >= cfg["cell_texture_min"]
        if textured.sum() < 4:
            return None, ""
        lost = (texture["cell_std"] < cfg["cell_uniform_std"]) | (
            texture["cell_lap"] < cfg["cell_lost_ratio"] * self._baseline["cell_lap"])
        lost &= texture["cell_mean"] < cfg["saturation_level"] - 10  # zone saturée = éblouissement, pas masque
        intact = texture["cell_lap"] >= 0.5 * self._baseline["cell_lap"]
        lost_ratio = float((lost & textured).sum() / textured.sum())
        intact_ratio = float((intact & textured).sum() / textured.sum())
        # Une partie de la scène reste nette : masque localisé (sinon c'est un flou global)
        if lost_ratio >= cfg["obstruction_cells_ratio"] and intact_ratio >= 0.15:
            return True, f"{lost_ratio:.0%} de la scène masquée"
        return False, ""

    def _check_degradation(self, texture):
        cfg = self.cfg
        if self._baseline is None:
            return None, ""
        glare = texture["saturated"] - self._baseline["saturated"]
        if glare >= cfg["glare_ratio"]:
            return True, f"éblouissement ({texture['saturated']:.0%} de pixels saturés)"
        if self._baseline["lap"] >= cfg["blur_min_reference"]:
            ratio = texture["lap"] / self._baseline["lap"]
            if ratio < cfg["blur_ratio"]:
                return True, f"flou (netteté {ratio:.0%} de la référence)"
        return False, ""

    def _check_frozen(self, gray):
        if self._previous is None or self._previous.shape != gray.shape:
            return None, ""
        diff = float(cv2.absdiff(gray, self._previous).mean())
        if diff < self.cfg["freeze_diff"]:
            return True, f"image inchangée (écart moyen {diff:.2f})"
        return False, ""

    def _check_displacement(self, gray, now):
        if not self.camera_fixed or self._reference_features is None:
            return None, ""
        if self._last_orb_t is not None and now - self._last_orb_t < self.cfg["displacement_interval_s"]:
            return self._displacement
        self._last_orb_t = now
        ref_keypoints, ref_descriptors = self._reference_features
        keypoints, descriptors = self._features(gray)
        result = (None, "")
        if descriptors is not None and len(keypoints) >= self.cfg["min_inliers"]:
            pairs = self._matcher.knnMatch(ref_descriptors, descriptors, k=2)
            good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
            inliers = 0
            if len(good) >= self.cfg["min_inliers"]:
                src = np.float32([ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                inliers = int(mask.sum()) if mask is not None else 0
                if homography is not None and inliers >= self.cfg["min_inliers"]:
                    height, width = gray.shape
                    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
                    moved = cv2.perspectiveTransform(corners, homography)
                    shift = float(np.linalg.norm(moved - corners, axis=2).mean() / np.hypot(width, height) * 100)
                    rotation = float(np.degrees(np.arctan2(homography[1, 0], homography[0, 0])))
                    detail = f"décalage {shift:.1f} % · rotation {rotation:+.1f}°"
                    moved_enough = (shift >= self.cfg["displacement_shift_pct"]
                                    or abs(rotation) >= self.cfg["displacement_rotation_deg"])
                    result = (moved_enough, detail)
            if result[0] is None and len(keypoints) >= 4 * self.cfg["min_inliers"]:
                # Scène texturée mais méconnaissable par rapport à la référence
                result = (True, f"scène différente de la référence ({inliers} points communs)")
        self._displacement = result
        return result

    def _check_replay(self, gray, now):
        cfg = self.cfg
        current = dhash(gray)
        old = [(t, hamming(current, h)) for t, h in self._hashes if now - t >= cfg["replay_min_lag_s"]]
        self._hashes.append((now, current))
        while self._hashes and now - self._hashes[0][0] > cfg["replay_history_s"]:
            self._hashes.popleft()
        if not old:
            return None, ""
        t_match, best = min(old, key=lambda item: (item[1], -item[0]))
        # Rejeu = retour à UN moment précis du passé : la meilleure
        # correspondance doit être nettement meilleure que toute image
        # éloignée de plus d'1 s. Une scène statique ressemble à tout son
        # historique : aucune correspondance n'y est unique.
        others = [d for t, d in old if abs(t - t_match) > 1.0]
        unique = bool(others) and min(others) >= best + cfg["replay_unique_bits"]
        if best <= cfg["replay_match_bits"] and unique:
            self._replay_run.append((now, now - t_match))
        # Fenêtre glissante : une analyse ratée (image rejouée échantillonnée à
        # une autre phase) n'interrompt pas la série.
        window = cfg["replay_window_s"]
        self._replay_samples = [t for t in self._replay_samples if now - t <= window] + [now]
        self._replay_run = [match for match in self._replay_run if now - match[0] <= window]
        if not self._replay_run:
            return False, ""
        median = float(np.median([lag for _, lag in self._replay_run]))
        coherent = [match for match in self._replay_run if abs(match[1] - median) <= cfg["replay_lag_tolerance_s"]]
        if len(coherent) >= cfg["replay_min_matches"] and len(coherent) >= 0.5 * len(self._replay_samples):
            # Le rejeu a commencé à la première correspondance de la série
            return True, f"images déjà vues il y a {median:.1f} s", coherent[0][0]
        return None, ""

    # ------------------------------------------------------------------
    def _decide(self, name, evidence, now):
        """Confirmation temporelle et fin de sabotage (avec sa durée)."""
        observed, detail, *onset = evidence
        state = self._types[name]
        if observed:
            if state["onset"] is None:
                state["onset"] = onset[0] if onset else now
            state["last_true"] = now
            state["detail"] = detail
            if state["since"] is None and now - state["onset"] >= self.cfg["confirm_s"]:
                state["since"] = state["onset"]
        elif observed is False:
            if state["since"] is not None:
                if now - state["last_true"] >= self.cfg["release_s"]:
                    self.events.append({
                        "type": name, "label": TAMPER_LABELS[name], "since": round(state["since"], 2),
                        "until": round(state["last_true"], 2),
                        "duration_s": round(state["last_true"] - state["since"], 1),
                    })
                    state.update(onset=None, last_true=None, since=None, detail="")
            elif state["onset"] is not None and now - state["last_true"] > self.cfg["gap_tolerance_s"]:
                state["onset"] = None

    def _update_baseline(self, texture, dt):
        if self._baseline is None:
            self._baseline = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in texture.items()}
            return
        # Suivi lent de la scène saine (éclairage, mouvement de caméra),
        # gelé dès qu'une anomalie est suspectée.
        alpha = min(1.0, dt / self.cfg["baseline_tau_s"])
        for key, value in texture.items():
            self._baseline[key] = (1 - alpha) * self._baseline[key] + alpha * value

    # ------------------------------------------------------------------
    def update(self, frame, now: float) -> dict:
        """
        Args:
            frame: image BGR reçue (n'importe quelle résolution).
            now: temps de la vidéo en secondes.
        Returns:
            dict : state (initialisation / fiable / compromise / desactive),
            compromised, active (sabotages confirmés), suspected, events, ms.
        """
        if not self.enabled:
            return {"state": "desactive", "compromised": False, "active": [], "suspected": [], "events": [], "ms": 0.0}
        # Seuils calibrés pour ~2 analyses/s : en traitement de fichier image par
        # image (60 img/s), le contrôle n'est refait qu'après min_interval_s.
        if self._last_result is not None and 0 <= now - self._last_t < self.cfg["min_interval_s"]:
            return dict(self._last_result, ms=0.0)
        started = time.perf_counter()
        gray = self._work_image(frame)
        if self._started is None:
            self._started = now
            if self.reference_path:
                self._reference = self._load_declared_reference(gray.shape)
                if self._reference is not None:
                    self._reference_source = "profil"
                    self._reference_features = self._features(self._reference)
                    self._baseline = None
                    self._update_baseline(self._texture(self._reference), 0)
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        texture = self._texture(gray)

        initialising = self._reference_source is None and now - self._started < self.cfg["reference_s"]
        # Pendant une autre anomalie (image masquée, figée, dégradée), les images
        # se ressemblent toutes : elles ne servent ni de preuve ni d'historique de rejeu.
        other_anomaly = (any(self._types[n]["onset"] is not None for n in ("obstruction", "flux_fige"))
                         or self._types["image_degradee"]["since"] is not None)
        evidence = {
            "obstruction": self._check_obstruction(texture),
            "flux_fige": self._check_frozen(gray),
            "rejeu": (None, "") if other_anomaly else self._check_replay(gray, now),
        }
        if initialising:
            # Premières secondes : on retient les images saines, la plus nette sert de référence
            if not evidence["obstruction"][0] and texture["std"] >= self.cfg["uniform_std"]:
                self._candidates.append((texture["lap"], gray, texture))
            evidence["image_degradee"] = (None, "")
            evidence["deplacement"] = (None, "")
        else:
            if self._reference_source is None and self._candidates:
                _, best, best_texture = max(self._candidates, key=lambda item: item[0])
                self._reference, self._reference_source = best, "session"
                self._reference_features = self._features(best) if self.camera_fixed else None
                for _, _, candidate in self._candidates:
                    self._update_baseline(candidate, self.cfg["reference_s"] / len(self._candidates))
                self._candidates = []
            evidence["image_degradee"] = (False, "") if evidence["obstruction"][0] else self._check_degradation(texture)
            evidence["deplacement"] = (None, "") if evidence["obstruction"][0] else self._check_displacement(gray, now)

        if evidence["obstruction"][0]:
            evidence["flux_fige"] = (None, "")  # un objectif couvert donne aussi une image constante
        for name in TAMPER_TYPES:
            self._decide(name, evidence[name], now)
        suspected = [name for name in TAMPER_TYPES if self._types[name]["onset"] is not None and self._types[name]["since"] is None]
        if self._reference_source is not None and not any(self._types[n]["onset"] is not None for n in TAMPER_TYPES):
            self._update_baseline(texture, dt)
        self._previous = gray
        self._last_t = now

        active = [
            {"type": name, "label": TAMPER_LABELS[name], "since": round(state["since"], 2),
             "duration_s": round(now - state["since"], 1), "detail": state["detail"]}
            for name, state in self._types.items() if state["since"] is not None
        ]
        if active:
            camera_state = "compromise"
        elif self._reference_source is None:
            camera_state = "initialisation"
        else:
            camera_state = "fiable"
        self._last_result = {
            "state": camera_state,
            "compromised": bool(active),
            "active": active,
            "suspected": suspected,
            "events": list(self.events),
            "reference": self._reference_source,
            "displacement_check": "actif" if self.camera_fixed else "désactivé (caméra mobile)",
            "ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return self._last_result
