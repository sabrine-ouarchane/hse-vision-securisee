"""
tools/robustness/perturbations.py — Partie 2 : perturbations naturelles.

Sept perturbations, sévérité 1 (légère) à 5 (forte), appliquées à l'image
d'origine (BGR uint8). La sévérité 0 renvoie l'image inchangée. Les
paramètres sont dans SEVERITIES ; le flou et la poussière sont exprimés à
l'échelle d'une image de 640 px (puis adaptés à la taille réelle) pour que
la même sévérité ait le même effet vu par le modèle.
"""

import cv2
import numpy as np

SEVERITIES = {
    "bruit_gaussien": [4, 8, 16, 24, 32],          # écart-type, niveaux de gris (0-255)
    "flou": [0.75, 1.5, 2.5, 3.5, 5.0],             # σ du flou gaussien, px à l'échelle 640
    "luminosite": [0.75, 0.6, 0.45, 0.32, 0.2],     # facteur d'assombrissement
    "contraste": [0.75, 0.55, 0.4, 0.28, 0.18],     # facteur de contraste autour de la moyenne
    "jpeg": [60, 40, 25, 15, 8],                    # qualité JPEG
    "poussiere": [0.1, 0.2, 0.3, 0.45, 0.6],        # opacité du voile (+ grains)
    "occlusion": [0.1, 0.2, 0.3, 0.4, 0.5],         # part de la surface de chaque objet masquée
}
DUST_COLOR = np.array([165, 185, 200], dtype=np.float32)  # beige (BGR)


def _scale(image):
    return max(image.shape[:2]) / 640.0


def apply(image, kind, severity, rng=None, boxes=None):
    """Image perturbée (uint8, même taille). `boxes` (xyxy pixels) sert à l'occlusion."""
    if kind not in SEVERITIES:
        raise ValueError(f"Perturbation inconnue : {kind}")
    if severity == 0:
        return image.copy()
    if not 1 <= severity <= 5:
        raise ValueError("La sévérité va de 0 à 5.")
    rng = rng or np.random.default_rng(0)
    level = SEVERITIES[kind][severity - 1]
    work = image.astype(np.float32)

    if kind == "bruit_gaussien":
        work += rng.normal(0.0, level, image.shape)
    elif kind == "flou":
        return cv2.GaussianBlur(image, (0, 0), level * _scale(image))
    elif kind == "luminosite":
        work *= level
    elif kind == "contraste":
        mean = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
        work = mean + (work - mean) * level
    elif kind == "jpeg":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif kind == "poussiere":
        work = work * (1 - level) + DUST_COLOR * level
        height, width = image.shape[:2]
        count = int(level * 0.01 * height * width / max(_scale(image), 1.0) ** 2)
        radius = max(1, round(_scale(image)))
        ys, xs = rng.integers(0, height, count), rng.integers(0, width, count)
        tones = rng.choice([40.0, 230.0], count)
        for y, x, tone in zip(ys, xs, tones):
            cv2.circle(work, (int(x), int(y)), radius, (tone, tone, tone), -1)
        work = cv2.GaussianBlur(work, (0, 0), 0.6 * _scale(image))
    elif kind == "occlusion":
        height, width = image.shape[:2]
        side = np.sqrt(level)
        for x1, y1, x2, y2 in (boxes if boxes is not None else []):
            w, h = (x2 - x1) * side, (y2 - y1) * side
            if w < 1 or h < 1:
                continue
            left = rng.uniform(x1, x2 - w)
            top = rng.uniform(y1, y2 - h)
            a, b = int(max(0, left)), int(max(0, top))
            c, d = int(min(width, left + w)), int(min(height, top + h))
            work[b:d, a:c] = 128.0
    return np.clip(np.rint(work), 0, 255).astype(np.uint8)
