"""
app/visualization/overlay.py

Annotation professionnelle des images analysées :
- boîte et libellé par personne suivie, couleur selon le statut décidé dans
  le temps (conforme / à vérifier / indéterminé / violation confirmée) ;
- détections brutes du modèle en trait fin (optionnel), distinctes des
  décisions ;
- engins avec leur état déclaré, liaison personne-engin colorée selon la
  situation, distance relative ou calibrée ;
- bandeau d'information (caméra, cadence d'analyse, état machine, compteurs) ;
- bandeau violet « SURVEILLANCE COMPROMISE » et cadre violet pendant un
  sabotage de caméra confirmé.

Le texte est rendu avec une police TrueType (Pillow) pour afficher
correctement les accents, et les libellés sont placés de façon à ne pas se
chevaucher.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Couleurs d'état communes à toute l'application (variables --st-* de dashboard/assets/hse.css), en RGB.
# Vérifié par test_security.OverlayColorTests : les deux fichiers ne peuvent pas diverger.
STATE_RGB = {
    "normal": (0x0B, 0x7A, 0x75),      # --st-normal : sarcelle, distincte du vert OCP
    "vigil": (0xB2, 0x5E, 0x00),       # --st-vigil : orange foncé, fond des étiquettes (texte blanc 4,7)
    "vigil_fill": (0xF0, 0xA0, 0x20),  # --st-vigil-fill : orange vif des traits
    "danger": (0xC6, 0x28, 0x28),      # --st-danger
    "off": (0x5F, 0x6B, 0x73),         # --st-off : indéterminé, hors surveillance
    "cyber": (0x6A, 0x2C, 0x91),       # --st-cyber : surveillance compromise (sabotage caméra)
}


def _bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


# Couleurs BGR (OpenCV) ; converties en RGB pour Pillow
COLORS = {
    "conforme": _bgr(STATE_RGB["normal"]),
    "violation": _bgr(STATE_RGB["danger"]),
    "a_verifier": _bgr(STATE_RGB["vigil_fill"]),
    "indetermine": _bgr(STATE_RGB["off"]),
    "marche": _bgr(STATE_RGB["danger"]),
    "arret": _bgr(STATE_RGB["normal"]),
    "inconnu": _bgr(STATE_RGB["vigil_fill"]),
    "danger": _bgr(STATE_RGB["danger"]),
    "danger_etat_inconnu": _bgr(STATE_RGB["danger"]),
    "vigilance": _bgr(STATE_RGB["vigil_fill"]),
    "machine_arretee": _bgr(STATE_RGB["off"]),
    "cyber": _bgr(STATE_RGB["cyber"]),
    "raw": (230, 230, 230),
    "panel": (32, 24, 18),
    "white": (255, 255, 255),
}
# Fond d'étiquette (texte blanc) : l'orange vif est trop clair pour du texte blanc (2,2), l'étiquette prend
# l'orange foncé ; les autres couleurs d'état portent le texte blanc à 5,2 ou plus.
LABEL_BACKGROUND = {COLORS["a_verifier"]: _bgr(STATE_RGB["vigil"])}
# Liseré sous chaque trait : clair sous une couleur foncée (lisible sur image sombre), sombre sous l'orange
# vif (lisible sur image claire). Le trait ou son liseré contraste toujours d'au moins 3:1 avec le fond.
HALO_LIGHT, HALO_DARK = (255, 255, 255), (34, 42, 27)


def _luminance(bgr):
    channels = [c / 255 for c in reversed(bgr)]
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def halo(color):
    return HALO_DARK if _luminance(color) > 0.3 else HALO_LIGHT


def label_background(color):
    return LABEL_BACKGROUND.get(tuple(color), tuple(color))
STATUS_TEXT = {
    "conforme": "CONFORME",
    "violation": "VIOLATION",
    "a_verifier": "À VÉRIFIER",
    "indetermine": "INDÉTERMINÉ",
}
ITEM_TEXT = {
    "casque": {"present": "Casque OK", "absent": "SANS CASQUE", "a_verifier": "Casque ?"},
    "gilet": {"present": "Gilet OK", "absent": "SANS GILET", "a_verifier": "Gilet ?"},
    "harnais": {"present": "Harnais OK", "absent": "SANS HARNAIS", "a_verifier": "Harnais ?"},
}
MACHINE_STATE_TEXT = {"marche": "EN MARCHE", "arret": "À L'ARRÊT", "inconnu": "ÉTAT INCONNU"}
SITUATION_TEXT = {
    "danger": "PROXIMITÉ DANGEREUSE",
    "danger_etat_inconnu": "PROXIMITÉ · ÉTAT MACHINE INCONNU",
    "a_verifier": "PROXIMITÉ À VÉRIFIER",
    "vigilance": "Vigilance",
    "machine_arretee": "Proche engin (arrêté)",
}
RAW_TEXT = {
    "Hardhat": "casque", "NO-Hardhat": "sans casque", "Safety Vest": "gilet",
    "NO-Safety Vest": "sans gilet", "Harness": "harnais", "NO-Harness": "sans harnais",
    "Phone": "téléphone",
}
FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf",
]
BOLD_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
]


def _rgb(bgr):
    return bgr[2], bgr[1], bgr[0]


def _load_font(candidates, size):
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


class OverlayRenderer:
    def __init__(self, show_raw: bool = True):
        self.show_raw = show_raw
        self._fonts = {}

    def _font(self, size, bold=False):
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = _load_font(BOLD_CANDIDATES if bold else FONT_CANDIDATES, size)
        return self._fonts[key]

    # ------------------------------------------------------------------
    def render(self, frame, *, persons, machines, situations, raw_detections, scale, hud):
        """
        Args:
            frame: image d'affichage (déjà redimensionnée), modifiée en place.
            scale: facteur coordonnées d'analyse -> coordonnées d'affichage.
            hud: dict (camera, fps, machine, counts, timestamp, integrity).
        """
        height, width = frame.shape[:2]
        unit = max(1.0, min(height, width) / 720)
        thickness = max(2, round(2 * unit))
        if hud.get("integrity", {}).get("compromised"):
            border = max(4, round(5 * unit))
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), COLORS["cyber"], border * 2)
            # liseré clair côté image : le cadre violet reste visible sur une image sombre
            cv2.rectangle(frame, (border, border), (width - 1 - border, height - 1 - border), halo(COLORS["cyber"]), 2)
        font_size = max(13, round(15 * unit))
        labels = []  # (priorité, ancre bbox, lignes [(texte, gras)], couleur)

        def s(bbox):
            return tuple(int(v * scale) for v in bbox)

        # Détections brutes (trait fin, sans décision)
        if self.show_raw:
            for det in raw_detections:
                if det["class_name"] not in RAW_TEXT:
                    continue
                x1, y1, x2, y2 = s(det["bbox"])
                color = COLORS["violation"] if det["class_name"].startswith("NO-") else COLORS["raw"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                labels.append((3, (x1, y1, x2, y2), [(f"{RAW_TEXT[det['class_name']]} {det['confidence']:.2f}", False)], color, "small"))

        # Engins
        machine_state = hud.get("machine", {}).get("state", "inconnu")
        for machine in machines:
            x1, y1, x2, y2 = s(machine["bbox"])
            color = COLORS[machine_state]
            self._dashed_rect(frame, (x1, y1), (x2, y2), color, thickness, gap=max(8, int(12 * unit)))
            text = f"ENGIN #{machine['id']} · {MACHINE_STATE_TEXT[machine_state]}"
            if not machine.get("visible", True):
                text += " (masqué)"
            labels.append((1, (x1, y1, x2, y2), [(text, True)], color, "inside"))

        # Liaisons personne-engin
        for situation in situations:
            color = COLORS[situation["status"]]
            p1 = tuple(int(v * scale) for v in situation["person_point"])
            p2 = tuple(int(v * scale) for v in situation["machine_point"])
            line_thickness = thickness + (1 if situation["severity"] == "critique" else 0)
            for paint, extra in ((halo(color), 2), (color, 0)):
                cv2.line(frame, p1, p2, paint, line_thickness + extra, cv2.LINE_AA)
                cv2.circle(frame, p1, line_thickness + 2 + extra // 2, paint, -1, cv2.LINE_AA)
                cv2.circle(frame, p2, line_thickness + 2 + extra // 2, paint, -1, cv2.LINE_AA)

        # Personnes
        situation_by_track = {sit["track_id"]: sit for sit in situations}
        for person in persons:
            x1, y1, x2, y2 = s(person["bbox"])
            status = person["status"]
            color = COLORS[status]
            situation = situation_by_track.get(person["track_id"])
            if status == "violation" or (situation and situation["severity"] == "critique"):
                alert_color = COLORS["violation"]
                tint = frame[max(0, y1):y2, max(0, x1):x2]
                if tint.size:
                    red = np.full_like(tint, alert_color)
                    frame[max(0, y1):y2, max(0, x1):x2] = cv2.addWeighted(tint, 0.82, red, 0.18, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), halo(alert_color), thickness + 3, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, thickness + 1, cv2.LINE_AA)
            else:
                self._corner_rect(frame, (x1, y1), (x2, y2), color, thickness)

            track = f"#{person['track_id']}" if person["track_id"] is not None else "#?"
            lines = [(f"{track} · {STATUS_TEXT[status]}", True)]
            details = []
            for item, texts in ITEM_TEXT.items():
                text = texts.get(person["items"].get(item))
                if text:
                    details.append(text)
            phone = person["items"].get("telephone")
            if phone == "detecte":
                details.append("TÉLÉPHONE")
            elif phone == "a_verifier":
                details.append("Téléphone ?")
            elif phone == "hors_zone":
                details.append("Tél. hors zone active")
            if details:
                lines.append((" · ".join(details), False))
            if status == "indetermine" and person["reasons"]:
                lines.append((person["reasons"][0], False))
            if situation:
                distance = (f"{situation['distance_m']:.1f} m" if situation.get("distance_m") is not None
                            else f"{situation['distance_rel']:.1f}× hauteur pers.")
                lines.append((f"{SITUATION_TEXT[situation['status']]} · {distance}", situation["severity"] == "critique"))
                if situation["severity"] == "critique":
                    color = COLORS["violation"]
            priority = 0 if status == "violation" or situation else 1
            label_color = COLORS["violation"] if status == "violation" else color
            labels.append((priority, (x1, y1, x2, y2), lines, label_color, "above"))

        return self._draw_text(frame, labels, hud, font_size, unit)

    # ------------------------------------------------------------------
    PENDING_TITLE = "ANALYSE EN COURS — aucune détection affichée pour l'instant"
    PENDING_DETAIL = "Image brute : l'absence de cadre ne signifie pas l'absence de violation."

    def render_pending(self, frame):
        """
        Première image brute, affichée avant la fin de la première analyse.
        Le bandeau est dessiné dans l'image même (et non seulement dans la page) :
        une capture d'écran de cette image ne peut pas passer pour « aucune violation ».
        """
        height, width = frame.shape[:2]
        unit = max(1.0, min(height, width) / 720)
        border = max(4, round(5 * unit))
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), COLORS["vigilance"], border * 2)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        pad = max(6, round(8 * unit))
        size = max(13, round(19 * unit))
        while size > 11:  # le titre doit tenir sur la largeur (vidéo verticale comprise)
            bold, regular = self._font(size, True), self._font(max(11, int(size * 0.8)))
            if max(draw.textlength(self.PENDING_TITLE, font=bold),
                   draw.textlength(self.PENDING_DETAIL, font=regular)) <= width - 4 * pad:
                break
            size -= 1
        title = draw.textbbox((0, 0), self.PENDING_TITLE, font=bold)
        detail = draw.textbbox((0, 0), self.PENDING_DETAIL, font=regular)
        band_h = (title[3] - title[1]) + (detail[3] - detail[1]) + 4 * pad
        top = (height - band_h) // 2
        draw.rectangle((0, top, width, top + band_h), fill=_rgb(LABEL_BACKGROUND[COLORS["a_verifier"]]))
        y = top + pad
        draw.text((pad * 2, y - title[1]), self.PENDING_TITLE, font=bold, fill=(255, 255, 255))
        y += title[3] - title[1] + pad * 2
        draw.text((pad * 2, y - detail[1]), self.PENDING_DETAIL, font=regular, fill=(255, 255, 255))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    @staticmethod
    def _corner_rect(frame, p1, p2, color, thickness):
        x1, y1 = p1
        x2, y2 = p2
        length = max(10, int(min(x2 - x1, y2 - y1) * 0.22))
        corners = []
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            corners += [((cx, cy), (cx + dx * length, cy)), ((cx, cy), (cx, cy + dy * length))]
        for paint, extra in ((halo(color), 2), (color, 0)):  # tous les liserés d'abord, puis la couleur
            cv2.rectangle(frame, p1, p2, paint, 1 + extra, cv2.LINE_AA)
            for a, b in corners:
                cv2.line(frame, a, b, paint, thickness + 1 + extra, cv2.LINE_AA)

    @staticmethod
    def _dashed_rect(frame, p1, p2, color, thickness, gap):
        x1, y1 = p1
        x2, y2 = p2
        dashes = []
        for start, end in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            length = int(np.hypot(end[0] - start[0], end[1] - start[1]))
            for offset in range(0, length, gap * 2):
                a = offset / max(length, 1)
                b = min(offset + gap, length) / max(length, 1)
                pa = (int(start[0] + (end[0] - start[0]) * a), int(start[1] + (end[1] - start[1]) * a))
                pb = (int(start[0] + (end[0] - start[0]) * b), int(start[1] + (end[1] - start[1]) * b))
                dashes.append((pa, pb))
        for paint, extra in ((halo(color), 2), (color, 0)):
            for pa, pb in dashes:
                cv2.line(frame, pa, pb, paint, thickness + extra, cv2.LINE_AA)

    def _draw_text(self, frame, labels, hud, font_size, unit):
        height, width = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image, "RGBA")
        placed = []
        pad = max(3, round(4 * unit))

        hud_bottom = self._draw_hud(draw, width, hud, font_size, pad)
        placed.append((0, 0, width, hud_bottom))

        for priority, anchor, lines, color, mode in sorted(labels, key=lambda item: item[0]):
            small = mode == "small"
            size = max(11, int(font_size * 0.8)) if small else font_size
            fonts = [self._font(size, bold) for _, bold in lines]
            sizes = [draw.textbbox((0, 0), text, font=font) for (text, _), font in zip(lines, fonts)]
            box_w = max(b[2] - b[0] for b in sizes) + 2 * pad
            line_h = [b[3] - b[1] + pad for b in sizes]
            box_h = sum(line_h) + pad
            x1, y1, x2, y2 = anchor
            if mode == "inside":
                candidates = [(x1 + 2, y1 + 2), (x1, y1 - box_h)]
            elif small:
                candidates = [(x2 + 2, y1), (x1, y1 - box_h), (x1, y2 + 2)]
            else:
                candidates = [(x1, y1 - box_h - 2), (x1, y2 + 2), (x1, y1 + 2)]
            position = self._place(candidates, box_w, box_h, placed, width, height, allow_skip=small)
            if position is None:
                continue
            lx, ly = position
            placed.append((lx, ly, lx + box_w, ly + box_h))
            if small:
                draw.rectangle((lx, ly, lx + box_w, ly + box_h), fill=(20, 20, 20, 170))
                text_color = (255, 140, 140) if color == COLORS["violation"] else _rgb(color)
                draw.text((lx + pad, ly + pad // 2 - sizes[0][1]), lines[0][0], font=fonts[0], fill=text_color)
                continue
            # fond opaque : le contraste du texte blanc ne dépend pas de l'image dessous
            draw.rectangle((lx, ly, lx + box_w, ly + line_h[0] + pad // 2), fill=_rgb(label_background(color)) + (255,))
            if len(lines) > 1:
                draw.rectangle((lx, ly + line_h[0] + pad // 2, lx + box_w, ly + box_h), fill=(18, 22, 28, 215))
            cursor = ly + pad // 2
            for (text, _), font, bbox, step in zip(lines, fonts, sizes, line_h):
                draw.text((lx + pad, cursor - bbox[1] + pad // 2), text, font=font, fill=(255, 255, 255))
                cursor += step
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _place(candidates, box_w, box_h, placed, width, height, allow_skip):
        def free(x, y):
            return all(x + box_w <= a or x >= c or y + box_h <= b or y >= d for a, b, c, d in placed)

        for x, y in candidates:
            x = min(max(0, x), max(0, width - box_w))
            y = min(max(0, y), max(0, height - box_h))
            if free(x, y):
                return x, y
        if allow_skip:
            return None
        x, y = candidates[0]
        x = min(max(0, x), max(0, width - box_w))
        y = min(max(0, y), max(0, height - box_h))
        for _ in range(12):  # glisse vers le bas jusqu'à trouver une place libre
            y = min(y + box_h // 2 + 2, max(0, height - box_h))
            if free(x, y):
                break
        return x, y

    def _draw_hud(self, draw, width, hud, font_size, pad) -> int:
        bold = self._font(font_size, True)
        regular = self._font(max(11, int(font_size * 0.85)))
        machine = hud.get("machine", {})
        counts = hud.get("counts", {})
        state = machine.get("state", "inconnu")
        left = f"{hud.get('camera', 'Caméra')}"
        right = hud.get("timestamp") or datetime.now().strftime("%H:%M:%S")
        line2 = (f"Analyse {hud.get('fps', 0):.1f} img/s · Personnes {counts.get('persons', 0)}"
                 f" · Violations {counts.get('violations', 0)} · À vérifier {counts.get('a_verifier', 0)}")
        if counts.get("machines"):
            line3 = f"Engin {machine.get('machine_id', '?')} : {MACHINE_STATE_TEXT[state]}"
            if state == "inconnu":
                line3 += " — règle " + ("prudente (traité comme en marche)" if machine.get("policy") == "prudent" else "surveillance")
            line3_color = _rgb(COLORS[state])
        else:
            line3, line3_color = "Aucun engin détecté dans le champ", (170, 180, 175)
        row1 = draw.textbbox((0, 0), left, font=bold)
        row2 = draw.textbbox((0, 0), line2, font=regular)
        total_h = (row1[3] - row1[1]) + 2 * (row2[3] - row2[1]) + 5 * pad
        draw.rectangle((0, 0, width, total_h), fill=(14, 20, 28, 190))
        y = pad
        draw.text((pad * 2, y - row1[1]), left, font=bold, fill=(255, 255, 255))
        rw = draw.textbbox((0, 0), right, font=bold)
        draw.text((width - (rw[2] - rw[0]) - pad * 2, y - rw[1]), right, font=bold, fill=(255, 255, 255))
        y += row1[3] - row1[1] + pad * 2
        draw.text((pad * 2, y - row2[1]), line2, font=regular, fill=(220, 226, 232))
        y += row2[3] - row2[1] + pad
        # état de l'engin : pastille de couleur + texte blanc (du texte coloré sur fond sombre serait sous 4,5)
        dot = row2[3] - row2[1]
        draw.ellipse((pad * 2, y, pad * 2 + dot, y + dot), fill=line3_color, outline=(255, 255, 255))
        draw.text((pad * 3 + dot, y - row2[1]), line3, font=regular, fill=(255, 255, 255))

        integrity = hud.get("integrity") or {}
        if integrity.get("compromised"):
            title = "SURVEILLANCE COMPROMISE — " + " · ".join(e["label"] for e in integrity["active"])
            event = integrity["active"][0]
            detail = (f"{event['detail']} · depuis {event['duration_s']:.0f} s · "
                      "aucune personne ne peut être déclarée conforme")
            band_top = total_h
            band_h = (row1[3] - row1[1]) + (row2[3] - row2[1]) + 4 * pad
            draw.rectangle((0, band_top, width, band_top + band_h), fill=_rgb(COLORS["cyber"]) + (235,))
            y = band_top + pad
            draw.text((pad * 2, y - row1[1]), title, font=bold, fill=(255, 255, 255))
            y += row1[3] - row1[1] + pad
            draw.text((pad * 2, y - row2[1]), detail, font=regular, fill=(240, 225, 245))
            total_h = band_top + band_h
        return total_h
