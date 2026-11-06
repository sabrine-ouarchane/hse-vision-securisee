"""
app/live/pipeline.py

Chaîne d'analyse HSE d'UNE image, partagée par le mode temps réel
(app/live/session.py) et le traitement de fichier (test_videov4.py) :

    contrôle d'intégrité de l'image (sabotage caméra)  ->
    détection + tracking  ->  association personne / EPI  ->  décision
    temporelle des comportements  ->  suivi des engins  ->  proximité
    homme-machine (état machine déclaré)  ->  annotation  ->  incidents
"""

import time

import cv2

from app.alerts.alert_manager import AlertManager
from app.alerts.incident_manager import IncidentManager
from app.config import load_camera_profile, load_detection_rules
from app.detection.hse_detector import HSEDetector
from app.epi.behavior_tracker import BehaviorTracker, ITEM_LABELS
from app.epi.compliance_checker import EPIComplianceChecker
from app.security.tamper_detector import TamperDetector, apply_prudence
from app.sensors.machine_state_provider import MachineStateProvider
from app.situations.proximity_checker import MachineTracker, ProximityChecker
from app.tracking.box_smoother import BoxSmoother
from app.visualization.overlay import OverlayRenderer


class HSEPipeline:
    def __init__(self, source_name: str, *, rules=None, profile=None, detector=None,
                 alerts_path=None, snapshot_dir=None, machine_states=None,
                 harness_required: bool | None = None, show_raw: bool | None = None,
                 audit=None, audit_context: dict | None = None):
        self.rules = rules or load_detection_rules()
        self.profile = profile or load_camera_profile(source_name)
        if harness_required is not None:
            self.profile["harness_required"] = harness_required
        self.harness_required = bool(self.profile.get("harness_required", False))

        self.detector = detector or HSEDetector(self.rules)
        self.detector.reset()
        self.tamper = TamperDetector(self.rules.get("tamper"), self.profile)
        self.smoother = BoxSmoother(alpha=0.6, max_missed_frames=5)
        self.checker = EPIComplianceChecker(
            harness_required=self.harness_required,
            confidence_thresholds=self.rules["violation_confidence_thresholds"],
        )
        self.behaviors = BehaviorTracker(
            self.rules, harness_required=self.harness_required,
            phone_active_area=self.profile.get("phone_active_area", "whole_frame"),
        )
        # Les réglages de proximité peuvent être affinés par caméra (profil "proximity")
        proximity_cfg = {**self.rules["proximity"], **self.profile.get("proximity", {})}
        self.machine_tracker = MachineTracker(proximity_cfg)
        self.proximity = ProximityChecker(proximity_cfg, self.profile.get("calibration"))
        self.machine_states = machine_states or MachineStateProvider()
        self.machine_id = self.profile.get("machine_id", "engin_pilote")

        temporal = self.rules["temporal"]
        self.incidents = IncidentManager(
            alert_manager=AlertManager(log_path=str(alerts_path), cooldown_frames=0) if alerts_path else None,
            snapshot_dir=snapshot_dir,
            close_after_s=temporal["incident_close_after_s"],
            merge_s=temporal["incident_merge_s"],
            source_name=self.profile["name"],
            audit=audit,
            audit_context=audit_context,
        )
        rendering = self.rules.get("rendering", {})
        self.max_side = rendering.get("max_side_px", 1280)
        self.renderer = OverlayRenderer(
            show_raw=rendering.get("show_raw_detections", True) if show_raw is None else show_raw
        )
        self._analysis_idx = 0

    def display_size(self, width: int, height: int) -> tuple[int, int, float]:
        scale = min(1.0, self.max_side / max(width, height)) if self.max_side else 1.0
        # dimensions paires : exigées par l'encodeur H.264
        return int(width * scale) // 2 * 2, int(height * scale) // 2 * 2, scale

    def process(self, frame, now: float, frame_idx: int, fps: float = 0.0, timestamp: str | None = None) -> dict:
        """
        Args:
            frame: image BGR à la résolution source.
            now: temps de la vidéo en secondes (temps réel pour une caméra).
            frame_idx: index de l'image dans la source.
            fps: cadence d'analyse effective, affichée dans le bandeau.
        """
        started = time.perf_counter()
        self._analysis_idx += 1
        shape = frame.shape[:2]

        # Intégrité de l'image AVANT YOLO (OpenCV seul, quelques ms)
        camera = self.tamper.update(frame, now)
        detections = self.detector.detect(frame)
        smoothed = self.smoother.smooth(detections, self._analysis_idx)
        statuses = self.checker.check(smoothed)
        persons = self.behaviors.update(statuses, now, shape)
        apply_prudence(persons, camera)  # caméra compromise : personne n'est « conforme »
        machines = self.machine_tracker.update(detections, now, shape)
        machine = self.machine_states.get(self.machine_id)
        machine["policy"] = self.machine_states.unknown_policy
        situations = self.proximity.check(persons, machines, now, shape, machine["state"], machine["policy"])
        inference_ms = (time.perf_counter() - started) * 1000

        counts = {
            "persons": len(persons),
            "violations": sum(p["status"] == "violation" for p in persons),
            "a_verifier": sum(p["status"] == "a_verifier" for p in persons),
            "conformes": sum(p["status"] == "conforme" for p in persons),
            "indetermines": sum(p["status"] == "indetermine" for p in persons),
            "machines": len(machines),
            "proximites": sum(s["severity"] is not None for s in situations),
        }
        width, height, scale = self.display_size(shape[1], shape[0])
        display = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
        annotated = self.renderer.render(
            display,
            persons=persons,
            machines=machines,
            situations=situations,
            raw_detections=detections,
            scale=scale,
            hud={
                "camera": self.profile["name"],
                "fps": fps,
                "machine": machine,
                "counts": counts,
                "timestamp": timestamp,
                "integrity": camera,
            },
        )

        active = []
        for person in persons:
            for item in person["violations"]:
                description = "usage du téléphone" if item == "telephone" else f"absence de {ITEM_LABELS[item]}"
                active.append({
                    "kind": item,
                    "track_id": person["track_id"],
                    "bbox": person["bbox"],
                    "severity": "critique",
                    "confidence": person["confidences"].get(item),
                    "message": f"Personne #{person['track_id']} : {description} (confirmée sur {person['observations']} observations)",
                })
        for situation in situations:
            if situation["severity"] is None:
                continue
            message = ProximityChecker.format_danger_message(situation)
            if situation["status"] == "a_verifier":
                message += " — à vérifier"
            active.append({
                "kind": "proximite",
                "track_id": situation["track_id"],
                "bbox": situation["bbox"],
                "severity": situation["severity"],
                "confidence": situation["confidence"],
                "message": message,
            })
        for event in camera["active"]:
            active.append({
                "kind": event["type"],
                "track_id": "camera",
                "bbox": (0, 0, shape[1], shape[0]),
                "severity": "critique",
                "confidence": None,
                "started_t": event["since"],
                "message": f"{event['label']} ({event['detail']}) — surveillance compromise, conformité non garantie",
            })
        integrity = {"compromised": camera["compromised"], "suspected": bool(camera["suspected"])}
        opened = self.incidents.update(active, now, frame_idx, annotated, scale, integrity)

        return {
            "annotated": annotated,
            "persons": persons,
            "machines": machines,
            "situations": situations,
            "machine": machine,
            "camera": camera,
            "counts": counts,
            "opened": opened,
            "inference_ms": round(inference_ms, 1),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        }
