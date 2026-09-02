"""
weapon_detector.py
-------------------
Wraps a YOLOv8 model fine-tuned specifically for weapons
(classes: Grenade, Gun, Knife, Pistol, handgun, rifle).

Why a SEPARATE model instead of adding "gun" as a class to the main
person/vehicle detector?
  - COCO (what yolov8n.pt is trained on) has no weapon classes at all.
  - Weapon detection needs a higher confidence bar and its own recall/
    precision tuning -- false negatives here are far more costly than
    false positives, so we deliberately keep this model and its alerting
    logic isolated from the general detector (see CONF_THRESHOLD below).

This is the CRITICAL severity path: every detection above threshold is
reported (no "wait and confirm over N frames" delay like suspicious
activity uses), because a missed weapon is worse than a false alarm.
"""

from ultralytics import YOLO

WEAPON_CLASSES = {
    0: "Grenade",
    1: "Gun",
    2: "Knife",
    3: "Pistol",
    4: "Handgun",
    5: "Rifle",
}

# Deliberately higher than the person/vehicle detector's threshold.
# Gun-shaped objects (phones, tools, umbrellas at odd angles) are a known
# false-positive source for weapon models trained on limited data, so we
# trade a bit of recall for fewer false CRITICAL alerts flooding the
# dashboard. Tune this down if the fine-tuned model proves reliable on
# real footage.
DEFAULT_CONF_THRESHOLD = 0.85


class WeaponDetection:
    __slots__ = ("class_name", "confidence", "bbox")

    def __init__(self, class_name, confidence, bbox):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox

    def __repr__(self):
        return f"<WeaponDetection {self.class_name} conf={self.confidence:.2f}>"


class WeaponDetector:
    def __init__(self, model_path="models/weapon_detection.pt",
                 conf_threshold=DEFAULT_CONF_THRESHOLD, device="cpu"):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame):
        """Run weapon detection on a single BGR frame. Returns list[WeaponDetection]."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            class_name = self.model.names.get(int(cls_id), WEAPON_CLASSES.get(int(cls_id), "weapon"))
            detections.append(WeaponDetection(class_name, float(conf), tuple(box.tolist())))
        return detections
