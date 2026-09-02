"""
detector.py
-----------
Wraps YOLOv8 to detect + track only the object classes relevant to
border surveillance: person, bicycle, car, motorcycle, bus, truck.

Uses YOLO's built-in ByteTrack tracker (model.track()) so every object
gets a persistent ID across frames -- required for:
  - virtual fence intrusion (has this ID crossed the line?)
  - loitering detection (has this ID stayed too long in one place?)
  - trajectory analysis (suspicious activity module, Part 3)
"""

from ultralytics import YOLO
import numpy as np

# COCO class IDs we care about for border surveillance.
# (Full COCO list has 80 classes; we only keep people + vehicles.)
RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class Detection:
    """A single detected+tracked object in one frame."""

    __slots__ = ("track_id", "class_id", "class_name", "confidence", "bbox", "center")

    def __init__(self, track_id, class_id, class_name, confidence, bbox):
        self.track_id = track_id          # persistent ID across frames (None if tracking unavailable)
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox                  # (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox
        # "center" for a person = feet position (bottom-center of box) works
        # better for zone/fence checks than the true box center.
        self.center = ((x1 + x2) / 2, y2)

    def __repr__(self):
        return f"<Detection id={self.track_id} class={self.class_name} conf={self.confidence:.2f}>"


class PersonVehicleDetector:
    """
    Thin wrapper around YOLOv8 that returns clean Detection objects
    for only person/vehicle classes, with tracking IDs.
    """

    def __init__(self, model_path="models/yolov8n.pt", conf_threshold=0.35, device="cpu"):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        # Restrict inference to only the classes we care about -> faster + cleaner.
        self.class_filter = list(RELEVANT_CLASSES.keys())

    def detect(self, frame, use_tracking=True):
        """
        Run detection (+ tracking) on a single BGR frame (numpy array).
        Returns a list of Detection objects.
        """
        if use_tracking:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                classes=self.class_filter,
                device=self.device,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                classes=self.class_filter,
                device=self.device,
                verbose=False,
            )

        detections = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        )

        for box, conf, cls_id, tid in zip(xyxy, confs, cls_ids, track_ids):
            class_name = RELEVANT_CLASSES.get(int(cls_id), "unknown")
            detections.append(
                Detection(
                    track_id=int(tid) if tid is not None else None,
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=tuple(box.tolist()),
                )
            )
        return detections
