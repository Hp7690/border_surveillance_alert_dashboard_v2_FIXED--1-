"""
anpr.py
--------
Automatic Number Plate Recognition, two-stage (this is a very common,
well-proven pattern for ANPR -- detection and recognition are different
problems and are best handled by specialized components):

  Stage 1: PLATE LOCALIZATION -- a YOLOv8 model fine-tuned specifically
  to find license plates (bounding box only, doesn't read characters).

  Stage 2: OCR -- crop the plate region, clean it up (grayscale, threshold,
  upscale small plates), and run Tesseract OCR to read the characters.

Notes on real-world accuracy:
  - Indian plates have varied fonts/formats (state code + district code +
    series + number) -- Tesseract's generic OCR won't be perfect out of
    the box. A production system would fine-tune OCR on Indian plates or
    use a specialized ANPR OCR (e.g. PaddleOCR with a plate-specific
    recognition model) -- flagged here as a known improvement area rather
    than pretending Tesseract-out-of-the-box is production-grade.
  - We deliberately whitelist to alphanumeric characters only and run a
    light regex cleanup, since raw OCR output is noisy.
"""

import re
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO


def _preprocess_plate_crop(plate_img):
    """Clean up a cropped plate region to make OCR more reliable:
    upscale (small/far plates), grayscale, and adaptive threshold."""
    if plate_img.size == 0:
        return None
    h, w = plate_img.shape[:2]
    # Upscale small crops -- Tesseract does poorly on tiny text.
    if w < 300:
        scale = 300 / max(w, 1)
        plate_img = cv2.resize(plate_img, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # denoise while keeping edges sharp
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    return thresh


def _clean_plate_text(raw_text):
    """Keep only alphanumeric characters, uppercase -- typical plate format."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text)
    return cleaned.upper()


class PlateReading:
    __slots__ = ("bbox", "text", "detection_confidence", "vehicle_track_id")

    def __init__(self, bbox, text, detection_confidence, vehicle_track_id=None):
        self.bbox = bbox
        self.text = text
        self.detection_confidence = detection_confidence
        self.vehicle_track_id = vehicle_track_id

    def __repr__(self):
        return f"<PlateReading text='{self.text}' conf={self.detection_confidence:.2f}>"


class ANPR:
    def __init__(self, plate_model_path="models/license_plate_detector.pt",
                 conf_threshold=0.4, device="cpu", min_text_length=3):
        self.model = YOLO(plate_model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.min_text_length = min_text_length  # discard OCR noise shorter than this

    def _locate_plates(self, frame):
        results = self.model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )
        boxes = []
        if not results:
            return boxes
        r_boxes = results[0].boxes
        if r_boxes is None or len(r_boxes) == 0:
            return boxes
        xyxy = r_boxes.xyxy.cpu().numpy()
        confs = r_boxes.conf.cpu().numpy()
        for box, conf in zip(xyxy, confs):
            boxes.append((tuple(box.tolist()), float(conf)))
        return boxes

    def read(self, frame, vehicle_bbox=None, vehicle_track_id=None):
        """
        If vehicle_bbox is given, only searches for plates within that
        vehicle's box (faster, fewer false positives than scanning the
        whole frame -- use this from the main pipeline per tracked vehicle).
        Returns list[PlateReading].
        """
        if vehicle_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
            x1, y1 = max(0, x1), max(0, y1)
            search_frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            search_frame = frame
            offset_x, offset_y = 0, 0

        if search_frame.size == 0:
            return []

        readings = []
        for (px1, py1, px2, py2), conf in self._locate_plates(search_frame):
            plate_crop = search_frame[int(py1):int(py2), int(px1):int(px2)]
            processed = _preprocess_plate_crop(plate_crop)
            if processed is None:
                continue

            raw_text = pytesseract.image_to_string(
                processed,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            text = _clean_plate_text(raw_text)
            if len(text) < self.min_text_length:
                text = ""  # OCR failed/too noisy to trust -- still report the detection, just no text

            abs_bbox = (offset_x + px1, offset_y + py1, offset_x + px2, offset_y + py2)
            readings.append(PlateReading(
                bbox=abs_bbox, text=text, detection_confidence=conf,
                vehicle_track_id=vehicle_track_id,
            ))
        return readings
