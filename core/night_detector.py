"""
night_detector.py
-------------------
Two-stage night-time movement detection:

  Stage 1 (cheap, runs every frame): background subtraction (MOG2) flags
  ANY pixel-level motion. IR/night CCTV footage is grainy, so this alone
  would false-trigger constantly on noise/branches/insects -- it's only
  a trigger, not a confirmation.

  Stage 2 (expensive, only runs when Stage 1 fires): low-light image
  enhancement (CLAHE histogram equalization) followed by the normal
  person/vehicle YOLO model, to confirm WHAT actually moved.

This two-stage design is why night monitoring doesn't need to run heavy
AI on every frame all night -- important for BOP power/compute limits.

Also includes simple day/night auto-detection from average frame
brightness, so the same pipeline can be left running 24/7 and only
apply night-specific logic when it's actually dark.
"""

import cv2
import numpy as np


def is_night_frame(frame, brightness_threshold=60):
    """Cheap heuristic: average grayscale brightness below threshold = night/low-light."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < brightness_threshold


def enhance_low_light(frame):
    """CLAHE (adaptive histogram equalization) on the luminance channel.
    Cheap (a few ms on CPU) and noticeably improves detector recall on
    dark/IR footage compared to running the raw frame through YOLO."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


class NightMotionDetector:
    def __init__(self, min_contour_area=400, history=300, var_threshold=25):
        # MOG2: adaptive background model, works reasonably on grainy IR video.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_contour_area = min_contour_area
        self._warmup_frames = 20  # let the background model stabilize before trusting it
        self._frame_count = 0

    def detect_motion(self, frame):
        """
        Stage 1. Returns (motion_detected: bool, motion_regions: list[(x,y,w,h)]).
        Call every frame -- this is cheap.
        """
        self._frame_count += 1
        fg_mask = self.bg_subtractor.apply(frame)

        # Clean up noise: erode then dilate to remove single-pixel speckle
        # while keeping real (larger) moving blobs intact.
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        if self._frame_count < self._warmup_frames:
            return False, []

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                regions.append((x, y, w, h))

        return len(regions) > 0, regions


class NightSurveillancePipeline:
    """
    Combines is_night_frame() + NightMotionDetector() + enhance_low_light()
    into one call, so app.py just does:

        night_events = night_pipeline.process(frame, person_vehicle_detector)

    and gets back confirmed detections (or an empty list if it's daytime,
    or if motion triggered but nothing was confirmed -- e.g. a moving
    branch or animal, so no false alert reaches the dashboard).
    """

    def __init__(self, detector, brightness_threshold=60):
        self.motion_detector = NightMotionDetector()
        self.detector = detector  # a PersonVehicleDetector instance (reused, not duplicated)
        self.brightness_threshold = brightness_threshold

    def process(self, frame):
        if not is_night_frame(frame, self.brightness_threshold):
            return {"is_night": False, "motion_triggered": False, "confirmed": []}

        motion, regions = self.motion_detector.detect_motion(frame)
        if not motion:
            return {"is_night": True, "motion_triggered": False, "confirmed": []}

        # Stage 2: only now do we pay for enhancement + full detection.
        enhanced = enhance_low_light(frame)
        confirmed = self.detector.detect(enhanced, use_tracking=True)

        return {
            "is_night": True,
            "motion_triggered": True,
            "motion_regions": regions,
            "confirmed": confirmed,  # list[Detection] -- empty if nothing real found
        }
