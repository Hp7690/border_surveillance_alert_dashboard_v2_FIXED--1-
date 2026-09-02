"""
face_recognizer.py
-------------------
Lightweight KNOWN-FACE whitelist recognition using OpenCV's built-in
LBPH recognizer (cv2.face.LBPHFaceRecognizer), so no extra heavy
dependency (dlib / face_recognition / deepface) is needed -- only
opencv-contrib-python, which already ships cv2.face.

Enrollment: call enroll(name, image_paths) once per person with a few
clear, front-facing photos (see enroll_faces.py for the CLI wrapper).

Recognition: call identify(face_crop_bgr) on a live camera face crop.
Returns (name, distance) if a confident match is found, else (None,
distance).

ACCURACY NOTE: LBPH is meaningfully less accurate than a modern deep
face-embedding model (ArcFace/FaceNet). It is a reasonable choice for
a SMALL enrolled watchlist (a handful of people) under fairly
consistent lighting/camera angle -- which matches this use case. If
you need to whitelist many people, or need it to work reliably across
very different lighting/angles, swap this module out for a proper
embedding-based recognizer; the enroll()/identify() interface is
intentionally kept simple so a swap wouldn't require touching app.py.

SECURITY NOTE: this suppresses PERSON/ZONE/SUSPICIOUS-ACTIVITY alerts
for recognized individuals. It deliberately does NOT suppress weapon
alerts (see app.py) -- a matched identity should never be able to
silence a weapon detection.
"""

import json
import os

import cv2
import numpy as np

FACE_SIZE = (200, 200)
# Bundled alongside this file rather than relying on cv2.data.haarcascades
# -- some opencv-contrib-python builds don't reliably ship/expose that
# path, which causes a confusing "Can't open file" error at runtime.
_HAAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")


class FaceRecognizer:
    def __init__(self, model_dir="models/known_faces", confidence_threshold=70):
        """
        confidence_threshold: LBPH's "confidence" output is actually a
        DISTANCE (lower = more similar, NOT a 0-1 probability). As a
        starting point: <50 is usually a strong match, 50-70 borderline,
        >80 is typically a different person. Tune this against your own
        enrollment photos + camera/lighting if you see false
        accepts/rejects -- pass a different value when constructing
        FaceRecognizer(), or edit the default here.
        """
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self.labels = {}  # int label id -> display name
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(_HAAR_PATH):
            raise FileNotFoundError(
                f"Haar cascade file missing at {_HAAR_PATH} -- "
                "face detection/recognition cannot work without it."
            )
        self.cascade = cv2.CascadeClassifier(_HAAR_PATH)
        self._trained = False
        self._load()

    def _labels_path(self):
        return os.path.join(self.model_dir, "labels.json")

    def _model_path(self):
        return os.path.join(self.model_dir, "lbph_model.yml")

    def _load(self):
        if os.path.exists(self._model_path()) and os.path.exists(self._labels_path()):
            self.recognizer.read(self._model_path())
            with open(self._labels_path(), "r") as f:
                self.labels = {int(k): v for k, v in json.load(f).items()}
            self._trained = True
            print(f"[FaceRecognizer] Loaded known face(s): {list(self.labels.values())}")
        else:
            print("[FaceRecognizer] No enrolled faces yet -- every face will be treated "
                  "as unknown (alerts will fire normally). Run enroll_faces.py to add "
                  "whitelisted people.")

    def _extract_face_gray(self, image_bgr):
        """Find the largest face in a STATIC enrollment photo, return a normalized grayscale crop."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # assume largest face = the subject
        return cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)

    def enroll(self, name, image_paths):
        """
        Add/extend a person's entry in the whitelist from one or more
        photo file paths, then retrain and persist the model.
        """
        samples, label_ids = [], []
        name_to_id = {v: k for k, v in self.labels.items()}
        label_id = name_to_id.get(name, (max(self.labels.keys()) + 1) if self.labels else 0)

        added = 0
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[FaceRecognizer] Could not read image: {path}")
                continue
            face = self._extract_face_gray(img)
            if face is None:
                print(f"[FaceRecognizer] No face found in: {path} -- skipped")
                continue
            samples.append(face)
            label_ids.append(label_id)
            added += 1

        if added == 0:
            raise ValueError(f"No usable face photos for '{name}' -- enrollment aborted.")

        self.labels[label_id] = name
        os.makedirs(self.model_dir, exist_ok=True)

        # LBPH needs (re)training from samples -- fine for a handful of
        # enrolled people, this completes in well under a second.
        if self._trained:
            self.recognizer.update(samples, np.array(label_ids))
        else:
            self.recognizer.train(samples, np.array(label_ids))
        self._trained = True

        self.recognizer.write(self._model_path())
        with open(self._labels_path(), "w") as f:
            json.dump({str(k): v for k, v in self.labels.items()}, f, indent=2)

        print(f"[FaceRecognizer] Enrolled '{name}' ({added} photo(s)). "
              f"Whitelist now: {list(self.labels.values())}")

    def identify(self, region_bgr):
        """
        region_bgr: an image region LIKELY CONTAINING a face -- e.g. a
        person's upper-body/head crop from a live camera frame. Pass a
        reasonably generous region (not an already-tightly-cropped face),
        since this re-runs the SAME Haar cascade face localization used
        during enroll() on it internally. This matters: comparing a face
        crop found one way (enrollment) to a crop found a different way
        (e.g. a pose-keypoint-derived box) tanks LBPH's accuracy, because
        it compares raw pixel patterns and is sensitive to
        framing/alignment. Re-localizing with the same method both times
        keeps the comparison apples-to-apples.

        Returns (name, distance) if confidently matched, else (None, distance_or_None).
        """
        if not self._trained or region_bgr is None or region_bgr.size == 0:
            return None, None
        face = self._extract_face_gray(region_bgr)
        if face is None:
            return None, None
        label_id, distance = self.recognizer.predict(face)
        if distance <= self.confidence_threshold:
            return self.labels.get(label_id), distance
        return None, distance
