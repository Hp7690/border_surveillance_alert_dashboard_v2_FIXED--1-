"""
app.py
------
Main entrypoint. Runs:
  1. A background thread that reads the video source frame-by-frame,
     runs detection + virtual fence checks, draws annotations, and
     raises alerts via AlertManager.
  2. A Flask web server that:
       - streams the annotated video as MJPEG (/video_feed)
       - serves the admin dashboard (/)
       - pushes real-time alerts to connected browsers via SocketIO

Usage:
    python app.py --source test_videos/synthetic_test.mp4
    python app.py --source 0                     # webcam
    python app.py --source rtsp://<camera-url>    # real CCTV stream
"""

import eventlet
import eventlet.tpool
eventlet.monkey_patch(thread=False)

import argparse
import csv
import io
import json
import os
import secrets
import threading
import time
import uuid
from collections import deque

import cv2
from flask import (Flask, Response, render_template, jsonify, request,
                    send_from_directory, session, redirect, url_for)
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from core.detector import PersonVehicleDetector
from core.virtual_fence import VirtualFence
from core.alert_manager import AlertManager
from core.event_log import EventLogger
from core.ledger import EvidenceLedger, ACTION_CODES
from core.webhooks import WebhookManager
from core.auth import verify_user, has_any_users
from core.weapon_detector import WeaponDetector
from core.suspicious_activity import SuspiciousActivityDetector
from core.night_detector import NightSurveillancePipeline, is_night_frame
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.anpr import ANPR

from datetime import timedelta


def _get_or_create_secret_key():
    """
    A hardcoded/known SECRET_KEY lets anyone forge a valid login
    session -- generate a real random one on first run and persist it
    to disk so sessions survive restarts, instead of hardcoding a
    known string in source control.
    """
    path = os.path.join("data", "secret_key.txt")
    os.makedirs("data", exist_ok=True)
    if os.path.exists(path):
        with open(path) as f:
            key = f.read().strip()
            if key:
                return key
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    return key


app = Flask(__name__)
app.config["SECRET_KEY"] = _get_or_create_secret_key()
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB cap on uploaded videos
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

event_logger = EventLogger()  # persistent SQLite + snapshot log (data/events.db, data/snapshots/)
webhook_manager = WebhookManager()  # external C2/SIEM push integration (webhooks.json)
alert_manager = AlertManager(socketio=socketio, event_logger=event_logger, webhook_manager=webhook_manager)

# ---- Multi-camera shared state ----
# Every camera gets its own entry, keyed by camera_id, in each of these:
camera_frames = {}       # camera_id -> latest annotated BGR frame (numpy array)
camera_frames_lock = threading.Lock()
camera_stats = {}        # camera_id -> {"fps":..., "frame_count":..., "active_tracks":...}
camera_registry = {}     # camera_id -> {"source": str(...), "status": "starting"|"running"|"error", "error": str|None}

# ---- Video clip recording ----
# A short clip (pre-alert context + post-alert follow-through) is saved
# for every alert at or above these severities -- much more useful for
# incident review than a single snapshot frame, especially for a
# CRITICAL weapon detection or a confirmed HIGH-severity intrusion.
CLIP_TRIGGER_SEVERITIES = {"CRITICAL", "HIGH"}
CLIP_PRE_SECONDS = 5    # how much buffered history to include before the trigger
CLIP_POST_SECONDS = 5   # how long to keep recording after the trigger
CLIP_FPS = 8.0          # output clip frame rate (doesn't need to match live FPS)
CLIP_DIR = os.path.join("data", "clips")

# ---- Tamper-evident evidence ledger / chain of custody (core/ledger.py) ----
# Every CRITICAL/HIGH alert (plus face/plate events) is hash-chained + signed
# together with the SHA-256 of its snapshot and clip; every operator
# acknowledge / view / action / export is appended to the same chain.
# Anchors (head hash every N blocks) are written to data/ledger_anchors.jsonl
# AND pushed to every registered webhook, so a copy of the chain head lives
# OFF this machine -- that is what makes rewriting history detectable.
ledger = EvidenceLedger(
    snapshot_dir=event_logger.snapshot_dir, clip_dir=CLIP_DIR,
    anchor_hook=lambda anchor: webhook_manager.notify(anchor),
)
alert_manager.ledger = ledger

# ---- Uploaded-video analysis ----
UPLOAD_DIR = os.path.join("data", "uploads")
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}


def _iou(box_a, box_b):
    """Intersection-over-union of two (x1,y1,x2,y2) boxes -- used to match
    a detected face's pose bbox to a tracked person's detection bbox."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class VideoProcessor(threading.Thread):
    def _is_known_person(self, bbox):
        """Returns the whitelisted name if this bbox overlaps a recently
        (within grace period) confirmed known person, else None. Position
        -based rather than track_id-based -- see comment at _known_people."""
        now = time.time()
        for kp in self._known_people:
            if now - kp["last_seen"] > self._KNOWN_GRACE_SECONDS:
                continue
            if _iou(bbox, kp["bbox"]) >= self._KNOWN_IOU_THRESHOLD:
                return kp["name"]
        return None

    def _mark_known(self, name, bbox):
        """Record/refresh a confirmed whitelisted-face sighting at this position."""
        now = time.time()
        for kp in self._known_people:
            if kp["name"] == name:
                kp["bbox"] = bbox
                kp["last_seen"] = now
                break
        else:
            self._known_people.append({"name": name, "bbox": bbox, "last_seen": now})
        # Periodically prune stale entries so this list doesn't grow forever.
        self._known_people = [k for k in self._known_people if now - k["last_seen"] <= self._KNOWN_GRACE_SECONDS * 4]

    def __init__(self, source, camera_id="BOP-01", loop=True, frame_skip=1):
        super().__init__(daemon=True)
        self.source = source
        self.camera_id = camera_id
        self.loop = loop  # True for live cameras/demo video; False for one-shot uploaded-video analysis
        # For a live camera you want to look at as many frames as
        # possible, so this stays 1. For a one-shot uploaded-video
        # analysis there's no "real-time" requirement -- skipping raw
        # frames between AI passes multiplies throughput roughly by
        # frame_skip, since the expensive part (YOLO inference) is what
        # dominates runtime, not the cheap cap.read() calls.
        self.frame_skip = max(1, frame_skip)
        self.detector = PersonVehicleDetector(model_path="models/yolov8n.pt")
        self.weapon_detector = WeaponDetector(model_path="models/weapon_detection.pt")

        # Virtual fence zone: a polygon roughly covering the lower-right
        # portion of the frame, simulating "the restricted border strip".
        # In a real deployment this is drawn once per camera during setup
        # (an admin clicks points on a still frame from that camera).
        self.fence = VirtualFence(
            name="Restricted Border Zone",
            polygon_points=[(550, 250), (960, 250), (960, 540), (550, 540)],
            loiter_seconds=8,
        )

        self.suspicious_detector = SuspiciousActivityDetector(
            pose_model_path="models/yolov8n-pose.pt",
            fence=self.fence,
        )

        # Night-time pipeline reuses the same person/vehicle detector
        # (no duplicate model loaded) -- only adds motion-triggering +
        # low-light enhancement in front of it.
        self.night_pipeline = NightSurveillancePipeline(detector=self.detector)

        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()  # known-face whitelist (see enroll_faces.py)
        # Cache of recently-confirmed whitelisted people, matched by
        # POSITION (bbox overlap) rather than the detector's track_id.
        # This matters because the underlying tracker can lose and
        # reassign a new track_id to the same physical person (we've
        # observed this happening frequently); keying purely off
        # track_id would then treat a re-identified person as a brand
        # new "unknown" every time their ID churns, generating spurious
        # generic person_detected log entries even for enrolled people.
        # Each entry: {"name": str, "bbox": (x1,y1,x2,y2), "last_seen": float}
        self._known_people = []
        self._KNOWN_GRACE_SECONDS = 3.0
        self._KNOWN_IOU_THRESHOLD = 0.25
        self.anpr = ANPR(plate_model_path="models/license_plate_detector.pt")

        # Run the (heavier) weapon + pose models only every Nth frame to
        # keep the pipeline real-time on CPU. Person/vehicle detection +
        # tracking still runs every frame (tracking needs continuity).
        self.heavy_model_stride = 3
        self._frame_idx = 0

        self._stop_flag = False
        self._anpr_error_logged = False
        self._prev_time = time.time()
        self._frame_count = 0

        # ---- Video clip recording (for CRITICAL/HIGH alerts) ----
        # Rolling buffer of recent annotated frames so a saved clip can
        # include a few seconds of PRE-alert context, not just what
        # happens after the trigger. Memory cost is modest: frames are
        # already resized in mjpeg_generator for streaming, but this
        # buffer holds the full-res annotated frame -- at ~6 FPS on CPU
        # and CLIP_PRE_SECONDS=5, that's ~30 frames per camera.
        self._frame_buffer = deque()  # (timestamp, frame) newest at the end
        self._active_clips = []  # list of {"alert_id", "frames": [(ts,frame)], "end_time"}

    def stop(self):
        self._stop_flag = True

    def run(self):
        camera_registry[self.camera_id] = camera_registry.get(self.camera_id, {})
        # Windows webcams: OpenCV's default backend (MSMF) often fails to open a local
        # camera index, while DirectShow works. Files/RTSP URLs keep the default backend.
        if isinstance(self.source, int) and os.name == "nt":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.source)   # fall back to default backend
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source for camera '{self.camera_id}': {self.source}")
            camera_registry[self.camera_id]["status"] = "error"
            camera_registry[self.camera_id]["error"] = "Could not open video source"
            return

        camera_registry[self.camera_id]["status"] = "running"
        camera_registry[self.camera_id]["error"] = None
        # For uploaded files this is a real, known frame count -- used to
        # show a progress bar. For live cameras/streams this is usually 0
        # (unknown), which the UI treats as "no progress bar, just a spinner".
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            total = 0
        camera_registry[self.camera_id]["total_frames"] = max(0, total)

        while not self._stop_flag:
            ret, frame = eventlet.tpool.execute(cap.read)
            if not ret:
                if self.loop:
                    # loop the video for continuous demo purposes
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # One-shot analysis (uploaded video) -- reached the end,
                    # stop here instead of looping.
                    break

            # Cheaply advance past (frame_skip - 1) extra raw frames --
            # only the LAST one read gets run through the (expensive) AI
            # pipeline below. cap.read() itself is comparatively cheap;
            # skipping full detection on the discarded frames is what
            # actually speeds up one-shot video analysis.
            if self.frame_skip > 1:
                reached_end = False
                for _ in range(self.frame_skip - 1):
                    ret2, frame2 = eventlet.tpool.execute(cap.read)
                    if not ret2:
                        reached_end = True
                        break
                    frame = frame2
                if reached_end:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

            # True position in the source file, independent of frame_skip
            # -- used for an accurate progress bar (self._frame_idx below
            # only counts frames actually RUN through the AI pipeline).
            try:
                camera_stats.setdefault(self.camera_id, {})["raw_frame_pos"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            except Exception:
                pass

            self._frame_idx += 1
            run_heavy = (self._frame_idx % self.heavy_model_stride == 0)

            try:
                self._process_frame(frame, run_heavy)
            except Exception as e:
                # Any single-frame failure (a model hiccup, a malformed
                # detection, etc.) must never take down the whole live
                # surveillance loop -- log it and keep monitoring.
                print(f"[WARN] Frame processing error (skipped this frame): {e}")

        cap.release()
        camera_registry[self.camera_id]["status"] = "stopped" if self._stop_flag else (
            "completed" if not self.loop else "stopped"
        )

    def _process_frame(self, frame, run_heavy):
        # Cheap brightness check -- computed early (not just in the
        # night-detection section below) so every alert raised this
        # frame can factor time-of-day into its threat score.
        is_night = is_night_frame(frame)

        detections = eventlet.tpool.execute(self.detector.detect, frame, use_tracking=True)
        fence_events = self.fence.evaluate(detections)

        # Dashboard demo alert: surface every newly observed person or
        # vehicle (AlertManager cooldown prevents frame-by-frame spam).
        VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")
        for det in detections:
            if det.class_name == "person":
                if self._is_known_person(det.bbox):
                    continue  # whitelisted face -- no alert
                alert = alert_manager.raise_alert(
                    event_type="person_detected",
                    camera_id=self.camera_id,
                    details={
                        "track_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 2),
                    },
                    frame=frame,
                    is_night=is_night,
                )
                self._maybe_record_clip(alert)
            elif det.class_name in VEHICLE_CLASSES:
                alert = alert_manager.raise_alert(
                    event_type="vehicle_detected",
                    camera_id=self.camera_id,
                    details={
                        "track_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 2),
                    },
                    frame=frame,
                    is_night=is_night,
                )
                self._maybe_record_clip(alert)

        # Any event from the fence -> raise via central alert manager
        for ev in fence_events:
            if self._is_known_person(ev.get("bbox", (0, 0, 0, 0))):
                continue  # whitelisted face -- no alert
            event_type = ev["type"]  # "intrusion" or "loitering"
            alert = alert_manager.raise_alert(
                event_type=event_type,
                camera_id=self.camera_id,
                details={
                    "track_id": ev["track_id"],
                    "class_name": ev["class_name"],
                    "confidence": round(ev["confidence"], 2),
                    "zone": ev["zone"],
                    "extra": ev.get("duration_seconds"),
                },
                frame=frame,
                is_night=is_night,
            )
            self._maybe_record_clip(alert)

        # ---- Gun / weapon detection (CRITICAL path) ----
        if run_heavy:
            weapons = eventlet.tpool.execute(self.weapon_detector.detect, frame)
            for w in weapons:
                alert = alert_manager.raise_alert(
                    event_type="weapon_detected",
                    camera_id=self.camera_id,
                    details={
                        "class_name": w.class_name,
                        "confidence": round(w.confidence, 2),
                    },
                    frame=frame,
                    is_night=is_night,
                )
                self._maybe_record_clip(alert)
                x1, y1, x2, y2 = [int(v) for v in w.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"WEAPON: {w.class_name} {w.confidence:.2f}",
                            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ---- Suspicious activity (posture + trajectory rules) ----
            # Pose estimation is run once here and reused for face
            # detection below, avoiding a duplicate inference pass.
            poses = eventlet.tpool.execute(self.suspicious_detector.pose_detector.detect, frame)
            sus_events = self.suspicious_detector.evaluate(frame, detections, poses=poses)
            for ev in sus_events:
                if self._is_known_person(ev.get("bbox", (0, 0, 0, 0))):
                    continue  # whitelisted face -- no alert
                alert = alert_manager.raise_alert(
                    event_type=ev["type"],
                    camera_id=self.camera_id,
                    details={
                        "track_id": ev["track_id"],
                        "class_name": "person",
                        **ev.get("extra", {}),
                    },
                    frame=frame,
                    is_night=is_night,
                )
                self._maybe_record_clip(alert)
                x1, y1, x2, y2 = [int(v) for v in ev["bbox"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(frame, ev["type"].upper(), (x1, min(frame.shape[0] - 5, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            # ---- Face detection + known-face whitelist check ----
            # (derived from the poses above -- no extra detection model)
            persons_in_frame = [d for d in detections if d.class_name == "person"]
            for pose in poses:
                face = self.face_detector.from_pose(pose)
                if face is None:
                    continue

                # Match this pose to a tracked person via IoU (same
                # approach suspicious_activity.py uses) so we know
                # WHICH track_id this face belongs to.
                best_det, best_iou = None, 0.0
                for det in persons_in_frame:
                    score = _iou(pose.bbox, det.bbox)
                    if score > best_iou:
                        best_iou, best_det = score, det
                track_id = best_det.track_id if (best_det is not None and best_iou >= 0.3) else None

                fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(frame.shape[1], fx2), min(frame.shape[0], fy2)

                # Recognition uses a MORE GENEROUS region than the tight
                # face box above -- the tracked person's own bbox if we
                # matched one, since that gives the internal Haar
                # detector enough context to reliably (re-)localize the
                # face itself (matching how enroll() finds faces in
                # photos). Falls back to the pose-derived box if no
                # track matched.
                if best_det is not None:
                    rx1, ry1, rx2, ry2 = [int(v) for v in best_det.bbox]
                    rx1, ry1 = max(0, rx1), max(0, ry1)
                    rx2, ry2 = min(frame.shape[1], rx2), min(frame.shape[0], ry2)
                    region = frame[ry1:ry2, rx1:rx2]
                else:
                    region = frame[fy1:fy2, fx1:fx2]

                name, dist = self.face_recognizer.identify(region) if region.size else (None, None)
                if self._frame_idx % 30 == 0:  # occasional debug line, not every heavy frame
                    print(f"[FaceRecognizer] nearest_match={name or 'unknown'} distance={dist} "
                          f"(threshold={self.face_recognizer.confidence_threshold})")

                if name:
                    # Whitelisted -- cache by POSITION (the person's own
                    # detection bbox, if we matched one) so
                    # person/zone/suspicious alerts stop firing for them
                    # even if the tracker later reassigns a new
                    # track_id, and skip the face_detected alert itself.
                    mark_bbox = best_det.bbox if best_det is not None else face.bbox
                    self._mark_known(name, mark_bbox)

                    # Still worth a RECORD in incident history (e.g. "who
                    # was on camera and when") even though no live
                    # alert/sound should fire for a whitelisted person.
                    alert_manager.log_silent(
                        event_type="known_person_seen",
                        camera_id=self.camera_id,
                        details={"name": name, "track_id": track_id},
                        frame=frame,
                    )

                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} (cleared)", (fx1, max(0, fy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                    alert = alert_manager.raise_alert(
                        event_type="face_detected",
                        camera_id=self.camera_id,
                        details={"class_name": "person"},
                        frame=frame,
                        is_night=is_night,
                    )
                    self._maybe_record_clip(alert)

            # ---- ANPR (search inside each tracked vehicle's box) ----
            for det in detections:
                if det.class_name not in ("car", "truck", "bus", "motorcycle"):
                    continue
                try:
                    plates = eventlet.tpool.execute(
                        self.anpr.read, frame, vehicle_bbox=det.bbox, vehicle_track_id=det.track_id
                    )
                except Exception as e:
                    # A missing/broken OCR backend (e.g. Tesseract not
                    # installed) must never take down the whole
                    # detection thread -- just skip plate-reading for
                    # this frame and keep going.
                    if not self._anpr_error_logged:
                        print(f"[WARN] ANPR/plate-reading disabled: {e}")
                        self._anpr_error_logged = True
                    plates = []
                for plate in plates:
                    alert = alert_manager.raise_alert(
                        event_type="vehicle_plate_read",
                        camera_id=self.camera_id,
                        details={
                            "track_id": det.track_id,
                            "class_name": det.class_name,
                            "plate_text": plate.text or "UNREADABLE",
                            "confidence": round(plate.detection_confidence, 2),
                        },
                        frame=frame,
                        is_night=is_night,
                    )
                    self._maybe_record_clip(alert)
                    px1, py1, px2, py2 = [int(v) for v in plate.bbox]
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    if plate.text:
                        cv2.putText(frame, plate.text, (px1, max(0, py1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # ---- Night-time movement detection ----
        # Reuses the detections already computed above (no duplicate
        # inference) -- only adds the cheap motion-trigger check on top,
        # per the two-stage design. (is_night itself was already
        # computed at the top of this function.)
        if is_night:
            motion, _regions = self.night_pipeline.motion_detector.detect_motion(frame)
            if motion:
                if detections:
                    # Motion + a real person/vehicle confirmed by the
                    # main detector at night -> escalate severity.
                    alert = alert_manager.raise_alert(
                        event_type="night_intrusion",
                        camera_id=self.camera_id,
                        details={
                            "class_name": detections[0].class_name,
                            "count": len(detections),
                        },
                        frame=frame,
                        is_night=is_night,
                    )
                    self._maybe_record_clip(alert)
                else:
                    # Motion but nothing the detector could confirm
                    # (could be an animal, foliage, or something the
                    # low-light conditions are hiding) -> low priority.
                    alert_manager.raise_alert(
                        event_type="night_motion",
                        camera_id=self.camera_id,
                        details={"note": "unconfirmed motion in low light"},
                        frame=frame,
                        is_night=is_night,
                    )
            cv2.putText(frame, "NIGHT MODE", (frame.shape[1] - 160, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        zone_alert_active = any(self.fence.is_inside(d.center) for d in detections)
        frame = self.fence.draw(frame, alert=zone_alert_active)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = (0, 255, 0) if det.class_name == "person" else (255, 150, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} #{det.track_id} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_count = self._frame_count + 1
        self._frame_count = frame_count
        now = time.time()
        fps = 1.0 / (now - self._prev_time) if now > self._prev_time else 0.0
        self._prev_time = now
        cam_stats = {
            "fps": round(fps, 1),
            "frame_count": frame_count,
            "active_tracks": len({d.track_id for d in detections if d.track_id is not None}),
        }
        camera_stats[self.camera_id] = cam_stats

        cv2.putText(frame, f"FPS: {cam_stats['fps']}  Tracks: {cam_stats['active_tracks']}",
                    (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        with camera_frames_lock:
            camera_frames[self.camera_id] = frame.copy()

        self._update_clip_buffers(frame, now)

    def _update_clip_buffers(self, frame, now):
        """Maintain the rolling pre-alert buffer and feed any in-progress
        clip recordings. Called once per processed frame."""
        snapshot = frame.copy()
        self._frame_buffer.append((now, snapshot))
        while self._frame_buffer and now - self._frame_buffer[0][0] > CLIP_PRE_SECONDS:
            self._frame_buffer.popleft()

        still_active = []
        for clip in self._active_clips:
            clip["frames"].append((now, snapshot))
            if now < clip["end_time"]:
                still_active.append(clip)
            else:
                # Post-trigger window elapsed -- encode and save. Offload
                # the actual encoding to the OS thread pool so it doesn't
                # stall the eventlet hub (same pattern as model inference).
                eventlet.spawn(self._finalize_clip, clip)
        self._active_clips = still_active

    def _start_clip_recording(self, alert_id):
        """Begin a clip recording for the given alert: seeds it with the
        already-buffered PRE-alert frames, then keeps collecting frames
        for CLIP_POST_SECONDS more (via _update_clip_buffers above)
        before being encoded and saved."""
        self._active_clips.append({
            "alert_id": alert_id,
            "frames": list(self._frame_buffer),  # copy -- pre-alert context
            "end_time": time.time() + CLIP_POST_SECONDS,
        })

    def _maybe_record_clip(self, alert):
        """Call with the return value of alert_manager.raise_alert() --
        starts a clip recording if this alert's severity warrants one."""
        if alert is not None and alert.severity in CLIP_TRIGGER_SEVERITIES:
            self._start_clip_recording(alert.id)

    def _finalize_clip(self, clip):
        frames = [f for (_ts, f) in clip["frames"]]
        if not frames:
            return
        try:
            os.makedirs(CLIP_DIR, exist_ok=True)
            h, w = frames[0].shape[:2]
            filename = f"{clip['alert_id']}.mp4"
            path = os.path.join(CLIP_DIR, filename)

            def _write():
                # Browsers (Chrome/Edge/Firefox) generally only play H.264
                # ("avc1") video inside an .mp4 container -- OpenCV's
                # default 'mp4v' fourcc produces a technically-valid MP4
                # that VLC etc. can open, but most browsers show a blank
                # player for it. Try H.264 first (Windows opencv-python/
                # opencv-contrib-python wheels bundle an FFmpeg plugin
                # that usually supports it); fall back to mp4v if this
                # particular OpenCV build can't open an avc1 writer.
                for fourcc_str in ("avc1", "H264", "mp4v"):
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    writer = cv2.VideoWriter(path, fourcc, CLIP_FPS, (w, h))
                    if writer.isOpened():
                        break
                    writer.release()
                else:
                    print(f"[WARN] Could not open a working video writer for clip {filename}")
                    return
                for f in frames:
                    writer.write(f)
                writer.release()

            eventlet.tpool.execute(_write)
            event_logger.set_clip_path(clip["alert_id"], filename)
            # Freeze the clip's SHA-256 into the evidence ledger.
            try:
                ledger.record_clip(clip["alert_id"], path)
            except Exception as e:
                alert_manager.ledger_failures += 1
                print(f"[WARN] Could not seal clip {filename} into ledger: {e}")
        except Exception as e:
            print(f"[WARN] Failed to save clip for alert {clip['alert_id']}: {e}")


def mjpeg_generator(camera_id):
    while True:
        with camera_frames_lock:
            frame = camera_frames.get(camera_id)
            frame = frame.copy() if frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)


# ---- Authentication ----
# Everything in this app requires login except the login page itself
# and Flask's built-in static-file endpoint. API/video/snapshot routes
# get a 401 JSON response instead of an HTML redirect when unauthenticated.
_AUTH_EXEMPT_ENDPOINTS = {"login", "static"}


@app.before_request
def _require_login():
    if request.endpoint in _AUTH_EXEMPT_ENDPOINTS or request.endpoint is None:
        return
    if session.get("user"):
        return
    if (request.path.startswith("/api/") or request.path.startswith("/video_feed")
            or request.path.startswith("/snapshots") or request.path.startswith("/clips")):
        return jsonify({"error": "authentication required"}), 401
    return redirect(url_for("login", next=request.path))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if verify_user(username, password):
            session["user"] = username
            session.permanent = True
            return redirect(request.args.get("next") or url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error, no_users=not has_any_users())


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/")
def index():
    return render_template("dashboard.html", current_user=session.get("user"))


@app.route("/history")
def history():
    return render_template("history.html", current_user=session.get("user"))


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", current_user=session.get("user"))


@app.route("/upload")
def upload_page():
    return render_template("upload.html", current_user=session.get("user"))


def _default_camera_id():
    """First registered camera -- used by the legacy no-argument routes
    below so a single-camera setup (the common case) keeps working
    exactly as before without needing a camera_id in the URL."""
    return next(iter(camera_registry), None)


@app.route("/api/cameras")
def api_cameras():
    """List all configured cameras and their live status -- powers the
    camera selector in the dashboard UI."""
    return jsonify([
        {"camera_id": cid, **info}
        for cid, info in camera_registry.items()
    ])


@app.route("/video_feed")
def video_feed_default():
    cam = _default_camera_id()
    return Response(mjpeg_generator(cam), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    return Response(mjpeg_generator(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/alerts")
def api_alerts():
    return jsonify(alert_manager.get_recent(limit=100))


@app.route("/api/stats")
def api_stats_default():
    cam = _default_camera_id()
    return jsonify(camera_stats.get(cam, {"fps": 0.0, "frame_count": 0, "active_tracks": 0}))


@app.route("/api/stats/<camera_id>")
def api_stats(camera_id):
    return jsonify(camera_stats.get(camera_id, {"fps": 0.0, "frame_count": 0, "active_tracks": 0}))


@app.route("/api/alerts/<alert_id>/ack", methods=["POST"])
def ack_alert(alert_id):
    """Acknowledge an alert. The operator's identity + time are written to the
    evidence ledger (chain of custody) -- not just a flag in memory."""
    in_memory = alert_manager.acknowledge(alert_id)
    if not in_memory and not ledger.custody(alert_id):
        return jsonify({"success": False, "error": "unknown alert"}), 404
    block = ledger.record_ack(alert_id, session.get("user"), ip=request.remote_addr)
    return jsonify({"success": True, "ledger_block": block["idx"], "ledger_hash": block["hash"]})


@app.route("/api/alerts/<alert_id>/action", methods=["POST"])
def action_alert(alert_id):
    """Record the response taken on an alert. Body: {"action": "<CODE>", "note": "..."}.
    Codes: DISPATCHED, ESCALATED, MONITORING, FALSE_ALARM, RESOLVED, OTHER."""
    if not ledger.custody(alert_id):
        return jsonify({"success": False, "error": "alert is not in the evidence ledger"}), 404
    body = request.get_json(silent=True) or {}
    try:
        block = ledger.record_action(alert_id, session.get("user"), body.get("action"),
                                     body.get("note", ""), ip=request.remote_addr)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    return jsonify({"success": True, "ledger_block": block["idx"], "ledger_hash": block["hash"]})


# ---------------- Evidence ledger / chain of custody ----------------
@app.route("/ledger")
def ledger_page():
    return render_template("custody.html", current_user=session.get("user"),
                           action_codes=sorted(ACTION_CODES))


@app.route("/api/ledger/status")
def api_ledger_status():
    st = ledger.stats()
    st["seal_failures"] = alert_manager.ledger_failures
    return jsonify(st)


@app.route("/api/ledger/alerts")
def api_ledger_alerts():
    return jsonify(ledger.list_sealed_alerts(limit=request.args.get("limit", default=100, type=int)))


@app.route("/api/ledger/verify")
def api_ledger_verify():
    """Re-derive every hash + signature in the chain and cross-check anchors."""
    return jsonify(ledger.verify_chain())


@app.route("/api/ledger/custody/<alert_id>")
def api_ledger_custody(alert_id):
    blocks = ledger.custody(alert_id)
    if not blocks:
        return jsonify({"error": "no ledger entries for this alert"}), 404
    return jsonify({"alert_id": alert_id, "blocks": blocks,
                    "evidence": ledger.verify_alert_evidence(alert_id)})


@app.route("/api/ledger/pubkey")
def api_ledger_pubkey():
    return Response(ledger.public_key_pem, mimetype="application/x-pem-file")


@app.route("/api/ledger/anchor", methods=["POST"])
def api_ledger_anchor():
    """Force an anchor checkpoint now (also pushed to webhooks)."""
    return jsonify(ledger.force_anchor())


@app.route("/api/ledger/bundle/<alert_id>")
def api_ledger_bundle(alert_id):
    """Court evidence bundle (ZIP). Logs an EXPORTED block attributed to the caller."""
    try:
        data = ledger.build_bundle(alert_id, session.get("user"), ip=request.remote_addr,
                                   verifier_script_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "verify_bundle.py"))
    except KeyError:
        return jsonify({"error": "no ledger entries for this alert"}), 404
    resp = Response(data, mimetype="application/zip")
    resp.headers["Content-Disposition"] = f"attachment; filename=evidence_{alert_id}.zip"
    return resp


@app.route("/api/events")
def api_events():
    """
    Persistent incident history (survives restarts), backed by SQLite --
    unlike /api/alerts which only reflects the current run's in-memory
    buffer. Supports optional query params: limit, event_type, severity,
    camera_id, start_ts, end_ts (unix timestamps), min_score (0-100
    threat score threshold).
    Example: /api/events?event_type=weapon_detected&min_score=60&limit=20
    """
    limit = request.args.get("limit", default=100, type=int)
    event_type = request.args.get("event_type")
    severity = request.args.get("severity")
    camera_id = request.args.get("camera_id")
    start_ts = request.args.get("start_ts", type=float)
    end_ts = request.args.get("end_ts", type=float)
    min_score = request.args.get("min_score", type=int)
    return jsonify(event_logger.query(
        limit=limit, event_type=event_type, severity=severity,
        camera_id=camera_id, start_ts=start_ts, end_ts=end_ts, min_score=min_score,
    ))


@app.route("/api/analytics/summary")
def api_analytics_summary():
    return jsonify(event_logger.summary())


@app.route("/api/analytics/by_type")
def api_analytics_by_type():
    days = request.args.get("days", type=int)
    camera_id = request.args.get("camera_id")
    start_ts = (time.time() - days * 86400) if days else None
    return jsonify(event_logger.stats_by_type(start_ts=start_ts, camera_id=camera_id))


@app.route("/api/analytics/by_camera")
def api_analytics_by_camera():
    days = request.args.get("days", type=int)
    start_ts = (time.time() - days * 86400) if days else None
    return jsonify(event_logger.stats_by_camera(start_ts=start_ts))


@app.route("/api/analytics/by_hour")
def api_analytics_by_hour():
    days = request.args.get("days", type=int)
    camera_id = request.args.get("camera_id")
    start_ts = (time.time() - days * 86400) if days else None
    return jsonify(event_logger.stats_by_hour(start_ts=start_ts, camera_id=camera_id))


@app.route("/api/analytics/timeseries")
def api_analytics_timeseries():
    bucket = request.args.get("bucket", "day")
    days = request.args.get("days", default=7, type=int)
    camera_id = request.args.get("camera_id")
    return jsonify(event_logger.timeseries(bucket=bucket, days=days, camera_id=camera_id))


@app.route("/api/events/export")
def api_events_export():
    """
    Export incident history for handoff to an external system --
    supports the same filters as /api/events, plus ?format=csv|json
    (default json). Example:
        /api/events/export?format=csv&event_type=weapon_detected&limit=500
    """
    fmt = request.args.get("format", "json").lower()
    limit = request.args.get("limit", default=1000, type=int)
    event_type = request.args.get("event_type")
    severity = request.args.get("severity")
    camera_id = request.args.get("camera_id")
    start_ts = request.args.get("start_ts", type=float)
    end_ts = request.args.get("end_ts", type=float)
    min_score = request.args.get("min_score", type=int)

    events = event_logger.query(
        limit=limit, event_type=event_type, severity=severity,
        camera_id=camera_id, start_ts=start_ts, end_ts=end_ts, min_score=min_score,
    )

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "date", "time", "event_type", "severity", "threat_score",
                          "camera_id", "description", "details"])
        for ev in events:
            writer.writerow([
                ev["id"], ev["date"], ev["time"], ev["event_type"], ev["severity"], ev["threat_score"],
                ev["camera_id"], ev["description"], json.dumps(ev["details"]),
            ])
        resp = Response(buf.getvalue(), mimetype="text/csv")
        resp.headers["Content-Disposition"] = "attachment; filename=ibvap_events.csv"
        return resp

    resp = jsonify(events)
    resp.headers["Content-Disposition"] = "attachment; filename=ibvap_events.json"
    return resp


@app.route("/api/webhooks", methods=["GET"])
def api_webhooks_list():
    """List registered C2/SIEM webhook endpoints -- every alert is POSTed
    (as JSON) to each of these URLs in real time. See core/webhooks.py."""
    return jsonify(webhook_manager.list())


@app.route("/api/webhooks", methods=["POST"])
def api_webhooks_add():
    """Register a new webhook. Body: {"url": "https://your-c2-system/ingest"}"""
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    if not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "url must start with http:// or https://"}), 400
    entry = webhook_manager.add(url)
    return jsonify(entry), 201


@app.route("/api/webhooks/<webhook_id>", methods=["DELETE"])
def api_webhooks_delete(webhook_id):
    ok = webhook_manager.remove(webhook_id)
    return jsonify({"success": ok})


@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    """
    Accepts an uploaded video file, runs it through the EXACT SAME
    detection/alert pipeline as a live camera (person/vehicle/weapon
    detection with tracking boxes, zone rules, threat scoring, alerting,
    logging) -- just once through the file instead of looping forever.
    Returns a camera_id the frontend uses with the existing
    /video_feed/<id> and /api/camera_status/<id> endpoints to watch it
    process, exactly like a live camera tab.
    """
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded (expected field 'video')"}), 400
    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '.{ext}'. "
                                  f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    safe_name = secure_filename(file.filename) or "video"
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")
    file.save(saved_path)

    camera_id = f"UPLOAD-{job_id}"
    camera_registry[camera_id] = {
        "source": saved_path, "status": "starting", "error": None,
        "is_upload": True, "original_filename": file.filename,
        "total_frames": 0,
    }
    processor = VideoProcessor(source=saved_path, camera_id=camera_id, loop=False, frame_skip=3)
    socketio.start_background_task(processor.run)

    return jsonify({
        "camera_id": camera_id,
        "video_feed_url": f"/video_feed/{camera_id}",
        "stats_url": f"/api/camera_status/{camera_id}",
    }), 201


@app.route("/api/camera_status/<camera_id>")
def api_camera_status(camera_id):
    """Combined stats + registry status for one camera/upload-job --
    powers the progress bar and 'processing complete' state on the
    upload page (and could be reused for a richer live-camera view)."""
    stats = camera_stats.get(camera_id, {"fps": 0.0, "frame_count": 0, "active_tracks": 0})
    reg = camera_registry.get(camera_id, {})
    return jsonify({**stats, **reg, "camera_id": camera_id})


@app.route("/api/uploads")
def api_uploads():
    """List upload-analysis jobs (most recent first) -- lets the upload
    page show past jobs even after a page refresh."""
    jobs = [
        {"camera_id": cid, **info}
        for cid, info in camera_registry.items()
        if info.get("is_upload")
    ]
    return jsonify(list(reversed(jobs)))


def _ledger_log_view(filename, artifact, base_dir):
    """Chain of custody: record WHO opened WHICH evidence file (with its hash at that moment)."""
    try:
        name = os.path.basename(filename)
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            ledger.record_view(os.path.splitext(name)[0], session.get("user"), artifact, path,
                               ip=request.remote_addr)
    except Exception as e:
        print(f"[WARN] could not log evidence view: {e}")


@app.route("/snapshots/<path:filename>")
def snapshot(filename):
    _ledger_log_view(filename, "snapshot", event_logger.snapshot_dir)
    return send_from_directory(event_logger.snapshot_dir, filename)


@app.route("/clips/<path:filename>")
def clip(filename):
    _ledger_log_view(filename, "clip", CLIP_DIR)
    return send_from_directory(CLIP_DIR, filename)


@socketio.on("connect")
def on_connect():
    if not session.get("user"):
        return False  # reject the connection -- not logged in
    # When an admin dashboard connects (logs in), immediately send them
    # the recent alert history so nothing is missed.
    socketio.emit("alert_backlog", alert_manager.get_recent(limit=50))


def load_camera_configs(args):
    """
    Multi-camera setup: create a cameras.json file (see
    cameras.example.json) listing every camera, e.g.:
        [
          {"camera_id": "BOP-01", "source": 0},
          {"camera_id": "BOP-02", "source": "rtsp://192.168.1.50/stream1"}
        ]
    If cameras.json isn't present, falls back to the single
    --source/--camera_id CLI args (original single-camera behaviour --
    nothing breaks for existing single-camera setups).
    """
    cfg_path = "cameras.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cams = json.load(f)
        if not cams:
            raise ValueError("cameras.json exists but is empty -- add at least one camera.")
        return cams
    source = int(args.source) if str(args.source).isdigit() else args.source
    return [{"camera_id": args.camera_id, "source": source}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="test_videos/synthetic_test.mp4",
                         help="Video file path, webcam index (0), or RTSP URL "
                              "(ignored if cameras.json is present -- see load_camera_configs)")
    parser.add_argument("--camera_id", default="BOP-01")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    cameras = load_camera_configs(args)

    # NOTE ON RESOURCE USE: each camera loads its OWN full set of AI
    # models (person/vehicle, weapon, pose, ANPR) -- there is no shared
    # model pool. On CPU-only hardware this adds up fast: expect FPS to
    # drop noticeably as you add more cameras. For a handful of cameras
    # on a laptop, consider raising VideoProcessor.heavy_model_stride
    # (see __init__) to keep things responsive. A real multi-camera
    # deployment would want either a GPU or a shared inference pool.
    for cam in cameras:
        camera_registry[cam["camera_id"]] = {
            "source": str(cam["source"]), "status": "starting", "error": None, "is_upload": False,
        }

    processors = []
    for cam in cameras:
        source = int(cam["source"]) if str(cam["source"]).isdigit() else cam["source"]
        processor = VideoProcessor(source=source, camera_id=cam["camera_id"])
        processors.append(processor)
        # IMPORTANT: run the processing loop as an eventlet greenthread (not a
        # raw OS thread) so that socketio.emit() calls made while raising
        # alerts execute in the correct eventlet context and reliably reach
        # connected browsers. The actual blocking calls inside the loop
        # (camera reads, model inference) are individually offloaded to a
        # real OS thread pool via eventlet.tpool.execute() so they don't
        # stall the eventlet hub -- see run()/_process_frame() below.
        socketio.start_background_task(processor.run)

    cam_ids = [c["camera_id"] for c in cameras]
    print(f"Dashboard running at http://localhost:{args.port} -- camera(s): {', '.join(cam_ids)}")
    socketio.run(app, host="0.0.0.0", port=args.port)
