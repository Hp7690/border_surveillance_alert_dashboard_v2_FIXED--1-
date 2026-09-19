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
import threading
import time

import cv2
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO

from core.detector import PersonVehicleDetector
from core.virtual_fence import VirtualFence
from core.alert_manager import AlertManager
from core.weapon_detector import WeaponDetector
from core.suspicious_activity import SuspiciousActivityDetector
from core.night_detector import NightSurveillancePipeline, is_night_frame
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.anpr import ANPR

app = Flask(__name__)
app.config["SECRET_KEY"] = "sih-border-surveillance-demo"
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

alert_manager = AlertManager(socketio=socketio)

# Shared state between the processing thread and the Flask routes.
latest_frame_lock = threading.Lock()
latest_frame = None
stats = {"fps": 0.0, "frame_count": 0, "active_tracks": 0}


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
    def __init__(self, source, camera_id="BOP-01"):
        super().__init__(daemon=True)
        self.source = source
        self.camera_id = camera_id
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
        # track_id -> whitelisted person's name, once a face on that
        # track has been confidently matched. Cached for the rest of
        # the track's lifetime so we don't need to re-recognize every
        # heavy frame, and so non-heavy frames (which still raise
        # person_detected every frame) also get suppressed immediately.
        self._known_tracks = {}
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

    def stop(self):
        self._stop_flag = True

    def run(self):
        global latest_frame
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {self.source}")
            return

        while not self._stop_flag:
            ret, frame = eventlet.tpool.execute(cap.read)
            if not ret:
                # loop the video for continuous demo purposes
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

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

    def _process_frame(self, frame, run_heavy):
        global latest_frame

        detections = eventlet.tpool.execute(self.detector.detect, frame, use_tracking=True)
        fence_events = self.fence.evaluate(detections)

        # Dashboard demo alert: surface every newly observed person or
        # vehicle (AlertManager cooldown prevents frame-by-frame spam).
        VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")
        for det in detections:
            if det.class_name == "person":
                if det.track_id in self._known_tracks:
                    continue  # whitelisted face -- no alert
                alert_manager.raise_alert(
                    event_type="person_detected",
                    camera_id=self.camera_id,
                    details={
                        "track_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 2),
                    },
                )
            elif det.class_name in VEHICLE_CLASSES:
                alert_manager.raise_alert(
                    event_type="vehicle_detected",
                    camera_id=self.camera_id,
                    details={
                        "track_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 2),
                    },
                )

        # Any event from the fence -> raise via central alert manager
        for ev in fence_events:
            if ev.get("track_id") in self._known_tracks:
                continue  # whitelisted face -- no alert
            event_type = ev["type"]  # "intrusion" or "loitering"
            alert_manager.raise_alert(
                event_type=event_type,
                camera_id=self.camera_id,
                details={
                    "track_id": ev["track_id"],
                    "class_name": ev["class_name"],
                    "confidence": round(ev["confidence"], 2),
                    "zone": ev["zone"],
                    "extra": ev.get("duration_seconds"),
                },
            )

        # ---- Gun / weapon detection (CRITICAL path) ----
        if run_heavy:
            weapons = eventlet.tpool.execute(self.weapon_detector.detect, frame)
            for w in weapons:
                alert_manager.raise_alert(
                    event_type="weapon_detected",
                    camera_id=self.camera_id,
                    details={
                        "class_name": w.class_name,
                        "confidence": round(w.confidence, 2),
                    },
                )
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
                if ev.get("track_id") in self._known_tracks:
                    continue  # whitelisted face -- no alert
                alert_manager.raise_alert(
                    event_type=ev["type"],
                    camera_id=self.camera_id,
                    details={
                        "track_id": ev["track_id"],
                        "class_name": "person",
                        **ev.get("extra", {}),
                    },
                )
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
                    # Whitelisted -- mark this track as known so
                    # person/zone/suspicious alerts stop firing for it,
                    # and skip the face_detected alert itself.
                    if track_id is not None:
                        self._known_tracks[track_id] = name
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} (cleared)", (fx1, max(0, fy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                    alert_manager.raise_alert(
                        event_type="face_detected",
                        camera_id=self.camera_id,
                        details={"class_name": "person"},
                    )

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
                    alert_manager.raise_alert(
                        event_type="vehicle_plate_read",
                        camera_id=self.camera_id,
                        details={
                            "track_id": det.track_id,
                            "class_name": det.class_name,
                            "plate_text": plate.text or "UNREADABLE",
                            "confidence": round(plate.detection_confidence, 2),
                        },
                    )
                    px1, py1, px2, py2 = [int(v) for v in plate.bbox]
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    if plate.text:
                        cv2.putText(frame, plate.text, (px1, max(0, py1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # ---- Night-time movement detection ----
        # Reuses the detections already computed above (no duplicate
        # inference) -- only adds the cheap motion-trigger check +
        # brightness check on top, per the two-stage design.
        night = is_night_frame(frame)
        if night:
            motion, _regions = self.night_pipeline.motion_detector.detect_motion(frame)
            if motion:
                if detections:
                    # Motion + a real person/vehicle confirmed by the
                    # main detector at night -> escalate severity.
                    alert_manager.raise_alert(
                        event_type="night_intrusion",
                        camera_id=self.camera_id,
                        details={
                            "class_name": detections[0].class_name,
                            "count": len(detections),
                        },
                    )
                else:
                    # Motion but nothing the detector could confirm
                    # (could be an animal, foliage, or something the
                    # low-light conditions are hiding) -> low priority.
                    alert_manager.raise_alert(
                        event_type="night_motion",
                        camera_id=self.camera_id,
                        details={"note": "unconfirmed motion in low light"},
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
        stats["fps"] = round(fps, 1)
        stats["frame_count"] = frame_count
        stats["active_tracks"] = len({d.track_id for d in detections if d.track_id is not None})

        cv2.putText(frame, f"FPS: {stats['fps']}  Tracks: {stats['active_tracks']}",
                    (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        with latest_frame_lock:
            latest_frame = frame.copy()


def mjpeg_generator():
    while True:
        with latest_frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/alerts")
def api_alerts():
    return jsonify(alert_manager.get_recent(limit=100))


@app.route("/api/stats")
def api_stats():
    return jsonify(stats)


@app.route("/api/alerts/<alert_id>/ack", methods=["POST"])
def ack_alert(alert_id):
    ok = alert_manager.acknowledge(alert_id)
    return jsonify({"success": ok})


@socketio.on("connect")
def on_connect():
    # When an admin dashboard connects (logs in), immediately send them
    # the recent alert history so nothing is missed.
    socketio.emit("alert_backlog", alert_manager.get_recent(limit=50))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="test_videos/synthetic_test.mp4",
                         help="Video file path, webcam index (0), or RTSP URL")
    parser.add_argument("--camera_id", default="BOP-01")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    processor = VideoProcessor(source=source, camera_id=args.camera_id)
    # IMPORTANT: run the processing loop as an eventlet greenthread (not a
    # raw OS thread) so that socketio.emit() calls made while raising
    # alerts execute in the correct eventlet context and reliably reach
    # connected browsers. The actual blocking calls inside the loop
    # (camera reads, model inference) are individually offloaded to a
    # real OS thread pool via eventlet.tpool.execute() so they don't
    # stall the eventlet hub -- see run()/_process_frame() below.
    socketio.start_background_task(processor.run)

    print(f"Dashboard running at http://localhost:{args.port}")
    socketio.run(app, host="0.0.0.0", port=args.port)
