"""
suspicious_activity.py
------------------------
Rule-based (not black-box) suspicious behaviour detection, combining:

  1. POSTURE rules (from pose keypoints, see pose_detector.py):
       - crawling   -> body is horizontal instead of upright
       - climbing   -> both wrists raised well above the shoulders

  2. TRAJECTORY rules (from tracked positions over time, see detector.py):
       - sudden_approach -> a track is moving fast, directly toward the
         fence zone (e.g. someone suddenly sprinting at the border)
       - erratic_movement -> a track keeps sharply changing direction
         (a person "checking" for gaps / probing the fence line)

Kept deliberately rule-based and explainable (not a trained anomaly
model) because:
  - there's no time/data to train a reliable custom anomaly detector
    for a hackathon timeline
  - judges and real security operators can understand *why* an alert
    fired, which matters a lot more here than in most ML applications
  - thresholds are easy to tune live during a demo
"""

import math
import time
from collections import deque

from core.pose_detector import PoseDetector


def _iou(box_a, box_b):
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


class SuspiciousActivityDetector:
    def __init__(self, pose_model_path="models/yolov8n-pose.pt",
                 fence=None,
                 history_seconds=3.0,
                 speed_threshold_px_per_sec=220,
                 direction_change_threshold_deg=70,
                 min_track_points=5):
        self.pose_detector = PoseDetector(model_path=pose_model_path)
        self.fence = fence  # optional VirtualFence, used for "moving toward fence" check

        self.history_seconds = history_seconds
        self.speed_threshold = speed_threshold_px_per_sec
        self.direction_change_threshold = direction_change_threshold_deg
        self.min_track_points = min_track_points

        # track_id -> deque[(timestamp, (x, y))]
        self._position_history = {}
        # avoid re-firing the same alert every frame while condition persists
        self._already_alerted = {}  # (track_id, activity_type) -> last_fired_time
        self._realert_cooldown = 6.0

    # ---------------- Posture rules ----------------

    @staticmethod
    def _is_crawling(pose):
        x1, y1, x2, y2 = pose.bbox
        width, height = x2 - x1, y2 - y1
        if height <= 0:
            return False
        # A standing/walking adult is taller than wide. A crawling/prone
        # person's bounding box flattens out (wide relative to tall).
        aspect_ratio = width / height
        return aspect_ratio > 1.4

    @staticmethod
    def _is_climbing(pose):
        l_wrist, r_wrist = pose.kp("left_wrist"), pose.kp("right_wrist")
        l_shoulder, r_shoulder = pose.kp("left_shoulder"), pose.kp("right_shoulder")
        if not (l_wrist and r_wrist and l_shoulder and r_shoulder):
            return False
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        # Image y-axis increases downward, so "above" means a SMALLER y value.
        wrists_raised = (l_wrist[1] < shoulder_y - 15) and (r_wrist[1] < shoulder_y - 15)
        return wrists_raised

    # ---------------- Trajectory rules ----------------

    def _update_history(self, track_id, position, now):
        if track_id not in self._position_history:
            self._position_history[track_id] = deque()
        hist = self._position_history[track_id]
        hist.append((now, position))
        while hist and now - hist[0][0] > self.history_seconds:
            hist.popleft()

    def _check_sudden_approach(self, track_id):
        hist = self._position_history.get(track_id)
        if not hist or len(hist) < self.min_track_points:
            return False, 0.0
        (t0, p0), (t1, p1) = hist[0], hist[-1]
        dt = t1 - t0
        if dt <= 0:
            return False, 0.0
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        speed = dist / dt
        if speed < self.speed_threshold:
            return False, speed

        if self.fence is not None:
            fence_center = self.fence.polygon.mean(axis=0)
            moving_toward_fence = (
                math.hypot(fence_center[0] - p1[0], fence_center[1] - p1[1])
                < math.hypot(fence_center[0] - p0[0], fence_center[1] - p0[1])
            )
            return moving_toward_fence, speed
        return True, speed

    def _check_erratic_movement(self, track_id):
        hist = self._position_history.get(track_id)
        if not hist or len(hist) < self.min_track_points + 2:
            return False
        points = [p for _, p in hist]
        angles = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            if dx == 0 and dy == 0:
                continue
            angles.append(math.degrees(math.atan2(dy, dx)))
        if len(angles) < 3:
            return False
        big_changes = 0
        for i in range(1, len(angles)):
            diff = abs(angles[i] - angles[i - 1])
            diff = min(diff, 360 - diff)
            if diff > self.direction_change_threshold:
                big_changes += 1
        return big_changes >= 3

    # ---------------- Main evaluate() ----------------

    def evaluate(self, frame, person_detections, poses=None):
        """
        frame: current BGR frame
        person_detections: list[Detection] (class_name == 'person') from
                            core.detector.PersonVehicleDetector, WITH track_ids
        poses: optional pre-computed list[PoseResult] (from self.pose_detector.detect(frame)
               or elsewhere) -- pass this in if the caller already ran pose
               estimation this frame (e.g. for face detection) to avoid a
               duplicate, redundant inference pass.

        Returns list of event dicts:
          {"type": "crawling"|"climbing"|"sudden_approach"|"erratic_movement",
           "track_id":, "bbox":, "extra": {...}}
        """
        now = time.time()
        events = []

        persons = [d for d in person_detections if d.class_name == "person"]
        for det in persons:
            if det.track_id is not None:
                self._update_history(det.track_id, det.center, now)

        # Posture rules need pose keypoints -> run pose model (unless the
        # caller already supplied poses this frame), then match each pose
        # result to a tracked person via IoU so we get a track_id.
        poses = poses if poses is not None else self.pose_detector.detect(frame)
        for pose in poses:
            best_det, best_iou = None, 0.0
            for det in persons:
                score = _iou(pose.bbox, det.bbox)
                if score > best_iou:
                    best_iou, best_det = score, det
            if best_det is None or best_iou < 0.3 or best_det.track_id is None:
                continue
            track_id = best_det.track_id

            if self._is_crawling(pose):
                events.extend(self._maybe_fire(track_id, "crawling", best_det.bbox, now, {}))
            if self._is_climbing(pose):
                events.extend(self._maybe_fire(track_id, "climbing", best_det.bbox, now, {}))

        # Trajectory rules apply to every tracked person, pose or not.
        for det in persons:
            if det.track_id is None:
                continue
            approaching, speed = self._check_sudden_approach(det.track_id)
            if approaching:
                events.extend(self._maybe_fire(
                    det.track_id, "sudden_approach", det.bbox, now,
                    {"speed_px_per_sec": round(speed, 1)}
                ))
            if self._check_erratic_movement(det.track_id):
                events.extend(self._maybe_fire(
                    det.track_id, "erratic_movement", det.bbox, now, {}
                ))

        # Clean up history for tracks no longer present.
        current_ids = {d.track_id for d in persons if d.track_id is not None}
        stale = set(self._position_history.keys()) - current_ids
        for sid in stale:
            self._position_history.pop(sid, None)

        return events

    def _maybe_fire(self, track_id, activity_type, bbox, now, extra):
        key = (track_id, activity_type)
        last = self._already_alerted.get(key, 0)
        if now - last < self._realert_cooldown:
            return []
        self._already_alerted[key] = now
        return [{
            "type": activity_type,
            "track_id": track_id,
            "bbox": bbox,
            "extra": extra,
        }]
