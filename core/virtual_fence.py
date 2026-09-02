"""
virtual_fence.py
-----------------
Defines a "virtual fence" as a polygon drawn over the camera frame
(e.g. the actual border line visible in the CCTV view). Any tracked
object whose position crosses into the restricted polygon triggers
an intrusion event.

Also does simple loitering detection: if the same track_id stays
inside a "watch zone" longer than a threshold, flag it -- useful for
the Suspicious Activity module (Part 3) but included here since it
reuses the same zone-checking logic.
"""

import time
import cv2
import numpy as np


class VirtualFence:
    def __init__(self, name, polygon_points, loiter_seconds=15):
        """
        polygon_points: list of (x, y) tuples defining the restricted zone,
                         in pixel coordinates of the camera frame.
        loiter_seconds: how long an object can stay in the zone before
                         it's flagged as "loitering" (not just "intrusion").
        """
        self.name = name
        self.polygon = np.array(polygon_points, dtype=np.int32)
        self.loiter_seconds = loiter_seconds

        # Track how long each track_id has been continuously inside the zone.
        # {track_id: first_seen_inside_timestamp}
        self._inside_since = {}
        # track_ids that have already fired an "intrusion" event this session,
        # so we don't spam an alert every single frame.
        self._already_alerted_intrusion = set()
        self._already_alerted_loiter = set()

    def is_inside(self, point):
        """point = (x, y). Returns True if point is inside the polygon."""
        result = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    def evaluate(self, detections):
        """
        Given a list of Detection objects (from detector.py), returns a
        list of event dicts for anything new this frame:
          {"type": "intrusion" | "loitering", "track_id": ..., "class_name": ..., "zone": self.name}

        Also cleans up state for track_ids no longer present.
        """
        events = []
        now = time.time()
        seen_ids_this_frame = set()

        for det in detections:
            if det.track_id is None:
                continue  # can't track loitering/dedup without a persistent ID

            inside = self.is_inside(det.center)
            seen_ids_this_frame.add(det.track_id)

            if inside:
                if det.track_id not in self._inside_since:
                    self._inside_since[det.track_id] = now

                # Fire "intrusion" once per track_id per entry into the zone.
                if det.track_id not in self._already_alerted_intrusion:
                    self._already_alerted_intrusion.add(det.track_id)
                    events.append(
                        {
                            "type": "intrusion",
                            "track_id": det.track_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                            "zone": self.name,
                            "bbox": det.bbox,
                        }
                    )

                # Fire "loitering" once, if they've been inside too long.
                duration = now - self._inside_since[det.track_id]
                if duration >= self.loiter_seconds and det.track_id not in self._already_alerted_loiter:
                    self._already_alerted_loiter.add(det.track_id)
                    events.append(
                        {
                            "type": "loitering",
                            "track_id": det.track_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                            "zone": self.name,
                            "duration_seconds": round(duration, 1),
                            "bbox": det.bbox,
                        }
                    )
            else:
                # Left the zone (or never entered) -> reset their "inside since" clock
                # so a future re-entry counts as a fresh intrusion.
                self._inside_since.pop(det.track_id, None)
                self._already_alerted_intrusion.discard(det.track_id)
                self._already_alerted_loiter.discard(det.track_id)

        # Clean up bookkeeping for IDs that vanished (left frame / lost track)
        stale_ids = set(self._inside_since.keys()) - seen_ids_this_frame
        for sid in stale_ids:
            self._inside_since.pop(sid, None)
            self._already_alerted_intrusion.discard(sid)
            self._already_alerted_loiter.discard(sid)

        return events

    def draw(self, frame, color_normal=(0, 255, 255), color_alert=(0, 0, 255), alert=False):
        """Draw the fence polygon on a frame for visualization."""
        color = color_alert if alert else color_normal
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=2)
        # Label
        x, y = self.polygon[0]
        cv2.putText(frame, self.name, (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
