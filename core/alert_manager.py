"""
alert_manager.py
------------------
Central place where every module (virtual fence, gun detector, suspicious
activity, night motion, etc.) sends its events. Responsible for:
  1. Assigning priority tiers (Critical/High/Medium/Low)
  2. Keeping an in-memory alert log (swap for a real DB in production)
  3. Pushing alerts to connected dashboards via SocketIO (real-time,
     point #1 from the requirements: "alert on system where admin is
     logged in")
  4. Queuing alerts even if no admin is currently connected, so nothing
     is missed when someone logs back in.

This is intentionally decoupled from the video/detection code -- any
detection module just calls alert_manager.raise_alert(...).
"""

import os
import time
import uuid
from collections import deque
from threading import Lock

from core.threat_score import compute_base_score

# Priority ordering: lower number = more urgent
PRIORITY = {
    # Weapon detection - highest severity, a missed detection is far
    # costlier than a false alarm, so every weapon class is CRITICAL.
    "gun_detected": ("CRITICAL", 0),
    "weapon_detected": ("CRITICAL", 0),

    # Zone / posture / trajectory based suspicious behaviour
    "intrusion": ("HIGH", 1),
    "crawling": ("HIGH", 1),
    "climbing": ("HIGH", 1),
    "sudden_approach": ("HIGH", 1),
    "erratic_movement": ("HIGH", 1),

    "loitering": ("MEDIUM", 2),
    "unrecognized_face": ("MEDIUM", 2),
    "face_detected": ("MEDIUM", 2),
    "vehicle_plate_read": ("MEDIUM", 2),
    "unrecognized_vehicle": ("MEDIUM", 2),

    "night_motion": ("LOW", 3),
    "night_intrusion": ("HIGH", 1),  # confirmed person/vehicle at night in the zone -> escalated

    # Demo / operator visibility: any tracked person appears in the dashboard.
    "person_detected": ("MEDIUM", 2),
    "vehicle_detected": ("MEDIUM", 2),
}


# ---- Evidence-ledger policy (see core/ledger.py) ----
# Every CRITICAL/HIGH alert is sealed into the tamper-evident ledger, plus
# the identity-related MEDIUM types below. Chatty demo alerts
# (person_detected / vehicle_detected / night_motion) are NOT sealed -- they
# would bloat the chain without adding evidentiary value.
LEDGER_SEVERITIES = {"CRITICAL", "HIGH"}
# type -> minimum seconds between ledger entries for the same (type, camera, track)
LEDGER_EXTRA_TYPES = {
    "face_detected": 30,
    "unrecognized_face": 30,
    "known_person_seen": 60,   # whitelist face-match, raised via log_silent()
    "vehicle_plate_read": 30,
}


class Alert:
    def __init__(self, event_type, camera_id, details):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.severity, self.priority_rank = PRIORITY.get(event_type, ("LOW", 3))
        self.camera_id = camera_id
        self.details = details
        self.timestamp = time.time()
        self.acknowledged = False
        self.threat_score = 0  # set by AlertManager.raise_alert() -- see core/threat_score.py

    def to_dict(self):
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity,
            "camera_id": self.camera_id,
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "threat_score": self.threat_score,
        }


class AlertManager:
    """
    In-memory alert bus. In production this would publish to Redis/MQTT
    so multiple backend workers and multiple dashboard instances can
    all subscribe -- but the interface (raise_alert / get_recent /
    the socketio push) stays the same.
    """

    def __init__(self, socketio=None, max_history=500, event_logger=None, webhook_manager=None, ledger=None):
        self.socketio = socketio  # Flask-SocketIO instance, set by app.py
        self.event_logger = event_logger  # core.event_log.EventLogger, set by app.py
        self.webhook_manager = webhook_manager  # core.webhooks.WebhookManager, set by app.py
        self.ledger = ledger  # core.ledger.EvidenceLedger, set by app.py (tamper-evident chain of custody)
        self.ledger_failures = 0  # surfaced on /api/ledger/status -- a silent gap in the chain matters
        self._ledger_last = {}
        self._history = deque(maxlen=max_history)
        self._lock = Lock()
        # Simple cooldown so the same event type from the same camera
        # doesn't spam the dashboard every frame.
        self._last_fired = {}
        self._cooldown_seconds = 5
        # Rolling per-track alert timestamps, used ONLY to compute the
        # threat-score "repeat behaviour" bonus below -- a track that
        # keeps triggering alerts in a short window (e.g. repeated
        # fence intrusions, or lingering after a first sighting) is
        # more concerning than a single one-off event, even at the
        # same nominal severity.
        self._track_history = {}
        self._REPEAT_WINDOW_SECONDS = 300  # 5 minutes
        self._REPEAT_BONUS_PER_HIT = 3
        self._REPEAT_BONUS_MAX = 15

    def _repeat_bonus(self, track_id):
        if track_id is None:
            return 0
        now = time.time()
        times = self._track_history.setdefault(track_id, [])
        times[:] = [t for t in times if now - t <= self._REPEAT_WINDOW_SECONDS]
        times.append(now)
        repeats_beyond_first = max(0, len(times) - 1)
        return min(self._REPEAT_BONUS_MAX, repeats_beyond_first * self._REPEAT_BONUS_PER_HIT)

    def _seal(self, alert, snapshot_file):
        """Write this alert (and its snapshot hash) into the evidence ledger.
        Must NEVER raise into the detection pipeline -- but failures are counted
        and exposed so a gap in the chain is visible, not silent."""
        if self.ledger is None:
            return
        min_gap = LEDGER_EXTRA_TYPES.get(alert.event_type)
        if alert.severity not in LEDGER_SEVERITIES and min_gap is None:
            return
        if min_gap is not None and alert.severity not in LEDGER_SEVERITIES:
            key = (alert.event_type, alert.camera_id, alert.details.get("track_id"))
            if alert.timestamp - self._ledger_last.get(key, 0) < min_gap:
                return
            self._ledger_last[key] = alert.timestamp
        try:
            snap_path = None
            if snapshot_file and self.event_logger is not None:
                snap_path = os.path.join(self.event_logger.snapshot_dir, snapshot_file)
            self.ledger.record_alert(alert.to_dict(), snapshot_path=snap_path)
        except Exception as e:
            self.ledger_failures += 1
            print(f"[ALERT][LEDGER ERROR] could not seal alert {alert.id}: {e}")

    def raise_alert(self, event_type, camera_id, details=None, frame=None, is_night=False):
        details = details or {}
        cooldown_key = (event_type, camera_id, details.get("track_id"))
        now = time.time()

        with self._lock:
            last_time = self._last_fired.get(cooldown_key, 0)
            if now - last_time < self._cooldown_seconds:
                return None  # suppress duplicate spam
            self._last_fired[cooldown_key] = now

            alert = Alert(event_type, camera_id, details)
            base_score = compute_base_score(event_type, alert.severity, details, is_night=is_night)
            alert.threat_score = min(100, base_score + self._repeat_bonus(details.get("track_id")))
            self._history.append(alert)

        # Push to every connected admin dashboard in real time.
        if self.socketio is not None:
            self.socketio.emit("new_alert", alert.to_dict())

        # Persist to disk (SQLite row + JPEG snapshot) so it survives a
        # restart and shows up in incident history/review.
        snapshot_file = None
        if self.event_logger is not None:
            snapshot_file = self.event_logger.log(alert.to_dict(), frame=frame)

        # Seal into the tamper-evident evidence ledger (hash of alert + snapshot).
        self._seal(alert, snapshot_file)

        # Push to any registered external systems (C2 platform, SIEM,
        # Slack/Teams bridge, ticketing system, etc.) -- see core/webhooks.py.
        if self.webhook_manager is not None:
            self.webhook_manager.notify(alert.to_dict())

        print(f"[ALERT][{alert.severity}] {event_type} @ camera={camera_id} :: {details}")
        return alert

    def log_silent(self, event_type, camera_id, details=None, frame=None):
        """
        Like raise_alert(), but does NOT push a live 'new_alert' to
        connected dashboards (no popup, no sound). Use this for events
        that should be RECORDED for the incident-history log but
        shouldn't interrupt/alarm anyone -- e.g. a whitelisted/known
        person being seen on camera. Still respects the same cooldown
        so it doesn't spam the log every frame while they're in view.
        """
        details = details or {}
        cooldown_key = (event_type, camera_id, details.get("track_id"))
        now = time.time()

        with self._lock:
            last_time = self._last_fired.get(cooldown_key, 0)
            if now - last_time < self._cooldown_seconds:
                return None
            self._last_fired[cooldown_key] = now
            alert = Alert(event_type, camera_id, details)

        snapshot_file = None
        if self.event_logger is not None:
            snapshot_file = self.event_logger.log(alert.to_dict(), frame=frame)

        self._seal(alert, snapshot_file)

        return alert

    def get_recent(self, limit=50):
        with self._lock:
            items = list(self._history)[-limit:]
        # Most recent first, critical first within same recency window
        items.sort(key=lambda a: (-a.timestamp,))
        return [a.to_dict() for a in items]

    def acknowledge(self, alert_id):
        with self._lock:
            for a in self._history:
                if a.id == alert_id:
                    a.acknowledged = True
                    return True
        return False
