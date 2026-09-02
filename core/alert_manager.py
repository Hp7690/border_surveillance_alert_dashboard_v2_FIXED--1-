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

import time
import uuid
from collections import deque
from threading import Lock

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


class Alert:
    def __init__(self, event_type, camera_id, details):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.severity, self.priority_rank = PRIORITY.get(event_type, ("LOW", 3))
        self.camera_id = camera_id
        self.details = details
        self.timestamp = time.time()
        self.acknowledged = False

    def to_dict(self):
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity,
            "camera_id": self.camera_id,
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }


class AlertManager:
    """
    In-memory alert bus. In production this would publish to Redis/MQTT
    so multiple backend workers and multiple dashboard instances can
    all subscribe -- but the interface (raise_alert / get_recent /
    the socketio push) stays the same.
    """

    def __init__(self, socketio=None, max_history=500, event_logger=None, webhook_manager=None):
        self.socketio = socketio  # Flask-SocketIO instance, set by app.py
        self.event_logger = event_logger  # core.event_log.EventLogger, set by app.py
        self.webhook_manager = webhook_manager  # core.webhooks.WebhookManager, set by app.py
        self._history = deque(maxlen=max_history)
        self._lock = Lock()
        # Simple cooldown so the same event type from the same camera
        # doesn't spam the dashboard every frame.
        self._last_fired = {}
        self._cooldown_seconds = 5

    def raise_alert(self, event_type, camera_id, details=None, frame=None):
        details = details or {}
        cooldown_key = (event_type, camera_id, details.get("track_id"))
        now = time.time()

        with self._lock:
            last_time = self._last_fired.get(cooldown_key, 0)
            if now - last_time < self._cooldown_seconds:
                return None  # suppress duplicate spam
            self._last_fired[cooldown_key] = now

            alert = Alert(event_type, camera_id, details)
            self._history.append(alert)

        # Push to every connected admin dashboard in real time.
        if self.socketio is not None:
            self.socketio.emit("new_alert", alert.to_dict())

        # Persist to disk (SQLite row + JPEG snapshot) so it survives a
        # restart and shows up in incident history/review.
        if self.event_logger is not None:
            self.event_logger.log(alert.to_dict(), frame=frame)

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

        if self.event_logger is not None:
            self.event_logger.log(alert.to_dict(), frame=frame)

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
