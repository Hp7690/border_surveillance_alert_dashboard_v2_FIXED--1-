"""
event_log.py
-------------
Persistent event logging: every alert gets a row in a SQLite database
(date, time, event type, severity, camera, description/details) AND a
JPEG snapshot of the camera frame at the moment the alert fired.

This is what makes alerts survive a server restart and enables an
"incident review" history view -- the in-memory AlertManager history
(core/alert_manager.py) only covers the CURRENT run and is capped at a
few hundred entries.

Storage layout:
    data/events.db           <- SQLite database
    data/snapshots/<id>.jpg  <- one snapshot image per alert

SQLite + a lock is more than sufficient at this alert volume (a
handful of events per second at most); no separate DB server needed.
"""

import json
import os
import sqlite3
import time
from threading import Lock

import cv2

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "events.db")
SNAPSHOT_DIR = os.path.join(DB_DIR, "snapshots")

# Cap how large a snapshot we persist -- alerts fire often enough
# (weapon/person detections) that full-resolution frames would burn
# disk space fast for little extra review value.
SNAPSHOT_MAX_WIDTH = 640
JPEG_QUALITY = 80


class EventLogger:
    def __init__(self, db_path=DB_PATH, snapshot_dir=SNAPSHOT_DIR):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir
        self._lock = Lock()
        os.makedirs(DB_DIR, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._init_db()

    def _connect(self):
        # check_same_thread=False: we guard all access with self._lock
        # ourselves since this is called from multiple threads/greenlets.
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    snapshot_path TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type)")

    @staticmethod
    def _describe(event_type, details):
        """Human-readable one-line description, e.g. 'Weapon detected: Pistol (92%)'."""
        d = details or {}
        if d.get("name"):
            # Whitelisted/known person sighting -- lead with their name.
            extra = f" (Track #{d['track_id']})" if d.get("track_id") is not None else ""
            return f"Known person on camera: {d['name']}{extra}"
        parts = []
        if d.get("class_name"):
            parts.append(str(d["class_name"]))
        if d.get("track_id") is not None:
            parts.append(f"Track #{d['track_id']}")
        if d.get("zone"):
            parts.append(f"Zone: {d['zone']}")
        if d.get("confidence") is not None:
            parts.append(f"{round(d['confidence'] * 100)}%")
        if d.get("plate_text"):
            parts.append(f"Plate: {d['plate_text']}")
        label = event_type.replace("_", " ").title()
        return f"{label} ({', '.join(parts)})" if parts else label

    def log(self, alert_dict, frame=None):
        """
        alert_dict: the dict produced by Alert.to_dict() (id, event_type,
        severity, camera_id, details, timestamp, ...).
        frame: optional BGR numpy frame (e.g. from OpenCV) to save as a
        snapshot alongside this event. Pass None to skip the snapshot
        (e.g. for events with no meaningful visual, though in practice
        we always have a frame available).
        """
        alert_id = alert_dict["id"]
        ts = alert_dict["timestamp"]
        dt = time.localtime(ts)
        date_str = time.strftime("%Y-%m-%d", dt)
        time_str = time.strftime("%H:%M:%S", dt)
        description = self._describe(alert_dict["event_type"], alert_dict.get("details"))

        snapshot_path = None
        if frame is not None:
            try:
                h, w = frame.shape[:2]
                if w > SNAPSHOT_MAX_WIDTH:
                    scale = SNAPSHOT_MAX_WIDTH / w
                    frame = cv2.resize(frame, (SNAPSHOT_MAX_WIDTH, int(h * scale)))
                filename = f"{alert_id}.jpg"
                full_path = os.path.join(self.snapshot_dir, filename)
                cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                snapshot_path = filename  # store relative -- served via /snapshots/<filename>
            except Exception as e:
                print(f"[EventLogger] Failed to save snapshot: {e}")

        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO events (id, ts, date, time, event_type, severity, camera_id, "
                    "description, details_json, snapshot_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        alert_id, ts, date_str, time_str,
                        alert_dict["event_type"], alert_dict["severity"], alert_dict["camera_id"],
                        description, json.dumps(alert_dict.get("details") or {}), snapshot_path,
                    ),
                )
        except Exception as e:
            print(f"[EventLogger] Failed to write event to DB: {e}")

    def query(self, limit=100, event_type=None, severity=None, camera_id=None,
              start_ts=None, end_ts=None):
        """Returns most-recent-first list of event dicts, with optional filters."""
        clauses, params = [], []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("ts <= ?")
            params.append(end_ts)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (f"SELECT id, ts, date, time, event_type, severity, camera_id, description, "
               f"details_json, snapshot_path FROM events {where} ORDER BY ts DESC LIMIT ?")
        params.append(limit)

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {
                "id": r[0], "timestamp": r[1], "date": r[2], "time": r[3],
                "event_type": r[4], "severity": r[5], "camera_id": r[6],
                "description": r[7], "details": json.loads(r[8]),
                "snapshot_url": f"/snapshots/{r[9]}" if r[9] else None,
            }
            for r in rows
        ]

    def count(self):
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
