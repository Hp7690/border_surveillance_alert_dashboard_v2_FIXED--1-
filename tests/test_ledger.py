"""
Tests for the evidence ledger. Run:  python -m unittest tests.test_ledger -v
Each "attack" test plays the role of a malicious insider and asserts the
tampering is DETECTED (a hash chain can't stop edits, it makes them visible).
"""
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.alert_manager import AlertManager
from core.event_log import EventLogger
from core.ledger import EvidenceLedger, block_hash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(path, data, mode="wb"):
    with open(path, mode) as f:
        f.write(data)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.snap = os.path.join(self.d, "snapshots")
        self.clips = os.path.join(self.d, "clips")
        os.makedirs(self.snap)
        os.makedirs(self.clips)
        self.anchors = []
        self.led = EvidenceLedger(
            db_path=os.path.join(self.d, "ledger.db"), key_path=os.path.join(self.d, "k.pem"),
            anchor_path=os.path.join(self.d, "anchors.jsonl"),
            snapshot_dir=self.snap, clip_dir=self.clips, anchor_every=5,
            anchor_hook=self.anchors.append)

    def tearDown(self):
        self.tmp.cleanup()

    def alert(self, aid="a1", etype="weapon_detected", sev="CRITICAL", details=None):
        snap = os.path.join(self.snap, f"{aid}.jpg")
        write(snap, b"\xff\xd8fake-jpeg-bytes-" + aid.encode())
        return self.led.record_alert({"id": aid, "event_type": etype, "severity": sev, "camera_id": "BOP-01",
                                      "threat_score": 90, "timestamp": 1.7e9, "details": details or {}}, snap)

    def raw_db(self):
        c = sqlite3.connect(self.led.db_path)
        for t in ("blocks_no_update", "blocks_no_delete"):   # insider drops the safety triggers
            c.execute(f"DROP TRIGGER IF EXISTS {t}")
        return c


class TestHappyPath(Base):
    def test_full_custody_trail_verifies(self):
        self.alert()
        clip = os.path.join(self.clips, "a1.mp4")
        write(clip, b"video-bytes")
        self.led.record_clip("a1", clip)
        self.led.record_ack("a1", "officer_sharma", "10.0.0.5")
        self.led.record_view("a1", "officer_sharma", "snapshot", os.path.join(self.snap, "a1.jpg"))
        self.led.record_action("a1", "officer_sharma", "dispatched", "QRF sent to sector 4")
        trail = [b["entry_type"] for b in self.led.custody("a1")]
        self.assertEqual(trail, ["ALERT_RAISED", "CLIP_SEALED", "ACKNOWLEDGED", "VIEWED_EVIDENCE", "ACTION_TAKEN"])
        self.assertEqual(self.led.custody("a1")[-1]["actor"], "officer_sharma")
        self.assertTrue(self.led.verify_chain()["ok"])
        ev = self.led.verify_alert_evidence("a1")
        self.assertEqual((ev["snapshot"]["status"], ev["clip"]["status"]), ("OK", "OK"))

    def test_bad_action_code_rejected(self):
        self.alert()
        with self.assertRaises(ValueError):
            self.led.record_action("a1", "x", "DELETE_EVERYTHING")

    def test_view_only_logged_for_sealed_alerts_and_deduped(self):
        p = os.path.join(self.snap, "zzz.jpg")
        write(p, b"x")
        self.assertIsNone(self.led.record_view("zzz", "u", "snapshot", p))   # never sealed
        self.alert()
        p1 = os.path.join(self.snap, "a1.jpg")
        self.assertIsNotNone(self.led.record_view("a1", "u", "snapshot", p1))
        self.assertIsNone(self.led.record_view("a1", "u", "snapshot", p1))   # thumbnail reload
        self.assertIsNotNone(self.led.record_view("a1", "other", "snapshot", p1))

    def test_identity_never_in_clear_on_chain(self):
        self.alert("f1", "known_person_seen", "LOW", {"name": "Harsh", "track_id": 3})
        raw = json.dumps(self.led.custody("f1"))
        self.assertNotIn("Harsh", raw)
        self.assertEqual(self.led.custody("f1")[0]["payload"]["details_commit"],
                         self.led.commit({"name": "Harsh", "track_id": 3}))

    def test_anchors_emitted_and_pushed_offbox(self):
        for i in range(12):
            self.alert(f"x{i}")
        self.assertTrue(len(self.anchors) >= 2)
        self.assertTrue(all(a["type"] == "ledger_anchor" for a in self.anchors))
        self.assertTrue(self.led.verify_chain()["anchors_checked"] >= 2)


class TestKeyHandling(Base):
    def test_empty_key_file_is_regenerated_and_key_is_reloadable(self):
        kp = os.path.join(self.d, "k2.pem")
        write(kp, b"")                                   # leftover from an interrupted first run
        a = EvidenceLedger(db_path=os.path.join(self.d, "l2.db"), key_path=kp, anchor_path=os.path.join(self.d, "a2.jsonl"))
        b = EvidenceLedger(db_path=os.path.join(self.d, "l2.db"), key_path=kp, anchor_path=os.path.join(self.d, "a2.jsonl"))
        self.assertEqual(a.key_fingerprint, b.key_fingerprint)
        self.assertTrue(b.verify_chain()["ok"])

    def test_key_creation_works_under_eventlet_patching(self):
        """Regression: os.fdopen is replaced by GreenPipe under eventlet.monkey_patch (breaks on Windows)."""
        import eventlet
        eventlet.monkey_patch(thread=False)
        kp = os.path.join(self.d, "k3.pem")
        EvidenceLedger(db_path=os.path.join(self.d, "l3.db"), key_path=kp, anchor_path=os.path.join(self.d, "a3.jsonl"))
        with open(kp, "rb") as f:
            self.assertIn(b"BEGIN PRIVATE KEY", f.read())


class TestAttacks(Base):
    def test_evidence_file_swapped(self):
        self.alert()
        write(os.path.join(self.snap, "a1.jpg"), b"photoshopped")
        self.assertEqual(self.led.verify_alert_evidence("a1")["snapshot"]["status"], "TAMPERED")

    def test_evidence_file_deleted(self):
        self.alert()
        os.remove(os.path.join(self.snap, "a1.jpg"))
        self.assertEqual(self.led.verify_alert_evidence("a1")["snapshot"]["status"], "MISSING")

    def test_ledger_row_edited(self):
        for i in range(4):
            self.alert(f"a{i}")
        c = self.raw_db()
        c.execute("UPDATE blocks SET actor='nobody' WHERE idx=3"); c.commit()
        r = self.led.verify_chain()
        self.assertFalse(r["ok"]); self.assertEqual(r["first_bad_idx"], 3)

    def test_ledger_row_deleted_from_middle(self):
        for i in range(4):
            self.alert(f"a{i}")
        c = self.raw_db()
        c.execute("DELETE FROM blocks WHERE idx=2"); c.commit()
        r = self.led.verify_chain()
        self.assertFalse(r["ok"]); self.assertIn("missing", r["reason"])

    def test_forged_hash_without_key_fails_signature(self):
        """Attacker edits a block AND recomputes hash + all later prev_hash links,
        but cannot sign -> signature check catches it."""
        for i in range(3):
            self.alert(f"a{i}")
        c = self.raw_db()
        rows = c.execute("SELECT idx, ts_us, entry_type, alert_id, actor, payload_json, prev_hash FROM blocks "
                         "ORDER BY idx").fetchall()
        prev = rows[0][6]
        for (idx, ts, et, aid, actor, pj, _p) in rows:
            actor = "intruder" if idx == 2 else actor
            h = block_hash(idx, ts, et, aid, actor, json.loads(pj), prev)
            c.execute("UPDATE blocks SET actor=?, prev_hash=?, hash=? WHERE idx=?", (actor, prev, h, idx))
            prev = h
        c.commit()
        r = self.led.verify_chain()
        self.assertFalse(r["ok"]); self.assertIn("signature", r["reason"])

    def test_rollback_past_an_anchor_is_detected(self):
        for i in range(8):
            self.alert(f"a{i}", sev="HIGH")
        c = self.raw_db()
        c.execute("DELETE FROM blocks WHERE idx > 3"); c.commit()     # wipe recent history incl. anchored block 5
        r = self.led.verify_chain()
        self.assertFalse(r["ok"]); self.assertIn("anchor", r["reason"])

    def test_critical_alert_and_human_actions_anchored_immediately(self):
        """A weapon alert or an operator action must be pushed off-box at once -- otherwise a
        rollback of the newest blocks (after the last periodic anchor) would go unnoticed."""
        self.alert("w1", "weapon_detected", "CRITICAL")
        self.assertEqual(self.anchors[-1]["idx"], self.led.head()["idx"])
        n = len(self.anchors)
        self.led.record_ack("w1", "officer_sharma")
        self.led.record_action("w1", "officer_sharma", "RESOLVED", "stood down")
        self.assertEqual(len(self.anchors), n + 2)
        c = self.raw_db()
        c.execute("DELETE FROM blocks WHERE idx = ?", (self.led.head()["idx"],)); c.commit()   # hide the last action
        self.assertFalse(self.led.verify_chain()["ok"])

    def test_known_limit_unanchored_tail_rollback_not_detectable_locally(self):
        """Documented limitation: HIGH alerts are anchored every N blocks, so removing only the
        blocks after the newest anchor is invisible to a purely local check."""
        for i in range(8):
            self.alert(f"a{i}", sev="HIGH")       # anchors at idx 5 only (anchor_every=5)
        c = self.raw_db()
        c.execute("DELETE FROM blocks WHERE idx > 5"); c.commit()
        self.assertTrue(self.led.verify_chain()["ok"])

    def test_triggers_block_casual_edits(self):
        self.alert()
        with self.assertRaises(sqlite3.DatabaseError):
            with sqlite3.connect(self.led.db_path) as c:
                c.execute("UPDATE blocks SET actor='x' WHERE idx=1")


class TestBundle(Base):
    def _export(self):
        self.alert()
        self.led.record_ack("a1", "officer_sharma")
        data = self.led.build_bundle("a1", "officer_sharma",
                                     verifier_script_path=os.path.join(ROOT, "verify_bundle.py"))
        out = os.path.join(self.d, "bundle")
        zipfile.ZipFile(io.BytesIO(data)).extractall(out)
        return out

    def _run(self, folder):
        return subprocess.run([sys.executable, os.path.join(folder, "verify_bundle.py"), folder],
                              capture_output=True, text=True)

    def test_bundle_verifies_offline_and_logs_export(self):
        out = self._export()
        self.assertIn("EXPORTED", [b["entry_type"] for b in self.led.custody("a1")])
        r = self._run(out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("VERIFICATION PASSED", r.stdout)

    def test_bundle_with_altered_evidence_fails(self):
        out = self._export()
        write(os.path.join(out, "evidence", "a1.jpg"), b"edited")
        r = self._run(out)
        self.assertEqual(r.returncode, 1); self.assertIn("VERIFICATION FAILED", r.stdout)

    def test_bundle_with_edited_custody_fails(self):
        out = self._export()
        p = os.path.join(out, "custody.json")
        with open(p) as f:
            c = json.load(f)
        c["blocks"][1]["actor"] = "someone_else"
        write(p, json.dumps(c), "w")
        r = self._run(out)
        self.assertEqual(r.returncode, 1); self.assertIn("hash=BAD", r.stdout)


class TestAlertManagerIntegration(Base):
    def test_raise_alert_seals_snapshot_hash_and_skips_noise(self):
        el = EventLogger(db_path=os.path.join(self.d, "events.db"), snapshot_dir=self.snap)
        self.led.snapshot_dir = self.snap
        am = AlertManager(event_logger=el, ledger=self.led)
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        a = am.raise_alert("weapon_detected", "BOP-01", {"class_name": "Pistol", "confidence": 0.9}, frame=frame)
        am.raise_alert("person_detected", "BOP-01", {"track_id": 1}, frame=frame)      # MEDIUM demo noise -> not sealed
        am.log_silent("known_person_seen", "BOP-01", {"name": "Harsh", "track_id": 1}, frame=frame)
        sealed = {x["alert_id"]: x for x in self.led.list_sealed_alerts()}
        self.assertIn(a.id, sealed)
        self.assertEqual(len(sealed), 2)   # weapon + known_person_seen; person_detected skipped
        self.assertEqual(self.led.verify_alert_evidence(a.id)["snapshot"]["status"], "OK")
        self.assertTrue(self.led.verify_chain()["ok"])
        self.assertEqual(am.ledger_failures, 0)

    def test_ledger_failure_never_crashes_pipeline(self):
        class Boom:
            def record_alert(self, *a, **k): raise RuntimeError("disk full")
        am = AlertManager(ledger=Boom())
        self.assertIsNotNone(am.raise_alert("weapon_detected", "BOP-01", {}))
        self.assertEqual(am.ledger_failures, 1)


if __name__ == "__main__":
    unittest.main()
