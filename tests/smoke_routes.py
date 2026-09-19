"""
Route-level smoke test of the ledger endpoints against the REAL app.py.
Heavy ML modules (ultralytics/YOLO/tesseract) are stubbed so this runs anywhere;
everything in app.py + core/ledger.py + core/alert_manager.py + core/event_log.py is real.
Run:  python tests/smoke_routes.py
"""
import io, json, os, subprocess, sys, tempfile, types, zipfile
from unittest.mock import MagicMock
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(); os.chdir(tmp)            # app writes ./data, ./webhooks.json here
sys.path.insert(0, ROOT)
for m in ("detector", "virtual_fence", "weapon_detector", "suspicious_activity",
          "night_detector", "face_detector", "face_recognizer", "anpr"):
    mod = types.ModuleType(f"core.{m}")
    for n in ("PersonVehicleDetector", "VirtualFence", "WeaponDetector", "SuspiciousActivityDetector",
              "NightSurveillancePipeline", "is_night_frame", "FaceDetector", "FaceRecognizer", "ANPR"):
        setattr(mod, n, MagicMock())
    sys.modules[f"core.{m}"] = mod
os.makedirs("templates", exist_ok=True)
import shutil; shutil.rmtree("templates"); shutil.copytree(os.path.join(ROOT, "templates"), "templates")

import app as A
from core.auth import create_user
create_user("officer_sharma", "pw1"); create_user("officer_rao", "pw2")

def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond: check.failed = True
check.failed = False

A.app.root_path = tmp   # send_from_directory resolves relative dirs against root_path; in production cwd == app dir
c = A.app.test_client()
check("ledger API requires login", c.get("/api/ledger/status").status_code == 401)
check("ledger page requires login", c.get("/ledger").status_code == 302)

frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
alert = A.alert_manager.raise_alert("weapon_detected", "BOP-01", {"class_name": "Pistol", "confidence": 0.9}, frame=frame)
aid = alert.id
clip_path = os.path.join(A.CLIP_DIR, f"{aid}.mp4"); os.makedirs(A.CLIP_DIR, exist_ok=True)
with open(clip_path, "wb") as f: f.write(b"fake-mp4-bytes")
A.ledger.record_clip(aid, clip_path)

# officer 1
c.post("/login", data={"username": "officer_sharma", "password": "pw1"})
check("ledger page renders", c.get("/ledger").status_code == 200 and b"EVIDENCE LEDGER" in c.get("/ledger").data)
check("status endpoint", c.get("/api/ledger/status").get_json()["blocks"] >= 3)
check("sealed alert listed", any(a["alert_id"] == aid for a in c.get("/api/ledger/alerts").get_json()))
snap = alert_file = f"{aid}.jpg"
check("snapshot served (+ view logged)", c.get(f"/snapshots/{snap}").status_code == 200)
check("clip served (+ view logged)", c.get(f"/clips/{aid}.mp4").status_code == 200)
r = c.post(f"/api/alerts/{aid}/ack"); check("ack ok", r.status_code == 200 and r.get_json()["success"])
check("ack unknown alert -> 404", c.post("/api/alerts/nope/ack").status_code == 404)
r = c.post(f"/api/alerts/{aid}/action", json={"action": "dispatched", "note": "QRF <b>sent</b>"})
check("action ok", r.status_code == 200)
check("bad action -> 400", c.post(f"/api/alerts/{aid}/action", json={"action": "HACK"}).status_code == 400)

# officer 2 (different identity)
c.get("/logout"); c.post("/login", data={"username": "officer_rao", "password": "pw2"})
c.post(f"/api/alerts/{aid}/action", json={"action": "RESOLVED", "note": "sector clear"})

d = c.get(f"/api/ledger/custody/{aid}").get_json()
trail = [(b["entry_type"], b["actor"]) for b in d["blocks"]]
print("     trail:", trail)
check("actors attributed correctly",
      ("ACKNOWLEDGED", "officer_sharma") in trail and ("ACTION_TAKEN", "officer_rao") in trail
      and ("VIEWED_EVIDENCE", "officer_sharma") in trail)
check("evidence OK", d["evidence"]["snapshot"]["status"] == "OK" and d["evidence"]["clip"]["status"] == "OK")
check("chain verifies", c.get("/api/ledger/verify").get_json()["ok"])

# bundle -> offline verify
r = c.get(f"/api/ledger/bundle/{aid}")
check("bundle is a zip", r.status_code == 200 and r.mimetype == "application/zip")
out = os.path.join(tmp, "bundle"); zipfile.ZipFile(io.BytesIO(r.data)).extractall(out)
check("bundle contains verifier + evidence", os.path.exists(f"{out}/verify_bundle.py") and os.path.exists(f"{out}/evidence/{aid}.jpg"))
p = subprocess.run([sys.executable, f"{out}/verify_bundle.py", out], capture_output=True, text=True)
check("offline verifier passes", p.returncode == 0, p.stdout.strip().splitlines()[-1])
check("EXPORT attributed to officer_rao", ("EXPORTED", "officer_rao") in
      [(b["entry_type"], b["actor"]) for b in c.get(f"/api/ledger/custody/{aid}").get_json()["blocks"]])

# tamper: swap the snapshot on disk
with open(os.path.join(A.event_logger.snapshot_dir, snap), "wb") as f: f.write(b"photoshopped")
check("tampered snapshot flagged via API", c.get(f"/api/ledger/custody/{aid}").get_json()["evidence"]["snapshot"]["status"] == "TAMPERED")
check("pubkey endpoint", b"BEGIN PUBLIC KEY" in c.get("/api/ledger/pubkey").data)
check("anchor endpoint", "idx" in c.post("/api/ledger/anchor").get_json())
print("\nRESULT:", "FAILED" if check.failed else "ALL PASSED"); sys.exit(1 if check.failed else 0)
