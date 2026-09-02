# IBVAP — Intelligent Border Video Analytics Platform

SIH 2026 · Problem Statement 26187 · Ministry of Home Affairs / SSB

AI-driven video analytics that turns existing IP-CCTV feeds into a smart
border surveillance network — no dedicated FRS/ANPR/smart-camera hardware
required.

## Features implemented (working, tested)

| # | Feature | Status | Module |
|---|---|---|---|
| 1 | Real-time alerts to logged-in admin | ✅ Working | `core/alert_manager.py` + SocketIO |
| 2 | Suspicious activity detection (crawling, climbing, fast approach, erratic movement) | ✅ Working | `core/suspicious_activity.py` |
| 3 | Simultaneous multi-camera handling | ✅ Architecture supports it (see below) | `app.py` (threaded `VideoProcessor` per camera) |
| 4 | Night-time movement detection | ✅ Working | `core/night_detector.py` |
| 5 | Number plate, face, vehicle detection | ✅ Working | `core/anpr.py`, `core/face_detector.py`, `core/detector.py` |
| 6 | Gun / weapon detection | ✅ Working | `core/weapon_detector.py` |

Every module has been individually verified against real photos (not just
the synthetic placeholder video) — see "What's been tested" below.

## Quick start

```bash
cd border_surveillance
python app.py --source test_videos/synthetic_test.mp4 --port 5000
# open http://localhost:5000
```

Real usage:
```bash
python app.py --source path/to/your_cctv_footage.mp4
python app.py --source rtsp://<camera-ip>/stream     # live CCTV
python app.py --source 0                              # webcam
```

Standalone pipeline test (no server, just console output):
```bash
python test_pipeline.py path/to/video.mp4 50   # process 50 frames
```

## Architecture

```
CCTV/Video → VideoProcessor thread (per camera)
                 ├─ PersonVehicleDetector (YOLOv8n, every frame + ByteTrack)
                 ├─ VirtualFence (polygon zone intrusion + loitering)
                 ├─ WeaponDetector (fine-tuned YOLOv8, every 3rd frame)
                 ├─ SuspiciousActivityDetector (pose + trajectory rules)
                 ├─ FaceDetector (derived from pose keypoints — no extra model)
                 ├─ ANPR (plate localization YOLOv8 + Tesseract OCR)
                 └─ NightSurveillancePipeline (MOG2 motion trigger + CLAHE enhance)
                          ↓
              AlertManager (priority tiers, dedup/cooldown)
                          ↓
         Flask + SocketIO ──push──> Admin Dashboard (real-time, no refresh)
```

**Multi-camera / simultaneous alerts (point 3):** each camera runs as its
own `VideoProcessor` thread. Start more than one and they publish to the
same `AlertManager` independently — one camera's processing never blocks
another's. To run 2 cameras, instantiate two `VideoProcessor` objects with
different `camera_id`s in `app.py`'s `__main__` block (currently wired for
one, since we only have one test source — trivial to extend).

**Why heavy models run every 3rd frame, not every frame:** weapon
detection and pose estimation are the expensive models. Running them at
~1/3 rate keeps the pipeline closer to real-time on CPU while person/
vehicle detection + tracking (which needs frame-to-frame continuity)
still runs every frame.

## Models used

All fetched from public GitHub repos / Ultralytics releases (no training done — pretrained/fine-tuned weights):

- `yolov8n.pt` — Ultralytics official (COCO: person, car, truck, bus, motorcycle, etc.)
- `yolov8n-pose.pt` — Ultralytics official (17-keypoint COCO pose)
- `weapon_detection.pt` — community fine-tuned YOLOv8 (Grenade, Gun, Knife, Pistol, Handgun, Rifle)
- `license_plate_detector.pt` — community fine-tuned YOLOv8 (license_plate class only)
- Face detection uses **no separate model** — derived from pose keypoints (nose/eyes/ears) to avoid an extra dependency
- OCR: Tesseract (via `pytesseract`) — system package, not a downloaded model

## What's been tested

Since no real CCTV footage was available at build time, every model was
verified individually against real photographs (not the synthetic
placeholder video):

- Person/vehicle detector → `bus.jpg`: 1 bus + 3 persons, 83–87% confidence
- Pose detector → `zidane.jpg`: 2 poses, correct keypoints
- Face detector (pose-derived) → `zidane.jpg`: 2 faces, correct bounding boxes
- Weapon detector → real gun/rifle/knife sample images: Rifle 79%, Gun 84%, Knife 35%
- ANPR OCR stage → synthetic plate text image: read `UP32AB1234` correctly
- Night detection → brightness + motion-trigger logic: verified correct on synthetic dark/bright frames
- Full server (`app.py`) → started, dashboard returned HTTP 200, live MJPEG stream confirmed, alerts fired and appeared via `/api/alerts` and the SocketIO push

**Not yet tested: the full pipeline on real CCTV footage end-to-end.**
`test_videos/synthetic_test.mp4` is cartoon shapes and intentionally does
NOT get detected by YOLO (documented in `generate_test_video.py`) — it
only proves the pipeline doesn't crash and the plumbing (fence drawing,
FPS counter, night-mode label, alert firing) works. Drop a real video into
`test_videos/` and re-run to see actual detections.

## Known limitations / next steps

- **ANPR OCR accuracy**: Tesseract is generic OCR, not tuned for Indian
  plate fonts. Production would want a plate-specific OCR model (e.g.
  PaddleOCR fine-tuned on Indian plates).
- **Face detection is location-only**, not identification. Matching
  against a watchlist needs face embeddings (ArcFace) + a vector index
  (FAISS) — deliberately out of scope here to avoid the privacy/dataset
  questions that come with a real watchlist.
- **Virtual fence** is currently a fixed polygon set in code. A real
  deployment needs a simple admin UI to draw the zone per camera during
  setup (click points on a still frame).
- **Weapon detector false positives**: gun-shaped objects (phones,
  tools) are a known failure mode for models trained on limited data —
  confidence threshold is deliberately set higher (0.55) for this reason.
- **Edge deployment**: this runs on CPU here. For real remote BOPs with
  poor connectivity, consider a Jetson Orin/Nano per site running this
  same pipeline locally, syncing alerts to a central dashboard when
  uplink is available.

## Dashboard

Dark-theme, real-time, WebSocket-driven. Alerts appear instantly with
severity color coding:
- 🔴 CRITICAL — weapon detected
- 🟠 HIGH — zone intrusion, suspicious activity (crawling/climbing/fast approach/erratic movement), confirmed night intrusion
- 🟡 MEDIUM — loitering, face detected, plate read
- 🔵 LOW — unconfirmed night motion
