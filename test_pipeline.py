"""Standalone smoke test: run every module on a video for N frames and
print what happens, without spinning up the Flask/SocketIO server.
Use this to sanity-check the pipeline before running the full app.
"""
import sys
import time
import cv2

from core.detector import PersonVehicleDetector
from core.virtual_fence import VirtualFence
from core.weapon_detector import WeaponDetector
from core.suspicious_activity import SuspiciousActivityDetector
from core.face_detector import FaceDetector
from core.anpr import ANPR
from core.night_detector import is_night_frame

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_videos/synthetic_test.mp4"
MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 40

print(f"Loading models...")
detector = PersonVehicleDetector(model_path="models/yolov8n.pt")
weapon_detector = WeaponDetector(model_path="models/weapon_detection.pt")
fence = VirtualFence(name="Restricted Border Zone",
                      polygon_points=[(550, 250), (960, 250), (960, 540), (550, 540)],
                      loiter_seconds=8)
suspicious_detector = SuspiciousActivityDetector(fence=fence)
face_detector = FaceDetector()
anpr = ANPR(plate_model_path="models/license_plate_detector.pt")
print("Models loaded OK.\n")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: could not open {VIDEO_PATH}")
    sys.exit(1)

frame_idx = 0
t_start = time.time()
total_detections = 0
total_weapon_hits = 0
total_sus_events = 0
total_faces = 0
total_plates = 0

while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print(f"[frame {frame_idx}] end of video reached")
        break

    detections = detector.detect(frame, use_tracking=True)
    fence_events = fence.evaluate(detections)
    weapons = weapon_detector.detect(frame)
    poses = suspicious_detector.pose_detector.detect(frame)
    sus_events = suspicious_detector.evaluate(frame, detections, poses=poses)
    faces = face_detector.detect_all(poses)
    night = is_night_frame(frame)

    plates = []
    for det in detections:
        if det.class_name in ("car", "truck", "bus", "motorcycle"):
            plates.extend(anpr.read(frame, vehicle_bbox=det.bbox, vehicle_track_id=det.track_id))

    total_detections += len(detections)
    total_weapon_hits += len(weapons)
    total_sus_events += len(sus_events)
    total_faces += len(faces)
    total_plates += len(plates)

    if detections or fence_events or weapons or sus_events or faces or plates:
        print(f"[frame {frame_idx}] det={len(detections)} fence={fence_events} "
              f"weapons={weapons} sus={sus_events} faces={len(faces)} plates={plates} night={night}")

    frame_idx += 1

elapsed = time.time() - t_start
cap.release()

print("\n--- SUMMARY ---")
print(f"Frames processed: {frame_idx}")
print(f"Time taken: {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS avg, ALL modules, CPU)")
print(f"Total detections: {total_detections}")
print(f"Total weapon hits: {total_weapon_hits}")
print(f"Total suspicious events: {total_sus_events}")
print(f"Total faces: {total_faces}")
print(f"Total plates: {total_plates}")
print("\nFull pipeline (all 6 features) ran without crashing. ✔")
