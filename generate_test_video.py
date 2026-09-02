"""
Generates a synthetic test video simulating:
  - a "person" (moving rectangle + circle head) walking across the frame,
    eventually crossing into the restricted (virtual fence) zone
  - a "vehicle" (moving rectangle) driving past

This is ONLY for testing the pipeline end-to-end before real CCTV
footage is available. Swap this out for actual footage anytime --
the detector/fence code doesn't care about the source.
"""

import cv2
import numpy as np
import os

OUT_PATH = "test_videos/synthetic_test.mp4"
W, H = 960, 540
FPS = 20
DURATION_SEC = 20
N_FRAMES = FPS * DURATION_SEC

os.makedirs("test_videos", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))


def draw_person(frame, x, y, color=(50, 200, 50)):
    # simple humanoid: head (circle) + body (rectangle) so YOLO has a
    # fighting chance of recognizing something person-shaped; note that
    # YOLO trained on real photos will NOT reliably detect cartoon shapes
    # -- this synthetic video is for testing PIPELINE PLUMBING, not
    # detection accuracy. Use real footage to validate detection quality.
    cv2.circle(frame, (x, y - 25), 10, color, -1)
    cv2.rectangle(frame, (x - 12, y - 15), (x + 12, y + 35), color, -1)


def draw_vehicle(frame, x, y, color=(200, 120, 50)):
    cv2.rectangle(frame, (x - 40, y - 15), (x + 40, y + 15), color, -1)
    cv2.circle(frame, (x - 25, y + 15), 6, (20, 20, 20), -1)
    cv2.circle(frame, (x + 25, y + 15), 6, (20, 20, 20), -1)


for i in range(N_FRAMES):
    frame = np.full((H, W, 3), (40, 40, 40), dtype=np.uint8)  # dark gray "ground"

    # simulated horizon / border line for visual context
    cv2.line(frame, (0, 100), (W, 100), (80, 80, 80), 2)
    cv2.putText(frame, "SYNTHETIC TEST FEED - BOP Camera 01", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    t = i / N_FRAMES

    # Person walks left -> right, crossing the frame over the full duration
    px = int(50 + t * (W - 100))
    py = 400
    draw_person(frame, px, py)

    # Vehicle drives right -> left in the upper lane, a bit faster
    vx = int(W - 50 - ((t * 1.4) % 1.0) * (W - 100))
    vy = 200
    draw_vehicle(frame, vx, vy)

    writer.write(frame)

writer.release()
print(f"Synthetic test video written to {OUT_PATH} ({N_FRAMES} frames, {DURATION_SEC}s)")
