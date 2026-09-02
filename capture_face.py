"""
capture_face.py
-----------------
Capture enrollment photos DIRECTLY FROM YOUR WEBCAM, instead of using
unrelated photos (e.g. professional headshots). This matters a lot for
accuracy: the face recognizer compares raw pixel patterns, so matching
lighting/camera/resolution between enrollment and live detection makes
a big difference.

Usage:
    python capture_face.py --name "Harsh"

Controls (a preview window will open):
    SPACE  - capture the current frame (look straight at the camera,
             good lighting, no mask/sunglasses)
    q      - quit and enroll all captured photos

Captures are saved to captures/<name>/ and then automatically enrolled
via core.face_recognizer.FaceRecognizer.enroll(). Aim for 5-8 captures
per person, with slightly different angles/expressions for best
accuracy.
"""
import argparse
import os
import time

import cv2

from core.face_recognizer import FaceRecognizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Display name for this person, e.g. 'Harsh'")
    parser.add_argument("--source", default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    out_dir = os.path.join("captures", args.name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {source}")
        raise SystemExit(1)

    cascade = cv2.CascadeClassifier(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "haarcascade_frontalface_default.xml")
    )

    print("Preview window opening. Press SPACE to capture, 'q' when done.")
    print(f"Captures will be saved to: {out_dir}")

    saved_paths = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(display, f"Captured: {count}  |  SPACE=capture  q=finish",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("capture_face.py - press SPACE to capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if len(faces) == 0:
                print("  No face detected in frame -- try again (better lighting/angle).")
                continue
            path = os.path.join(out_dir, f"{args.name}_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(path, frame)
            saved_paths.append(path)
            count += 1
            print(f"  Captured {count}: {path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not saved_paths:
        print("No photos captured -- nothing to enroll.")
        raise SystemExit(0)

    print(f"\nEnrolling {len(saved_paths)} photo(s) for '{args.name}'...")
    fr = FaceRecognizer()
    fr.enroll(args.name, saved_paths)
