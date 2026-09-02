"""
enroll_faces.py
-----------------
Add a person to the dashboard's known-face WHITELIST. Once enrolled,
the live pipeline stops raising person/zone/suspicious-activity alerts
when that specific person is on camera (weapon alerts still always
fire, regardless of identity -- see core/face_recognizer.py).

Usage:
    python enroll_faces.py --name "Officer_Sharma" --images photo1.jpg photo2.jpg

Tips for good accuracy:
  - Use 2-4 clear, well-lit, front-facing photos per person.
  - Re-run with the SAME --name later to add more photos for the same
    person (helps accuracy across different lighting/angles).
  - Each photo must contain exactly one clearly visible face; group
    photos or photos where the face is small/angled will be skipped.

Run this once per whitelisted person before starting app.py.
"""
import argparse

from core.face_recognizer import FaceRecognizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True,
                         help="Display name for this person, e.g. 'Officer_Sharma' (no spaces recommended)")
    parser.add_argument("--images", nargs="+", required=True,
                         help="One or more photo file paths")
    args = parser.parse_args()

    fr = FaceRecognizer()
    fr.enroll(args.name, args.images)
