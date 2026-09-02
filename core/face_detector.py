"""
face_detector.py
------------------
Face "detection" derived from the pose model's head keypoints (nose,
eyes, ears) rather than a separate dedicated face-detection model.

Why this approach instead of a standalone face detector (Haar/YuNet)?
  - We already run YOLOv8-Pose for suspicious-activity posture analysis
    (see pose_detector.py / suspicious_activity.py) on every "heavy"
    frame -- reusing its head keypoints for face localization means face
    detection comes essentially for free, with no extra model to load
    and no extra inference pass over the frame.
  - The pose model is trained on real human data and is already proven
    reliable in this pipeline (verified on real photos), which is a
    safer bet than pulling in another dependency.

Scope note: same as before -- this only locates "a face is here", it
does not IDENTIFY the person. Real facial recognition against a
watchlist would need a dedicated embedding model (ArcFace) + a vector
index (FAISS) compared against enrolled photos -- left as a documented
Phase-2 extension to avoid the privacy/dataset scope that comes with a
real watchlist database.
"""


class FaceDetection:
    __slots__ = ("bbox", "confidence")

    def __init__(self, bbox, confidence):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence


class FaceDetector:
    """
    Not a model wrapper -- a thin geometric helper that turns a
    PoseResult's head keypoints into a face bounding box. Call
    `from_pose(pose)` once per detected pose (already computed
    elsewhere in the pipeline).
    """

    def __init__(self, min_keypoints_required=2, box_margin_ratio=0.6):
        self.min_keypoints_required = min_keypoints_required
        self.box_margin_ratio = box_margin_ratio

    def from_pose(self, pose):
        """
        pose: a PoseResult from core.pose_detector.PoseDetector.detect()
        Returns a FaceDetection, or None if not enough head keypoints
        were visible (e.g. person facing away from camera).
        """
        head_points = []
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            pt = pose.kp(name)
            if pt is not None:
                head_points.append(pt)

        if len(head_points) < self.min_keypoints_required:
            return None

        xs = [p[0] for p in head_points]
        ys = [p[1] for p in head_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Head keypoints only mark facial landmarks (eyes/nose/ears), not
        # the full face extent, so expand the tight box outward -- sized
        # relative to the person's own bbox height for scale-invariance
        # (a face close to camera vs. far away should scale accordingly).
        person_height = pose.bbox[3] - pose.bbox[1]
        margin = max(person_height * 0.08, 15)  # empirical: ~8% of body height per side

        x1 = x_min - margin
        y1 = y_min - margin * 1.2  # slightly more headroom above eyes/nose
        x2 = x_max + margin
        y2 = y_max + margin * 1.5  # extend down to include chin/jaw

        return FaceDetection(bbox=(x1, y1, x2, y2), confidence=pose.confidence)

    def detect_all(self, poses):
        """Convenience: run from_pose() over a list of PoseResults, skipping Nones."""
        results = []
        for pose in poses:
            face = self.from_pose(pose)
            if face is not None:
                results.append(face)
        return results
