"""
pose_detector.py
------------------
Wraps YOLOv8-Pose to extract body keypoints for every person in a frame.
Used by suspicious_activity.py to detect postures like crawling/climbing
that a plain bounding-box detector can't distinguish from "standing".

COCO 17-keypoint order (what yolov8n-pose.pt outputs):
  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
"""

from ultralytics import YOLO

KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}


class PoseResult:
    __slots__ = ("bbox", "keypoints", "confidence")

    def __init__(self, bbox, keypoints, confidence):
        self.bbox = bbox                # (x1,y1,x2,y2)
        self.keypoints = keypoints      # np.array shape (17, 3) -> x, y, visibility
        self.confidence = confidence

    def kp(self, name):
        """Return (x, y, visible) for a named keypoint, or None if low-confidence/missing."""
        idx = KP[name]
        x, y, v = self.keypoints[idx]
        if v < 0.3:  # visibility/confidence too low to trust
            return None
        return (float(x), float(y))


class PoseDetector:
    def __init__(self, model_path="models/yolov8n-pose.pt", conf_threshold=0.4, device="cpu"):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame):
        """Returns list[PoseResult] for every person detected in the frame."""
        results = self.model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )
        out = []
        if not results:
            return out

        r = results[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return out

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()  # (N, 17, 3)

        for bbox, conf, kp_array in zip(boxes_xyxy, confs, kpts):
            out.append(PoseResult(tuple(bbox.tolist()), kp_array, float(conf)))
        return out
