# modules/tracker.py — SORT Tracker
# Kalman Filter + Hungarian Algorithm

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize  import linear_sum_assignment


class KalmanBoxTracker:
    """
    Single vehicle track — Kalman Filter based.
    State: [x1, y1, x2, y2, vx, vy, vw, vh]
    """
    count = 0

    def __init__(self, bbox):
        # Kalman Filter — 8 state, 4 measurement
        self.kf         = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F       = np.eye(8)
        self.kf.F[0, 4] = 1
        self.kf.F[1, 5] = 1
        self.kf.F[2, 6] = 1
        self.kf.F[3, 7] = 1

        self.kf.H       = np.eye(4, 8)
        self.kf.R      *= 10.0
        self.kf.P      *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4]   = np.array(bbox).reshape((4, 1))

        KalmanBoxTracker.count += 1
        self.track_id    = KalmanBoxTracker.count
        self.hit_streak  = 0
        self.age         = 0
        self.hits        = 0
        self.time_since_update = 0

        # Store class info
        self.class_id   = None
        self.class_name = "vehicle"
        self.confidence = 0.0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits             += 1
        self.hit_streak       += 1
        self.kf.update(np.array(bbox).reshape((4, 1)))

    def predict(self):
        self.kf.predict()
        self.age              += 1
        if self.time_since_update > 0:
            self.hit_streak    = 0
        self.time_since_update += 1
        return self.kf.x[:4].flatten()

    def get_state(self):
        return self.kf.x[:4].flatten()


def iou(bb1, bb2):
    """IoU between two bboxes [x1,y1,x2,y2]."""
    xx1 = max(bb1[0], bb2[0])
    yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2])
    yy2 = min(bb1[3], bb2[3])

    w   = max(0.0, xx2 - xx1)
    h   = max(0.0, yy2 - yy1)
    intersection = w * h

    area1 = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-6)


class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracking.
    Hungarian assignment + Kalman Filter.
    """

    def __init__(self, max_age=8, min_hits=3,
                 iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        """
        detections: list of dicts from VehicleDetector

        Returns list of track dicts:
        {
            track_id, bbox, center,
            width_px, class_name,
            class_id, confidence, hit_streak
        }
        """
        self.frame_count += 1

        # Predict existing trackers
        predicted = []
        for t in self.trackers:
            pred = t.predict()
            predicted.append(pred)

        # Match detections to trackers
        if len(detections) > 0 and len(self.trackers) > 0:
            det_boxes  = [d["bbox"] for d in detections]
            matches, unmatched_dets, unmatched_trks = \
                self._associate(det_boxes, predicted)
        else:
            matches        = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.trackers)))

        # Update matched trackers
        for d_idx, t_idx in matches:
            det = detections[d_idx]
            self.trackers[t_idx].update(det["bbox"])
            self.trackers[t_idx].class_name = \
                det["class_name"]
            self.trackers[t_idx].class_id   = \
                det["class_id"]
            self.trackers[t_idx].confidence = \
                det["confidence"]

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            trk = KalmanBoxTracker(det["bbox"])
            trk.class_name = det["class_name"]
            trk.class_id   = det["class_id"]
            trk.confidence = det["confidence"]
            self.trackers.append(trk)

        # Build output
        tracks = []
        for t in self.trackers:
            if (t.time_since_update < self.max_age and
                    (t.hit_streak >= self.min_hits or
                     self.frame_count <= self.min_hits)):

                state      = t.get_state()
                x1, y1, x2, y2 = state
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w  = x2 - x1

                tracks.append({
                    "track_id":   t.track_id,
                    "bbox":       [x1, y1, x2, y2],
                    "center":     (cx, cy),
                    "width_px":   w,
                    "class_name": t.class_name,
                    "class_id":   t.class_id,
                    "confidence": t.confidence,
                    "hit_streak": t.hit_streak
                })

        # Remove dead trackers
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update < self.max_age
        ]

        return tracks

    def _associate(self, det_boxes, trk_boxes):
        """Hungarian assignment based on IoU."""
        if len(trk_boxes) == 0:
            return ([], list(range(len(det_boxes))), [])

        # IoU matrix
        iou_matrix = np.zeros(
            (len(det_boxes), len(trk_boxes))
        )
        for d, db in enumerate(det_boxes):
            for t, tb in enumerate(trk_boxes):
                iou_matrix[d, t] = iou(db, tb)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(
            -iou_matrix
        )

        matches        = []
        unmatched_dets = []
        unmatched_trks = list(range(len(trk_boxes)))

        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] >= self.iou_threshold:
                matches.append((d, t))
                if t in unmatched_trks:
                    unmatched_trks.remove(t)
            else:
                unmatched_dets.append(d)

        for d in range(len(det_boxes)):
            if d not in [m[0] for m in matches]:
                unmatched_dets.append(d)

        unmatched_dets = list(set(unmatched_dets))
        return matches, unmatched_dets, unmatched_trks