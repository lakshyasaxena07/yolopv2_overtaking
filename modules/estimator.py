# modules/estimator.py — YOLOPv2 Version
# Lane detector dependency hatai — YOLOPv2 se lanes milti hain

import cv2
import time
import numpy as np
from collections import defaultdict, deque


class Estimator:
    """
    Per-vehicle distance, speed, direction estimation.
    YOLOPv2 version — lane_detector dependency nahi.
    """

    def __init__(self, config):
        self.cfg             = config
        self._dist_history   = defaultdict(
            lambda: deque(maxlen=15))
        self._raw_dist_history = defaultdict(
            lambda: deque(maxlen=5))
        self._bbox_history   = defaultdict(
            lambda: deque(maxlen=10))
        self._dir_votes      = defaultdict(
            lambda: deque(maxlen=8))
        self._speed_smoothed = {}
        self.EMA_ALPHA       = 0.35

        # Ego speed
        self._ego_speed_mps  = 0.0
        self._ego_history    = deque(maxlen=10)
        self._prev_gray      = None

    # ── Distance ──────────────────────────────────────────────

    def estimate_distance(self, track_id, x1, y1, x2, y2, class_id=2):
        """
        Pinhole camera model se distance with Moving Average.
        dist = (real_width * focal_length) / pixel_width
        """
        width_map = {
            2: self.cfg.REAL_CAR_WIDTH_M,
            7: self.cfg.REAL_TRUCK_WIDTH_M,
            5: self.cfg.REAL_BUS_WIDTH_M,
            3: self.cfg.REAL_CAR_WIDTH_M,
            1: self.cfg.REAL_BIKE_WIDTH_M
        }
        real_width_m = width_map.get(class_id, self.cfg.REAL_CAR_WIDTH_M)
        
        pix_w = x2 - x1
        if pix_w < 5:
            raw_dist = 999.0
        else:
            raw_dist = (real_width_m * self.cfg.FOCAL_LENGTH_PX) / pix_w
            
        self._raw_dist_history[track_id].append(raw_dist)
        avg_dist = sum(self._raw_dist_history[track_id]) / len(self._raw_dist_history[track_id])
        return round(avg_dist, 2)

    # ── Speed ─────────────────────────────────────────────────

    def estimate_speed(self, track_id, distance_m):
        """
        Distance history se relative speed.
        Positive = approaching, Negative = moving away.
        """
        now  = time.monotonic()
        hist = self._dist_history[track_id]
        hist.append((now, distance_m))

        if len(hist) < 3:
            return 0.0

        t0, d0 = hist[0]
        t1, d1 = hist[-1]
        dt     = t1 - t0

        if dt < 0.05:
            return self._speed_smoothed.get(
                track_id, 0.0)

        raw    = np.clip((d0 - d1) / dt, -60, 60)
        prev   = self._speed_smoothed.get(track_id, raw)
        smooth = (self.EMA_ALPHA * raw +
                  (1 - self.EMA_ALPHA) * prev)
        self._speed_smoothed[track_id] = smooth
        return round(smooth, 2)

    # ── Direction ─────────────────────────────────────────────

    def estimate_direction(self, track_id,
                           x1, y1, x2, y2, frame_h=720):
        """
        3 signal voting se direction detect karo.
        oncoming / same_direction / unknown
        """
        bbox = [x1, y1, x2, y2]
        self._bbox_history[track_id].append(bbox)
        hist = self._bbox_history[track_id]

        if len(hist) < 5:
            return "unknown"

        bboxes = list(hist)

        # Signal 1 — bbox width change rate
        widths     = [b[2] - b[0] for b in bboxes]
        w_start    = np.mean(widths[:3])
        w_end      = np.mean(widths[-3:])
        width_rate = (w_end - w_start) / max(w_start, 1)

        if width_rate > 0.08:
            sig1 = "oncoming"
        elif width_rate < -0.05:
            sig1 = "same_direction"
        else:
            sig1 = "unknown"

        # Signal 2 — Y center movement
        y_centers = [(b[1] + b[3]) / 2 for b in bboxes]
        y_start   = np.mean(y_centers[:3])
        y_end     = np.mean(y_centers[-3:])
        y_delta   = y_end - y_start

        if y_delta > 6:
            sig2 = "oncoming"
        elif y_delta < -4:
            sig2 = "same_direction"
        else:
            sig2 = "unknown"

        # Signal 3 — Vertical position in frame
        cy      = (y1 + y2) / 2
        y_ratio = cy / max(frame_h, 1)

        if y_ratio < 0.35:
            sig3 = "oncoming"
        elif y_ratio > 0.55:
            sig3 = "same_direction"
        else:
            sig3 = "unknown"

        # Voting — 2/3 agree
        signals  = [sig1, sig2, sig3]
        oncoming = signals.count("oncoming")
        same     = signals.count("same_direction")

        if oncoming >= 2:
            direction = "oncoming"
        elif same >= 2:
            direction = "same_direction"
        else:
            direction = "unknown"

        # Temporal smoothing
        self._dir_votes[track_id].append(direction)
        votes = list(self._dir_votes[track_id])
        total = len(votes)

        if votes.count("oncoming") >= total * 0.6:
            return "oncoming"
        elif votes.count("same_direction") >= total * 0.6:
            return "same_direction"
        else:
            return "unknown"

    # ── Ego Speed ─────────────────────────────────────────────

    def estimate_ego_motion(self, frame):
        """Optical flow se ego vehicle speed."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        features = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=80,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7
        )

        if features is None or len(features) < 5:
            self._prev_gray = gray
            return self._ego_speed_mps

        new_feat, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, features, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                10, 0.03
            )
        )
        self._prev_gray = gray

        if new_feat is None:
            return self._ego_speed_mps

        good_old = features[status == 1]
        good_new = new_feat[status == 1]

        if len(good_old) < 3:
            return self._ego_speed_mps

        dy        = good_new[:, 1] - good_old[:, 1]
        median_dy = np.median(dy)
        filtered  = dy[np.abs(dy - median_dy) < 5.0]

        if len(filtered) == 0:
            return self._ego_speed_mps

        ego_mps = abs(float(np.mean(filtered))) * 0.15
        self._ego_history.append(ego_mps)
        self._ego_speed_mps = float(
            np.mean(self._ego_history))
        return self._ego_speed_mps

    def get_ego_speed(self):
        return self._ego_speed_mps

    # ── Cleanup ───────────────────────────────────────────────

    def cleanup(self, active_ids):
        """Stale track data remove karo."""
        for store in [self._dist_history,
                      self._raw_dist_history,
                      self._bbox_history,
                      self._dir_votes]:
            for k in list(store.keys()):
                if k not in active_ids:
                    del store[k]
        for k in list(self._speed_smoothed.keys()):
            if k not in active_ids:
                del self._speed_smoothed[k]