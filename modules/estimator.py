# modules/estimator.py — YOLOPv2 Production Version
# SAFETY-CRITICAL: Fixes class 3 width, instant directional heuristic,
# downsampled optical flow, and dt=0 division guard.

import cv2
import time
import numpy as np
from collections import defaultdict, deque


class FocalLengthCalibrator:
    """
    Dynamic Focal Length Calibration Helper.

    SAFETY-CRITICAL: The pinhole model dist = (real_width × focal_length) / pixel_width
    is only accurate when FOCAL_LENGTH_PX matches the actual camera.
    Phone cameras (DroidCam/Iriun) have variable effective focal lengths
    depending on zoom, stabilization, and virtual driver downsampling.

    Usage:
        calibrator = FocalLengthCalibrator()
        # Place a car at known distance (e.g., 10m) and measure its pixel width
        new_focal = calibrator.calibrate(
            known_width_m=1.8,
            measured_pixel_width=513,
            known_distance_m=10.0
        )
        config.FOCAL_LENGTH_PX = new_focal
    """

    @staticmethod
    def calibrate(known_width_m, measured_pixel_width, known_distance_m):
        """
        Derive focal length from a known reference measurement.

        Physics: focal_px = (pixel_width × distance) / real_width
        """
        if known_width_m <= 0 or measured_pixel_width <= 0 or known_distance_m <= 0:
            raise ValueError("All calibration inputs must be positive")
        focal = (measured_pixel_width * known_distance_m) / known_width_m
        print(f"[CALIBRATION] Computed focal length: {focal:.1f}px "
              f"(from {known_width_m}m object at {known_distance_m}m = {measured_pixel_width}px)")
        return focal


class Estimator:
    """
    Per-vehicle distance, speed, direction estimation.

    SAFETY-CRITICAL CHANGES:
    - Class 3 (motorcycle) uses REAL_MOTORCYCLE_WIDTH_M (0.7m), NOT car width (1.8m)
    - Instant directional heuristic for tracks with < 5 frames history
    - Optical flow downsampled to 320×180 (saves ~10ms/frame CPU time)
    - Division-by-zero guard on dt for speed estimation
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

        # Ego speed (downsampled optical flow)
        self._ego_speed_mps  = 0.0
        self._ego_history    = deque(maxlen=10)
        self._prev_gray      = None
        self._flow_w = getattr(config, 'EGO_FLOW_WIDTH', 320)
        self._flow_h = getattr(config, 'EGO_FLOW_HEIGHT', 180)

    # ── Distance ──────────────────────────────────────────────

    def estimate_distance(self, track_id, x1, y1, x2, y2, class_id=2):
        """
        Pinhole camera model: dist = (real_width × focal_length) / pixel_width

        SAFETY-CRITICAL FIX: Class 3 (motorcycle) now correctly uses 0.7m width.
        Previous bug: class 3 mapped to REAL_CAR_WIDTH_M (1.8m), causing
        every motorcycle distance to be overestimated by 2.5x.
        A bike at 10m was reported as 25m — lethal on Indian roads.
        """
        # SAFETY-CRITICAL: Correct width mapping
        # Class 3 = motorcycle (COCO), NOT car. Width = 0.7m, not 1.8m.
        width_map = {
            2: self.cfg.REAL_CAR_WIDTH_M,         # car: 1.8m
            7: self.cfg.REAL_TRUCK_WIDTH_M,        # truck: 2.5m
            5: self.cfg.REAL_BUS_WIDTH_M,          # bus: 2.6m
            3: getattr(self.cfg, 'REAL_MOTORCYCLE_WIDTH_M', 0.7),  # motorcycle: 0.7m
            1: self.cfg.REAL_BIKE_WIDTH_M          # bicycle: 0.7m
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

        SAFETY-CRITICAL FIX: Guard against dt=0 (Windows timer resolution ~15.6ms).
        Without this, np.clip(inf, -60, 60) = 60 → false "60 m/s approaching" signal.
        """
        now  = time.monotonic()
        hist = self._dist_history[track_id]
        hist.append((now, distance_m))

        if len(hist) < 3:
            return 0.0

        t0, d0 = hist[0]
        t1, d1 = hist[-1]
        dt     = t1 - t0

        # SAFETY-CRITICAL: Guard against dt ≈ 0
        # time.monotonic() on Windows has ~15.6ms resolution.
        # Two calls in the same tick → dt=0 → division by zero → inf → false alarm.
        if dt < 0.02:
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
        3-signal voting direction detection with Instant Directional Heuristic.

        SAFETY-CRITICAL FIX: Eliminates the 0.5s blind window on new tracks.

        OLD BEHAVIOR: len(hist) < 5 → return "unknown"
        At SKIP_FRAMES=3, 30fps → 0.5s of "unknown" direction.
        An oncoming vehicle at 80 km/h covers 11m in that window — enough to kill.

        NEW BEHAVIOR: For tracks with < 5 frames, provide a 'Preliminary' direction
        based on vertical position + early width growth rate.
        Conservative bias: uncertain vehicles near the horizon are treated as
        potentially oncoming (fail-safe).
        """
        bbox = [x1, y1, x2, y2]
        self._bbox_history[track_id].append(bbox)
        hist = self._bbox_history[track_id]

        # ── INSTANT DIRECTIONAL HEURISTIC (< 5 frames) ──────────
        # SAFETY-CRITICAL: Provides preliminary direction to eliminate blind window.
        if len(hist) < 5:
            cy = (y1 + y2) / 2
            y_ratio = cy / max(frame_h, 1)

            # With 2+ frames, check if vehicle is growing (approaching from opposite direction)
            growing = False
            if len(hist) >= 2:
                w_old = hist[0][2] - hist[0][0]
                w_new = hist[-1][2] - hist[-1][0]
                if w_old > 0:
                    growth_rate = (w_new - w_old) / max(w_old, 1)
                    growing = growth_rate > 0.05

            # Vehicles high in frame (far) + growing width = likely oncoming
            if y_ratio < 0.35 and growing:
                direction = "oncoming"
            elif y_ratio < 0.28:
                # Very high in frame = far away = conservative: assume oncoming
                direction = "oncoming"
            elif y_ratio > 0.60:
                direction = "same_direction"
            else:
                direction = "unknown"

            # Still add to temporal voting for smooth transition to full estimation
            self._dir_votes[track_id].append(direction)
            return direction

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

    # ── Ego Speed (Downsampled Optical Flow) ──────────────────

    def estimate_ego_motion(self, frame):
        """
        Optical flow ego speed — downsampled to 320×180.

        OPTIMIZATION: Full 1280×720 optical flow cost ~8-13ms per frame (CPU).
        Downsampling to 320×180 reduces this to ~1-2ms — a 6x speedup.
        The ego speed estimate is a HUD display metric, not used in any
        safety-critical decision, so reduced resolution is acceptable.
        """
        # OPTIMIZATION: Downsample BEFORE grayscale conversion
        small = cv2.resize(frame, (self._flow_w, self._flow_h),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        features = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=50,          # Reduced from 80 (smaller frame needs fewer points)
            qualityLevel=0.01,
            minDistance=10,          # Reduced from 20 (smaller frame)
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

        # Scale factor: optical flow magnitude scales with resolution
        # 320/1280 = 0.25, so multiply by 4 to compensate for downsample
        scale_factor = self.cfg.FRAME_WIDTH / self._flow_w
        ego_mps = abs(float(np.mean(filtered))) * 0.15 * scale_factor
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