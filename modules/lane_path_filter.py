# modules/lane_path_filter.py — Dynamic Lane-Based Path Filtering
# Replaces hardcoded pixel-ratio zone logic with contour-based ego-lane detection
# Includes Blind-Mode fallback with temporal memory and drivable-area proxy

import cv2
import numpy as np


class LanePathFilter:
    """
    Dynamic path filter using YOLOPv2 lane-line mask.
    
    Identifies ego-lane boundaries from the red lane-line contours,
    fits slope lines for horizon projection, and provides:
      - Right-side corridor filtering
      - Drivable-area (green mask) gating
      - Dynamic zone classification (ego / overtake / right_shoulder)
    
    Blind-Mode Fallback:
      - 30-frame temporal memory with progressive widening when lines are lost
      - Drivable-area proxy boundaries when memory expires
      - High-sensitivity gap thresholds and forced CAUTION state
    
    Falls back to fixed pixel ratios only as the absolute last resort.
    """

    # ── Mode constants ────────────────────────────────────────
    MODE_OPTICAL = "OPTICAL"       # Real lane lines detected this frame
    MODE_MEMORY  = "BLIND/MEMORY"  # Using remembered lines (temporal buffer)
    MODE_VIRTUAL = "BLIND/VIRTUAL" # Using drivable-area proxy edges
    MODE_FALLBACK = "BLIND/FALLBACK"  # Absolute fallback (fixed ratios)

    # ── Blind-Mode tuning ─────────────────────────────────────
    MEMORY_FRAMES = 45                # Raised from 30 for smoother transition
    WIDEN_PERCENT_PER_FRAME = 0.03    # 3% of original span per lost frame
    CONFIDENCE_DECAY_RATE = 1.0 / 45  # Decays to 0.0 over MEMORY_FRAMES
    BLIND_GAP_THRESHOLD = 30.0        # Reduced from 35m (safer but less panicky)

    def __init__(self, fallback_ego_ratio=0.50, fallback_right_ratio=0.75):
        # Fallback ratios (current hardcoded behavior)
        self._fb_ego = fallback_ego_ratio
        self._fb_right = fallback_right_ratio

        # Fitted line parameters: x = x0 + (y - y0) / slope_m
        # Left ego boundary
        self._left_x0 = None
        self._left_y0 = None
        self._left_slope = None   # dy/dx from fitLine

        # Right ego boundary
        self._right_x0 = None
        self._right_y0 = None
        self._right_slope = None

        # Frame dimensions (HUD space: 720x1280)
        self._h = 720
        self._w = 1280

        # Whether dynamic lines were found THIS frame
        self._dynamic = False

        # ── Temporal Memory (Blind-Mode) ──────────────────────
        self._memory_left = None   # (x0, y0, slope) from last valid frame
        self._memory_right = None
        self._memory_span = 0.0    # Original left-right span at save time
        self._frames_since_valid = 0  # Counter since last optical detection
        self._mode = self.MODE_FALLBACK  # Current operating mode
        self._confidence_score = 1.0    # 1.0 = full confidence, decays in Memory

        # ── Drivable-Area Proxy cache ─────────────────────────
        self._virtual_left_x = None
        self._virtual_right_x = None

    @property
    def mode(self):
        """Current operating mode string for HUD display."""
        return self._mode

    @property
    def mode_display(self):
        """Formatted mode string with confidence for HUD badge."""
        if self._mode == self.MODE_OPTICAL:
            return "OPTICAL (100%)"
        elif self._mode == self.MODE_MEMORY:
            pct = int(self._confidence_score * 100)
            return f"MEMORY ({pct}%)"
        elif self._mode == self.MODE_VIRTUAL:
            return "VIRTUAL (GREEN-MASK)"
        else:
            return "FALLBACK (0%)"

    @property
    def confidence_score(self):
        """Internal confidence score (1.0 = optical, decays toward 0.0)."""
        return self._confidence_score

    @property
    def is_blind(self):
        """True when operating in any Blind sub-mode."""
        return self._mode in (self.MODE_MEMORY, self.MODE_VIRTUAL, self.MODE_FALLBACK)

    @property
    def blind_gap_threshold(self):
        """Raised gap threshold for Blind-Mode (35m vs 25m)."""
        return self.BLIND_GAP_THRESHOLD

    def update(self, ll_mask, da_mask, frame_h=720, frame_w=1280):
        """
        Process the lane-line mask to find ego-lane boundaries.
        Implements the full Optical → Memory → Virtual → Fallback cascade.
        
        Args:
            ll_mask: Binary lane-line mask (720x1280), 0/1 int array
            da_mask: Binary drivable-area mask (720x1280), 0/1 int array
            frame_h: Mask height (720)
            frame_w: Mask width (1280)
        """
        self._h = frame_h
        self._w = frame_w
        self._dynamic = False

        # Try optical detection first
        optical_ok = self._try_optical(ll_mask, frame_h, frame_w)

        if optical_ok:
            # ── MODE: OPTICAL ─────────────────────────────────
            self._mode = self.MODE_OPTICAL
            self._dynamic = True
            self._frames_since_valid = 0
            self._confidence_score = 1.0
            # Save to memory buffer (including span for %-based widening)
            self._memory_left = (self._left_x0, self._left_y0, self._left_slope)
            self._memory_right = (self._right_x0, self._right_y0, self._right_slope)
            self._memory_span = abs(self._right_x0 - self._left_x0)
            return

        # Optical failed — increment lost counter
        self._frames_since_valid += 1

        if (self._memory_left is not None and
                self._frames_since_valid <= self.MEMORY_FRAMES):
            # ── MODE: MEMORY (widening) ───────────────────────
            self._mode = self.MODE_MEMORY
            self._apply_memory_with_widening()
            self._dynamic = True
            return

        # Memory expired — try drivable-area proxy
        virtual_ok = self._try_drivable_proxy(da_mask, frame_h, frame_w)
        if virtual_ok:
            # ── MODE: VIRTUAL ─────────────────────────────────
            self._mode = self.MODE_VIRTUAL
            self._dynamic = True
            return

        # ── MODE: FALLBACK (fixed ratios) ─────────────────────
        self._mode = self.MODE_FALLBACK
        self._dynamic = False

    def _try_optical(self, ll_mask, frame_h, frame_w):
        """
        Attempt to detect lane-line contours from the red mask.
        Returns True if successful and sets _left/_right parameters.
        """
        if ll_mask is None:
            return False

        # Convert to uint8 for contour detection
        mask_u8 = (ll_mask * 255).astype(np.uint8)

        # Morphological cleanup — connect broken lane segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            return False

        # Filter out tiny noise contours (need sufficient points for fitLine)
        valid_contours = [c for c in contours if len(c) >= 20]
        if len(valid_contours) < 2:
            return False

        # Scan row: near the bottom but above HUD bar (frame_h - 100)
        scan_y = frame_h - 100
        center_x = frame_w // 2

        # For each contour, find its x-coordinate at the scan row
        contour_x_at_scan = []
        for c in valid_contours:
            pts = c.reshape(-1, 2)
            near_mask = np.abs(pts[:, 1] - scan_y) < 40
            if np.sum(near_mask) < 3:
                continue
            near_pts = pts[near_mask]
            avg_x = np.mean(near_pts[:, 0])
            contour_x_at_scan.append((avg_x, c))

        if len(contour_x_at_scan) < 2:
            return False

        # Sort by x-coordinate
        contour_x_at_scan.sort(key=lambda item: item[0])

        # Find the two lines closest to center
        left_candidates = [(x, c) for x, c in contour_x_at_scan if x <= center_x]
        right_candidates = [(x, c) for x, c in contour_x_at_scan if x > center_x]

        if left_candidates and right_candidates:
            left_line = left_candidates[-1]
            right_line = right_candidates[0]
        elif len(contour_x_at_scan) >= 2:
            dists = [(abs(x - center_x), x, c) for x, c in contour_x_at_scan]
            dists.sort(key=lambda item: item[0])
            pair = sorted(dists[:2], key=lambda item: item[1])
            left_line = (pair[0][1], pair[0][2])
            right_line = (pair[1][1], pair[1][2])
        else:
            return False

        # Fit lines using cv2.fitLine
        left_fit = self._fit_line(left_line[1])
        right_fit = self._fit_line(right_line[1])

        if left_fit is None or right_fit is None:
            return False

        self._left_x0, self._left_y0, self._left_slope = left_fit
        self._right_x0, self._right_y0, self._right_slope = right_fit
        return True

    def _apply_memory_with_widening(self):
        """
        Use the last valid boundaries but progressively widen them
        by 5% of original span per frame, with confidence decay.
        """
        lx0, ly0, ls = self._memory_left
        rx0, ry0, rs = self._memory_right

        # Confidence Decay: drops linearly from 1.0 → 0.0 over MEMORY_FRAMES
        self._confidence_score = max(
            0.0, 1.0 - self._frames_since_valid * self.CONFIDENCE_DECAY_RATE
        )

        # Widen: 5% of original span per lost frame (percentage-based)
        span = max(self._memory_span, 100.0)  # Floor to avoid zero-span edge case
        widen = self._frames_since_valid * self.WIDEN_PERCENT_PER_FRAME * span

        self._left_x0 = lx0 - widen
        self._left_y0 = ly0
        self._left_slope = ls

        self._right_x0 = rx0 + widen
        self._right_y0 = ry0
        self._right_slope = rs

    def _try_drivable_proxy(self, da_mask, frame_h, frame_w):
        """
        Drivable-Edge Proxy: find left-most and right-most non-zero pixels
        of the da_mask (Green) in the bottom 10% of the frame.
        Uses these as 'Virtual Lane Lines'.
        """
        if da_mask is None:
            return False

        # Bottom 10% of the frame
        band_top = int(frame_h * 0.9)
        band = da_mask[band_top:frame_h, :]

        if band.size == 0:
            return False

        # Find columns that have drivable pixels
        col_sums = np.sum(band, axis=0)
        drivable_cols = np.where(col_sums > 0)[0]

        if len(drivable_cols) < 20:
            return False  # Not enough drivable area

        left_edge = int(drivable_cols[0])
        right_edge = int(drivable_cols[-1])

        if right_edge - left_edge < 50:
            return False  # Too narrow to be meaningful

        # Store as virtual boundaries (use vertical slope = very steep)
        # These don't converge toward horizon — they're flat columns
        center_y = (band_top + frame_h) // 2
        self._left_x0 = float(left_edge)
        self._left_y0 = float(center_y)
        self._left_slope = 50.0   # Near-vertical: minimal convergence

        self._right_x0 = float(right_edge)
        self._right_y0 = float(center_y)
        self._right_slope = -50.0  # Near-vertical: minimal convergence

        self._virtual_left_x = left_edge
        self._virtual_right_x = right_edge

        # Virtual mode = zero optical confidence
        self._confidence_score = 0.0
        return True

    def _fit_line(self, contour):
        """
        Fit a line to a contour using cv2.fitLine.
        Returns (x0, y0, slope) where slope = vy/vx, or None on failure.
        """
        if len(contour) < 10:
            return None

        try:
            vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        except Exception:
            return None

        # slope = vy/vx (rise/run)
        if abs(vx) < 1e-6:
            slope = 1e6 if vy > 0 else -1e6
        else:
            slope = vy / vx

        return (float(x0), float(y0), float(slope))

    def get_ego_boundaries(self, y):
        """
        Get the ego-lane left and right x-boundaries at a given y-coordinate.
        Uses slope projection: x = x0 + (y - y0) / slope
        
        Args:
            y: Y-coordinate in HUD space (0=top, 720=bottom)
            
        Returns:
            (left_x, right_x) in HUD pixel coordinates
        """
        if not self._dynamic:
            # Fallback to fixed ratios
            return (int(self._w * (1.0 - self._fb_right)),
                    int(self._w * self._fb_ego))

        left_x = self._project_x(self._left_x0, self._left_y0, self._left_slope, y)
        right_x = self._project_x(self._right_x0, self._right_y0, self._right_slope, y)

        # Clamp to frame
        left_x = max(0, min(self._w - 1, int(left_x)))
        right_x = max(0, min(self._w - 1, int(right_x)))

        # Sanity: left must be < right
        if left_x >= right_x:
            return (int(self._w * (1.0 - self._fb_right)),
                    int(self._w * self._fb_ego))

        return (left_x, right_x)

    def _project_x(self, x0, y0, slope, y):
        """Project the fitted line to get x at a given y."""
        if abs(slope) < 1e-6:
            return x0
        return x0 + (y - y0) / slope

    def is_right_of_corridor(self, bbox, orig_h, orig_w):
        """
        Check if a detection's bottom-center is to the right of the right ego boundary.
        
        Args:
            bbox: [x1, y1, x2, y2] in ORIGINAL frame coordinates
            orig_h, orig_w: Original frame dimensions
            
        Returns:
            True if the detection should be filtered out (right of corridor)
        """
        bc_x, bc_y = self._bbox_to_hud(bbox, orig_h, orig_w)
        _, right_x = self.get_ego_boundaries(bc_y)
        return bc_x > right_x

    def is_on_drivable(self, bbox, da_mask, orig_h, orig_w):
        """
        5x5 area vote on the drivable-area (green) mask.
        
        Args:
            bbox: [x1, y1, x2, y2] in ORIGINAL frame coordinates
            da_mask: Binary drivable mask (720x1280)
            orig_h, orig_w: Original frame dimensions
            
        Returns:
            True if the vehicle is on drivable area
        """
        if da_mask is None:
            return True

        bc_x, bc_y = self._bbox_to_hud(bbox, orig_h, orig_w)

        x_min = max(0, bc_x - 2)
        x_max = min(self._w, bc_x + 3)
        y_min = max(0, bc_y - 2)
        y_max = min(self._h, bc_y + 3)

        roi = da_mask[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return False

        return (np.sum(roi) / roi.size) > 0.40

    def classify_zone(self, bbox, orig_h, orig_w):
        """
        Classify which zone the detection falls in, using dynamic boundaries.
        
        Args:
            bbox: [x1, y1, x2, y2] in ORIGINAL frame coordinates
            orig_h, orig_w: Original frame dimensions
            
        Returns:
            "ego_lane", "overtake_lane", or "right_shoulder"
        """
        bc_x, bc_y = self._bbox_to_hud(bbox, orig_h, orig_w)

        left_x, right_x = self.get_ego_boundaries(bc_y)

        if bc_x > right_x:
            return "right_shoulder"
        elif bc_x < left_x:
            return "overtake_lane"
        else:
            return "ego_lane"

    def get_dynamic_overtake_bounds(self, y):
        """
        Get the overtake-lane boundaries for OvertakingAnalyzer.
        Overtake zone = area to the left of the left ego boundary.
        
        Returns:
            (ov_x1, ov_x2) — the overtake lane x-range at the given y
        """
        left_x, right_x = self.get_ego_boundaries(y)

        if not self._dynamic:
            return (int(self._w * 0.50), int(self._w * 0.75))

        ov_x2 = left_x
        ov_x1 = max(0, left_x - int(self._w * 0.25))

        return (ov_x1, ov_x2)

    def _bbox_to_hud(self, bbox, orig_h, orig_w):
        """
        Convert bbox bottom-center from original coordinates to HUD (720x1280) coordinates.
        """
        cx = (bbox[0] + bbox[2]) // 2
        by = bbox[3]

        hud_x = min(max(int(cx * 1280 / orig_w), 0), 1279)
        hud_y = min(max(int(by * 720 / orig_h), 0), 719)

        return (hud_x, hud_y)

    @property
    def is_dynamic(self):
        """Whether dynamic lane boundaries are active (optical, memory, or virtual)."""
        return self._dynamic
