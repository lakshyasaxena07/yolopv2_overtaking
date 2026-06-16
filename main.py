# main.py — YOLOPv2 Overtaking Safety System (Production Async Pipeline)
# Architecture: WatchdogCamera → DropOldestBuffer → InferenceThread → ResultQueue → HUD

import cv2, numpy as np, sys, time, queue, threading, argparse
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from collections import deque
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from modules.yolopv2_detector import YOLOPv2Detector
from modules.tracker import SORTTracker
from modules.estimator import Estimator
from modules.ttc_engine import TTCEngine, SafetyLevel
from modules.lane_path_filter import LanePathFilter

YOLOPV2_DIR = Path(r"E:\Minor 2\claude\YOLOPv2")
sys.path.insert(0, str(YOLOPV2_DIR))

# SAFETY-CRITICAL: Class 3 = motorcycle (COCO), NOT "vehicle"
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "vehicle", 5: "bus", 7: "truck"}


def select_video(cfg):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    vd = cfg.VIDEOS_FOLDER
    if not Path(vd).exists():
        vd = str(Path.home())
    p = filedialog.askopenfilename(
        title="Select Video",
        initialdir=vd,
        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")],
    )
    root.destroy()
    return p if p else None


# ════════════════════════════════════════════════════════════
# SECTION 1: WATCHDOG CAMERA (Fault Tolerance — FAIL-1 Fix)
# ════════════════════════════════════════════════════════════
class WatchdogCamera:
    """Camera with fault detection. 300ms timeout → SENSOR_FAILURE + auto-reconnect."""

    def __init__(self, src=0, width=1280, height=720, timeout_ms=300):
        self.src, self.width, self.height = src, width, height
        self.timeout_s = timeout_ms / 1000.0
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = False, None
        self.sensor_failed, self.stopped = False, False
        self._lock = threading.Lock()
        self._last_t = time.monotonic()
        self._recon_lock = threading.Lock()
        ret, f = self.cap.read()
        if ret and f is not None:
            self.frame = cv2.resize(f, (width, height))
            self.grabbed = True
            self._last_t = time.monotonic()

    def start(self):
        if not self.cap.isOpened():
            self.sensor_failed = True
            return self
        threading.Thread(target=self._grab, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()
        return self

    def _grab(self):
        while not self.stopped:
            with self._recon_lock:
                cap = self.cap
            if cap is not None and cap.isOpened():
                ret, f = cap.read()
            else:
                ret, f = False, None
            if ret and f is not None:
                f = cv2.resize(f, (self.width, self.height))
                with self._lock:
                    self.grabbed, self.frame = True, f
                    self._last_t = time.monotonic()
                    if self.sensor_failed:
                        print("[WATCHDOG] Sensor recovered!")
                        self.sensor_failed = False
            else:
                time.sleep(0.01)

    def _watchdog(self):
        while not self.stopped:
            time.sleep(0.05)
            if (
                time.monotonic() - self._last_t > self.timeout_s
                and not self.sensor_failed
            ):
                print(f"[WATCHDOG] SENSOR FAILURE")
                self.sensor_failed = True
                self._reconnect()

    def _reconnect(self):
        while not self.stopped and self.sensor_failed:
            print("[WATCHDOG] Reconnecting...")
            with self._recon_lock:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = cv2.VideoCapture(self.src)
            time.sleep(2.0)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    self._last_t = time.monotonic()
                    print("[WATCHDOG] Reconnected!")
                    return

    def read(self):
        with self._lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def release(self):
        self.stopped = True
        time.sleep(0.1)
        if self.cap:
            self.cap.release()

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def get(self, p):
        return self.cap.get(p) if self.cap else 0


# ════════════════════════════════════════════════════════════
# SECTION 2: ASYNC PIPELINE
# ════════════════════════════════════════════════════════════
class DropOldestBuffer:
    """Thread-safe frame buffer. When full, drops oldest — GPU never waits."""

    def __init__(self, maxsize=2):
        self._q = queue.Queue(maxsize=maxsize)

    def put_latest(self, frame):
        while True:
            try:
                self._q.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

    def get(self, timeout=1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


@dataclass
class PipelineResult:
    seg_frame: np.ndarray
    tracks: list
    safety_level: object
    safety_reason: str
    ego_speed: float
    lane_mode: str
    confidence: float
    inference_fps: float
    orig_shape: tuple


class TimedSafetyHistory:
    """Time-normalized 500ms safety buffer. Consistent regardless of FPS."""

    def __init__(self, window_s=0.5):
        self._w, self._entries = window_s, []

    def append(self, level, reason):
        now = time.monotonic()
        self._entries.append((now, level, reason))
        cutoff = now - self._w
        self._entries = [(t, l, r) for t, l, r in self._entries if t >= cutoff]

    def get_stable(self):
        if not self._entries:
            return SafetyLevel.UNSAFE, "Initializing..."
        self._entries = [
            (t, l, r) for t, l, r in self._entries if t >= time.monotonic() - self._w
        ]
        u = any(l == SafetyLevel.UNSAFE for _, l, _ in self._entries)
        r = any(l == SafetyLevel.RISKY for _, l, _ in self._entries)
        if u:
            return SafetyLevel.UNSAFE, next(
                r2 for _, l, r2 in reversed(self._entries) if l == SafetyLevel.UNSAFE
            )
        if r:
            return SafetyLevel.RISKY, next(
                r2 for _, l, r2 in reversed(self._entries) if l == SafetyLevel.RISKY
            )
        return SafetyLevel.SAFE, self._entries[-1][2]


# ════════════════════════════════════════════════════════════
# SECTION 3: OVERTAKING ANALYZER
# ════════════════════════════════════════════════════════════
class OvertakingAnalyzer:
    def __init__(self, config):
        self.cfg = config
        self._history = deque([True] * 5, maxlen=10)

    def analyze(self, frame_w, frame_h, tracked, path_filter=None):
        scan_y = frame_h - 100
        if path_filter is not None and path_filter.is_dynamic:
            ov_x1, ov_x2 = path_filter.get_dynamic_overtake_bounds(scan_y)
            ov_x1 = int(ov_x1 * frame_w / 1280)
            ov_x2 = int(ov_x2 * frame_w / 1280)
            on_x1, on_x2 = ov_x1, ov_x2
        else:
            ov_x1 = int(frame_w * 0.50)
            ov_x2 = int(frame_w * 0.75)
            on_x1 = ov_x1
            on_x2 = ov_x2
        for v in tracked:
            cx = (v["bbox"][0] + v["bbox"][2]) // 2
            if v.get("direction") == "oncoming" and on_x1 <= cx <= on_x2:
                if v.get("distance", 0) > 50:
                    continue
                self._history.append(False)
                return False, f"Oncoming ({v.get('distance',0):.0f}m)"
            if ov_x1 <= cx <= ov_x2:
                if v.get("distance", 0) > 50:
                    continue
                self._history.append(False)
                return False, f"Vehicle in overtake lane ({v.get('distance',0):.0f}m)"
        self._history.append(True)
        if sum(self._history) >= len(self._history) * 0.55:
            return True, "Clear to overtake"
        return False, "Checking..."


# ════════════════════════════════════════════════════════════
# SECTION 4: INFERENCE THREAD
# ════════════════════════════════════════════════════════════
class InferenceThread(threading.Thread):
    """Consumes frames from buffer, runs full pipeline, produces PipelineResult."""

    def __init__(self, frame_buf, result_q, cfg):
        super().__init__(daemon=True)
        self.buf, self.rq, self.cfg = frame_buf, result_q, cfg
        self.stopped = False
        self.show_da, self.show_ll = cfg.SHOW_DRIVABLE, cfg.SHOW_LANES
        self.detector = YOLOPv2Detector(cfg)
        self.tracker = SORTTracker()
        self.estimator = Estimator(cfg)
        self.ttc = TTCEngine(cfg)
        self.analyzer = OvertakingAnalyzer(cfg)
        self.path_filter = LanePathFilter()
        self.safety_hist = TimedSafetyHistory(
            getattr(cfg, "SAFETY_HISTORY_WINDOW_S", 0.5)
        )
        self._fps_q = deque(maxlen=30)
        self._t_prev = time.time()

    def run(self):
        while not self.stopped:
            try:
                frame = self.buf.get(timeout=0.5)
                if frame is None:
                    continue
                result = self._process(frame)
                # Drop-oldest on result queue too
                if self.rq.full():
                    try:
                        self.rq.get_nowait()
                    except:
                        pass
                self.rq.put(result)
            except Exception as e:
                import traceback

                print(f"\n[!] INFERENCE ERROR: {e}")
                traceback.print_exc()
                time.sleep(0.1)  # Prevent tight error loop

    def _process(self, frame):
        t_now = time.time()
        self._fps_q.append(1.0 / max(t_now - self._t_prev, 0.001))
        self._t_prev = t_now
        fps = float(np.mean(self._fps_q))
        orig_h, orig_w = frame.shape[:2]

        # ── OPTIMIZATION ──────────────────────────────────────
        # Only run Ego Motion (Optical Flow) every 2nd frame
        # This saves ~15% CPU/GPU overhead per cycle.
        do_ego = getattr(self, "_ego_counter", 0) % 2 == 0
        self._ego_counter = getattr(self, "_ego_counter", 0) + 1

        dets, seg_frame, orig_shape, da_mask, ll_mask = self.detector.detect(
            frame, show_da=self.show_da, show_ll=self.show_ll
        )

        self.path_filter.update(ll_mask, da_mask, 720, 1280)
        is_blind = self.path_filter.is_blind

        track_input = [
            {
                "bbox": d["bbox"],
                "confidence": d["conf"],
                "class_id": d["cls"],
                "class_name": COCO_NAMES.get(d["cls"], "vehicle"),
            }
            for d in dets
        ]
        tracks = self.tracker.update(track_input)

        tracked = []
        for t in tracks:
            tid = t["track_id"]
            x1 = int(t["bbox"][0])
            y1 = int(t["bbox"][1])
            x2 = int(t["bbox"][2])
            y2 = int(t["bbox"][3])
            cid = t.get("class_id", 2)
            dist = self.estimator.estimate_distance(tid, x1, y1, x2, y2, cid)
            spd = self.estimator.estimate_speed(tid, dist)
            dire = self.estimator.estimate_direction(tid, x1, y1, x2, y2, orig_h)
            tracked.append(
                {
                    "id": tid,
                    "bbox": [x1, y1, x2, y2],
                    "distance": dist,
                    "rel_speed_kmh": spd * 3.6,
                    "direction": dire,
                    "class_name": COCO_NAMES.get(cid, "vehicle"),
                    "cls": cid,
                }
            )

        if do_ego:
            ego_spd = self.estimator.estimate_ego_motion(frame)
        else:
            ego_spd = self.estimator.get_ego_speed()

        feasible, reason = self.analyzer.analyze(
            orig_w, orig_h, tracked, self.path_filter
        )

        # Build enriched tracks with kinematic TTC (replaces static gap)
        enriched = []
        for v in tracked:
            bbox = v["bbox"]
            if self.path_filter.is_right_of_corridor(bbox, orig_h, orig_w):
                continue
            on_road = self.path_filter.is_on_drivable(bbox, da_mask, orig_h, orig_w)
            if v["direction"] == "oncoming":
                zone = "oncoming_lane"
            else:
                zone = self.path_filter.classify_zone(bbox, orig_h, orig_w)

            # SAFETY-CRITICAL: Kinematic TTC replaces static 25-30m gap
            # rel_speed_kmh > 0 = approaching, < 0 = receding
            # DO NOT use abs() — receding vehicles must NOT trigger TTC
            approach_mps = (
                max(v["rel_speed_kmh"], 0.0) / 3.6
            )  # Only positive = approaching
            if approach_mps > 0.1:
                vehicle_ttc = v["distance"] / approach_mps
            else:
                vehicle_ttc = float("inf")

            is_oncoming = v["direction"] == "oncoming"
            ttc_thresh = (
                self.cfg.TTC_UNSAFE_ONCOMING
                if is_oncoming
                else self.cfg.TTC_UNSAFE_SAME_DIR
            )

            enriched.append(
                {
                    "track_id": v["id"],
                    "bbox": v["bbox"],
                    "distance_m": v["distance"],
                    "speed_kph": v["rel_speed_kmh"],
                    "direction": v["direction"],
                    "class_name": v["class_name"],
                    "is_oncoming": is_oncoming,
                    "is_relevant": on_road,
                    "is_too_close": vehicle_ttc < ttc_thresh
                    and zone in ("ego_lane", "overtake_lane"),
                    "is_critical": v["distance"] < 8.0,
                    "is_parked": False,
                    "approach_rate": approach_mps,
                    "zone": zone,
                    "ttc": vehicle_ttc,
                }
            )

        ttc_dec = self.ttc.evaluate(enriched, None)
        critical = [v for v in enriched if v.get("is_critical")]
        too_close = [v for v in enriched if v.get("is_too_close")]

        if critical:
            raw_safety, raw_reason = SafetyLevel.UNSAFE, "CRITICAL | Brake Now"
        elif not feasible:
            raw_safety, raw_reason = SafetyLevel.UNSAFE, f"NO OVERTAKE | {reason}"
        elif ttc_dec.level == SafetyLevel.UNSAFE:
            raw_safety, raw_reason = SafetyLevel.UNSAFE, ttc_dec.reason
        elif too_close:
            raw_safety, raw_reason = (
                SafetyLevel.RISKY,
                "CAUTION | TTC too low for maneuver",
            )
        elif ttc_dec.level == SafetyLevel.RISKY:
            raw_safety, raw_reason = SafetyLevel.RISKY, ttc_dec.reason
        elif is_blind:
            ov_occ = any(v.get("zone") == "overtake_lane" for v in enriched)
            if ov_occ:
                raw_safety, raw_reason = (
                    SafetyLevel.RISKY,
                    "CAUTION | Overtake Occupied (Blind)",
                )
            else:
                raw_safety, raw_reason = (
                    SafetyLevel.SAFE,
                    "SAFE | Blind Driving (Lines Lost)",
                )
        else:
            raw_safety, raw_reason = SafetyLevel.SAFE, "SAFE TO OVERTAKE"

        # Time-normalized smoothing (500ms window)
        self.safety_hist.append(raw_safety, raw_reason)
        final_safety, final_reason = self.safety_hist.get_stable()

        self.estimator.cleanup({t["id"] for t in tracked})

        # Scale bboxes to HUD (1280x720)
        disp_h, disp_w = seg_frame.shape[:2]
        sx, sy = disp_w / max(orig_w, 1), disp_h / max(orig_h, 1)
        scaled = []
        for v in tracked:
            sv = v.copy()
            b = v["bbox"]
            sv["bbox"] = [
                int(b[0] * sx),
                int(b[1] * sy),
                int(b[2] * sx),
                int(b[3] * sy),
            ]
            scaled.append(sv)

        return PipelineResult(
            seg_frame=seg_frame,
            tracks=scaled,
            safety_level=final_safety,
            safety_reason=final_reason,
            ego_speed=ego_spd,
            lane_mode=self.path_filter.mode_display,
            confidence=self.path_filter.confidence_score,
            inference_fps=fps,
            orig_shape=(orig_h, orig_w),
        )


# ════════════════════════════════════════════════════════════
# SECTION 5: HUD RENDERER
# ════════════════════════════════════════════════════════════
class HUDRenderer:
    INFO_COL = (0, 255, 255) # Yellow
    SAFE_COL = (0, 220, 0) #Green
    CAUTION_COL = (0, 165, 255) #Orange
    UNSAFE_COL = (0, 0, 255) #Red

    def render(
        self,
        frame,
        fps,
        tracks,
        ego_speed,
        safety,
        reason,
        cfg,
        lane_mode="OPTICAL",
        confidence=1.0,
        separate_window=False,
    ):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 52), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            self.INFO_COL,
            2,
        )
        cv2.putText(
            frame,
            f"Tracks: {len(tracks)}",
            (170, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            self.INFO_COL,
            2,
        )
    #     cv2.putText(
    #     frame,
    #     "---------Your Custom Text-----------",
    #     (100, 200),          # x, y position (top-left corner)
    #     cv2.FONT_HERSHEY_SIMPLEX,  # font type
    #     1.0,                 # font size
    #     (255, 255, 0),       # color (B, G, R format)
    #     2                    # thickness
    # )

        cv2.putText(
            frame,
            f"Ego: {ego_speed*3.6:.1f} km/h",
            (360, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            self.INFO_COL,
            2,
        )

        mode_txt = f"MODE: {lane_mode}"
        if "VIRTUAL" in lane_mode:
            mode_col = (0, 0, 255)
        elif "MEMORY" in lane_mode:
            mode_col = (0, 140, 255)
        elif "FALLBACK" in lane_mode:
            mode_col = (0, 0, 200)
        else:
            mode_col = (0, 220, 0)
        mt_size = cv2.getTextSize(mode_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        mx = w - mt_size[0] - 12
        cv2.rectangle(frame, (mx - 6, 10), (w - 4, 42), (30, 30, 30), -1)
        cv2.putText(
            frame, mode_txt, (mx, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_col, 2
        )

        for v in tracks:
            x1, y1, x2, y2 = (
                int(v["bbox"][0]),
                int(v["bbox"][1]),
                int(v["bbox"][2]),
                int(v["bbox"][3]),
            )
            dist = v.get("distance", 0)
            spd = v.get("rel_speed_kmh", 0)
            cls = v.get("class_name", "vehicle")
            tid = v.get("id", 0)
            dire = v.get("direction", "")
            col = (0, 0, 220) if dire == "oncoming" else (0, 220, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            spd_s = f"+{spd:.0f}" if spd > 0 else f"{spd:.0f}"
            lbl = f"#{tid} {cls} {dist:.0f}m {spd_s}kph"
            lx, ly = max(x1, 4), max(y1 - 6, 22)
            cv2.rectangle(
                frame, (lx - 2, ly - 18), (lx + len(lbl) * 9, ly + 4), (15, 15, 15), -1
            )
            cv2.putText(
                frame, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2
            )

        if safety == SafetyLevel.SAFE:
            bcol, txt, tcol = (0, 130, 0), "SAFE", self.SAFE_COL
        elif safety == SafetyLevel.RISKY:
            bcol, txt, tcol = (0, 100, 160), "CAUTION", self.CAUTION_COL
        else:
            bcol, txt, tcol = (0, 0, 140), "UNSAFE", self.UNSAFE_COL

        if separate_window:
            banner = np.zeros((85, w, 3), dtype=np.uint8)
            cv2.rectangle(banner, (0, 0), (w, 85), bcol, -1)
            ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0]
            cv2.putText(
                banner,
                txt,
                ((w - ts[0]) // 2, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                1.8,
                tcol,
                3,
            )
            rs = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(
                banner,
                reason,
                ((w - rs[0]) // 2, 73),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
            )
        else:
            cv2.rectangle(frame, (0, h - 85), (w, h), bcol, -1)
            ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0]
            cv2.putText(
                frame,
                txt,
                ((w - ts[0]) // 2, h - 45),
                cv2.FONT_HERSHEY_DUPLEX,
                1.8,
                tcol,
                3,
            )
            rs = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(
                frame,
                reason,
                ((w - rs[0]) // 2, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
            )
            banner = None
        cv2.putText(
            frame,
            "D=drivable  L=lanes  R=restart  Q=quit",
            (w - 400, h - 90 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 170, 170),
            1,
        )
        return frame, banner

    def render_sensor_failure(self, frame, separate_window=False):
        """Red fullscreen SENSOR FAILURE overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        txt = "SENSOR FAILURE"
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 2.0, 4)[0]
        cv2.putText(
            frame,
            txt,
            ((w - ts[0]) // 2, (h + ts[1]) // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            2.0,
            (255, 255, 255),
            4,
        )
        cv2.putText(
            frame,
            "DRIVE MANUALLY — Reconnecting...",
            ((w - 450) // 2, (h + ts[1]) // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )
        if separate_window:
            banner = np.zeros((85, w, 3), dtype=np.uint8)
            cv2.rectangle(banner, (0, 0), (w, 85), (0, 0, 180), -1)
            cv2.putText(
                banner,
                txt,
                ((w - ts[0]) // 2, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                1.8,
                (255, 255, 255),
                3,
            )
        else:
            banner = None
        return frame, banner


# ════════════════════════════════════════════════════════════
# SECTION 6: MAIN RUN LOOP
# ════════════════════════════════════════════════════════════
def run(video_path, cfg, separate_window=False):
    hud = HUDRenderer()
    frame_buf = DropOldestBuffer(maxsize=2)
    result_q = queue.Queue(maxsize=1)

    if isinstance(video_path, int):
        print(f"Live Camera (Index {video_path}) with Watchdog...")
        cam = WatchdogCamera(
            video_path, timeout_ms=getattr(cfg, "WATCHDOG_TIMEOUT_MS", 300)
        ).start()
    else:
        cam = cv2.VideoCapture(video_path)

    if not cam.isOpened():
        print(f"Video open nahi hui: {video_path}")
        return False

    video_fps = cam.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120:
        video_fps = 30.0
    print(f"Video FPS: {video_fps:.1f}")

    # Start inference thread
    inf_thread = InferenceThread(frame_buf, result_q, cfg)
    inf_thread.start()

    cv2.namedWindow("YOLOPv2 Overtaking Safety", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOPv2 Overtaking Safety", cfg.OUTPUT_WIDTH, cfg.OUTPUT_HEIGHT)

    if separate_window:
        cv2.namedWindow("Safety Alert", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Safety Alert", cfg.OUTPUT_WIDTH, 85)

    last_result = None
    paused = False
    # Blank frame for sensor failure overlay
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)

    name = (
        Path(video_path).name
        if isinstance(video_path, str)
        else f"Live Camera ({video_path})"
    )
    print(f"\nStarting: {name}")
    print("D=drivable  L=lanes  Space=pause  R=restart  Q=quit\n")

    is_cam = isinstance(video_path, int)  # CRASH-1 fix: define BEFORE loop
    frame_interval = 1.0 / max(cfg.TARGET_FPS, 1) if not is_cam else 0
    last_read_time = 0.0

    while True:
        loop_start = time.time()

        if not paused:
            # Check sensor failure (live camera only)
            if is_cam and hasattr(cam, "sensor_failed") and cam.sensor_failed:
                display, banner = hud.render_sensor_failure(blank.copy(), separate_window)
                cv2.imshow("YOLOPv2 Overtaking Safety", display)
                if separate_window and banner is not None:
                    cv2.imshow("Safety Alert", banner)
                if (cv2.waitKey(100) & 0xFF) in [ord("q"), ord("Q"), 27]:
                    break
                continue

            # PERF-1 fix: Throttle video reads to TARGET_FPS
            # Without this, video file is consumed in seconds
            if not is_cam and getattr(cfg, "SYNC_VIDEO", True):
                now = time.time()
                if now - last_read_time < frame_interval:
                    # Don't read a new frame yet — just check for results
                    try:
                        last_result = result_q.get_nowait()
                    except queue.Empty:
                        pass
                    if last_result is not None:
                        display = last_result.seg_frame.copy()
                        display, banner = hud.render(
                            display,
                            last_result.inference_fps,
                            last_result.tracks,
                            last_result.ego_speed,
                            last_result.safety_level,
                            last_result.safety_reason,
                            cfg,
                            lane_mode=last_result.lane_mode,
                            confidence=last_result.confidence,
                            separate_window=separate_window,
                        )
                        cv2.imshow("YOLOPv2 Overtaking Safety", display)
                        if separate_window and banner is not None:
                            cv2.imshow("Safety Alert", banner)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), ord("Q"), 27]:
                        inf_thread.stopped = True
                        if hasattr(cam, "release"):
                            cam.release()
                        cv2.destroyAllWindows()
                        return False
                    elif key == ord(" "):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif key in [ord("d"), ord("D")]:
                        inf_thread.show_da = not inf_thread.show_da
                    elif key in [ord("l"), ord("L")]:
                        inf_thread.show_ll = not inf_thread.show_ll
                    continue
                last_read_time = now

            ret, frame = cam.read()
            if not ret:
                if isinstance(video_path, str):
                    print("Video ended.")
                    break
                continue

            # Feed frame to async pipeline (drop-oldest)
            frame_buf.put_latest(frame)

            # Get latest inference result (non-blocking)
            try:
                last_result = result_q.get_nowait()
            except queue.Empty:
                pass

            # Render
            if last_result is not None:
                display = last_result.seg_frame.copy()
                display, banner = hud.render(
                    display,
                    last_result.inference_fps,
                    last_result.tracks,
                    last_result.ego_speed,
                    last_result.safety_level,
                    last_result.safety_reason,
                    cfg,
                    lane_mode=last_result.lane_mode,
                    confidence=last_result.confidence,
                    separate_window=separate_window,
                )
            else:
                display = cv2.resize(frame, (1280, 720))
                if separate_window:
                    banner = np.zeros((85, 1280, 3), dtype=np.uint8)
                else:
                    banner = None

            cv2.imshow("YOLOPv2 Overtaking Safety", display)
            if separate_window and banner is not None:
                cv2.imshow("Safety Alert", banner)

        # Use waitKey for timing — keeps UI responsive (CRASH-5 fix)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            inf_thread.stopped = True
            if hasattr(cam, "release"):
                cam.release()
            cv2.destroyAllWindows()
            return False
        elif key in [ord("r"), ord("R")]:
            if isinstance(video_path, str):
                inf_thread.stopped = True
                if hasattr(cam, "release"):
                    cam.release()
                cv2.destroyAllWindows()
                return True
        elif key == ord(" "):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key in [ord("d"), ord("D")]:
            inf_thread.show_da = not inf_thread.show_da
            print(f"Drivable: {'ON' if inf_thread.show_da else 'OFF'}")
        elif key in [ord("l"), ord("L")]:
            inf_thread.show_ll = not inf_thread.show_ll
            print(f"Lanes: {'ON' if inf_thread.show_ll else 'OFF'}")

    inf_thread.stopped = True
    if hasattr(cam, "release"):
        cam.release()
    cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    # while True:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, help="Camera index")
    parser.add_argument("--sep", action="store_true", help="Separate alert window")
    args = parser.parse_args()
    cfg = Config()
    print("=" * 55)
    print("  YOLOPv2 Overtaking Safety — PRODUCTION PIPELINE")
    print("=" * 55)
    
    separate_window = args.sep
    if not separate_window and not any(arg in sys.argv for arg in ['--sep']):
        ans = input("Do you want a separate alert window? [y/N]: ").strip().lower()
        separate_window = (ans == 'y')

    if args.cam is not None:
        video = args.cam
    else:
        mode = input("Choose: [1] Video | [2] Laptop Cam | [3] Phone Cam: ")
        if mode == "1":
            video = select_video(cfg)
        elif mode == "2":
            video = 0
        elif mode == "3":
            ci = input("Phone Camera Index (1 for droidcam, 2 for webcam): ")
            try:
                video = int(ci)
            except:
                video = 1
        else:
            print("Invalid.")
            sys.exit()
    if video is not None:
        run(video, cfg, separate_window=separate_window)
    print("\nGoodbye!")
    cv2.destroyAllWindows()
