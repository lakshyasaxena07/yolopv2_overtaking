# main.py — YOLOPv2 Overtaking Safety System
# Controls: D=drivable  L=lanes  Space=pause  R=restart  Q=quit

import cv2
import numpy as np
import sys
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from collections import deque
import threading
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from modules.yolopv2_detector import YOLOPv2Detector
from modules.tracker import SORTTracker
from modules.estimator import Estimator
from modules.ttc_engine import TTCEngine, SafetyLevel
from modules.lane_path_filter import LanePathFilter

YOLOPV2_DIR = Path(r"E:\Minor 2\claude\YOLOPv2")
sys.path.insert(0, str(YOLOPV2_DIR))

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "vehicle", 5: "bus", 7: "truck"}


def select_video(cfg):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    videos_dir = cfg.VIDEOS_FOLDER
    if not Path(videos_dir).exists():
        videos_dir = str(Path.home())
    path = filedialog.askopenfilename(
        title="Select Video",
        initialdir=videos_dir,
        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")],
    )
    root.destroy()
    return path if path else None


class OvertakingAnalyzer:

    def __init__(self, config):
        self.cfg = config
        self._history = deque([True] * 5, maxlen=10)

    def analyze(self, frame_w, frame_h, tracked, path_filter=None):
        """Analyze overtaking feasibility using dynamic lane boundaries."""
        # Use dynamic boundaries from path_filter if available
        scan_y = frame_h - 100  # Same scan row as LanePathFilter
        if path_filter is not None and path_filter.is_dynamic:
            ov_x1, ov_x2 = path_filter.get_dynamic_overtake_bounds(scan_y)
            # Scale from HUD (1280) to original frame width
            ov_x1 = int(ov_x1 * frame_w / 1280)
            ov_x2 = int(ov_x2 * frame_w / 1280)
            on_x1, on_x2 = ov_x1, ov_x2
        else:
            # Fallback: hardcoded ratios
            if self.cfg.DRIVING_MODE == "india":
                ov_x1 = int(frame_w * 0.50)
                ov_x2 = int(frame_w * 0.75)
                on_x1 = int(frame_w * 0.50)
                on_x2 = int(frame_w * 0.75)
            else:
                ov_x1 = int(frame_w * 0.50)
                ov_x2 = int(frame_w * 0.75)
                on_x1 = 0
                on_x2 = int(frame_w * 0.25)

        for v in tracked:
            cx = (v["bbox"][0] + v["bbox"][2]) // 2
            if v.get("direction") == "oncoming" and on_x1 <= cx <= on_x2:
                if v.get("distance", 0) > 50:
                    continue
                self._history.append(False)
                return (False, f"Oncoming ({v.get('distance',0):.0f}m)")
            if ov_x1 <= cx <= ov_x2:
                if v.get("distance", 0) > 50:
                    continue
                self._history.append(False)
                return (
                    False,
                    f"Vehicle in overtake lane ({v.get('distance',0):.0f}m)",
                )

        self._history.append(True)
        if sum(self._history) >= len(self._history) * 0.55:
            return True, "Clear to overtake"
        return False, "Checking..."


class HUDRenderer:

    INFO_COL = (0, 255, 255)
    SAFE_COL = (0, 220, 0)
    CAUTION_COL = (0, 165, 255)
    UNSAFE_COL = (0, 0, 255)

    def render(self, frame, fps, tracks, ego_speed,
               safety, reason, cfg, lane_mode="OPTICAL", confidence=1.0):
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0,0), (w,52),
                      (20,20,20), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (12,36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, self.INFO_COL, 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}",
                    (170,36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, self.INFO_COL, 2)
        cv2.putText(frame,
                    f"Ego: {ego_speed*3.6:.1f} km/h",
                    (360,36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, self.INFO_COL, 2)

        # ── Mode indicator (top-right) ────────────────────
        mode_txt = f"MODE: {lane_mode}"
        if "VIRTUAL" in lane_mode:
            mode_col = (0, 0, 255)    # Red for virtual/green-mask
        elif "MEMORY" in lane_mode:
            mode_col = (0, 140, 255)  # Orange for memory
        elif "FALLBACK" in lane_mode:
            mode_col = (0, 0, 200)    # Dark red for fallback
        else:
            mode_col = (0, 220, 0)    # Green for optical
        mt_size = cv2.getTextSize(mode_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        mx = w - mt_size[0] - 12
        # Background pill for mode badge
        cv2.rectangle(frame, (mx - 6, 10), (w - 4, 42), (30, 30, 30), -1)
        cv2.putText(frame, mode_txt, (mx, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_col, 2)

        for v in tracks:
            x1 = int(v["bbox"][0])
            y1 = int(v["bbox"][1])
            x2 = int(v["bbox"][2])
            y2 = int(v["bbox"][3])
            dist = v.get("distance", 0)
            spd  = v.get("rel_speed_kmh", 0)
            cls  = v.get("class_name", "vehicle")
            tid  = v.get("id", 0)
            dire = v.get("direction", "")

            col = (0,0,220) if dire == "oncoming" \
                  else (0,220,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)

            spd_s = f"+{spd:.0f}" if spd > 0 \
                    else f"{spd:.0f}"
            lbl   = (f"#{tid} {cls} "
                     f"{dist:.0f}m {spd_s}kph")
            lx = max(x1, 4)
            ly = max(y1-6, 22)
            cv2.rectangle(frame,
                (lx-2, ly-18),
                (lx+len(lbl)*9, ly+4),
                (15,15,15), -1)
            cv2.putText(frame, lbl, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,255,100), 2)

        if safety == SafetyLevel.SAFE:
            bcol, txt, tcol = \
                (0,130,0), "SAFE", self.SAFE_COL
        elif safety == SafetyLevel.RISKY:
            bcol, txt, tcol = \
                (0,100,160), "CAUTION", self.CAUTION_COL
        else:
            bcol, txt, tcol = \
                (0,0,140), "UNSAFE", self.UNSAFE_COL

        cv2.rectangle(frame,(0,h-85),(w,h),bcol,-1)
        ts = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0]
        cv2.putText(frame, txt,
            ((w-ts[0])//2, h-45),
            cv2.FONT_HERSHEY_DUPLEX, 1.8, tcol, 3)
            
        rs = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, reason,
            ((w-rs[0])//2, h-12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (220,220,220), 2)
            
        cv2.putText(frame,
            "D=drivable  L=lanes  R=restart  Q=quit",
            (w-400, h - 90 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (170,170,170), 1)
        return frame

class ThreadedCamera:
    """Background frame grabber to prevent USB camera latency and GPU starvation."""
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.width = width
        self.height = height
        # Removed cap.set() hardware overrides here to prevent green screen issues with DroidCam
        self.grabbed, frame = self.cap.read()
        if self.grabbed and frame is not None:
            self.frame = cv2.resize(frame, (self.width, self.height))
        else:
            self.frame = None
        self.started = False
        self.read_lock = threading.Lock()
        self.stopped = False

    def start(self):
        if not self.cap.isOpened():
            return self
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed and frame is not None:
                # Safely upsample the frame in software without breaking the virtual camera driver
                frame = cv2.resize(frame, (self.width, self.height))
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def release(self):
        self.stopped = True
        if self.started:
            self.thread.join()
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, propId):
        return self.cap.get(propId)

def run(video_path, cfg):
    detector = YOLOPv2Detector(cfg)
    tracker = SORTTracker()
    estimator = Estimator(cfg)
    ttc = TTCEngine(cfg)
    analyzer = OvertakingAnalyzer(cfg)
    path_filter = LanePathFilter()
    hud = HUDRenderer()

    if isinstance(video_path, int):
        print(f"Configuring Live Camera (Index {video_path}) with Threaded Grabber...")
        # Force 720p to prevent 4K overhead from phone
        cap = ThreadedCamera(video_path, width=1280, height=720).start()
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Video open nahi hui: {video_path}")
        return False

    # Video FPS sync
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120:
        video_fps = 30.0
    frame_delay = 1.0 / video_fps
    print(f"Video FPS: {video_fps:.1f}")

    cv2.namedWindow("YOLOPv2 Overtaking Safety", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOPv2 Overtaking Safety", cfg.OUTPUT_WIDTH, cfg.OUTPUT_HEIGHT)

    fps_ctr = deque(maxlen=30)
    frame_n = 0
    paused = False
    show_da = cfg.SHOW_DRIVABLE
    show_ll = cfg.SHOW_LANES

    # Cache — only update every SKIP_FRAMES
    last_seg_frame = None
    last_tracks = []
    last_safety = SafetyLevel.UNSAFE
    last_reason = "Initializing..."
    ego_spd = 0.0
    t_prev = time.time()

    # Check karein ki path string hai ya camera index (int)
    if isinstance(video_path, str):
        display_name = Path(video_path).name
    else:
        display_name = f"Live Camera (Index {video_path})"

    print(f"\nStarting: {display_name}")
    print("D=drivable  L=lanes  Space=pause"
          "  R=restart  Q=quit\n")
    
    while True:
        loop_start = time.time()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video ended.")
                break

            frame_n += 1
            t_now = time.time()
            fps_ctr.append(1.0 / max(t_now - t_prev, 0.001))
            t_prev = t_now
            fps = float(np.mean(fps_ctr))

            orig_h, orig_w = frame.shape[:2]

            # Detection every SKIP_FRAMES
            if frame_n % cfg.SKIP_FRAMES == 0:

                detections, seg_frame, orig_shape, da_mask, ll_mask = detector.detect(
                    frame, show_da=show_da, show_ll=show_ll
                )
                last_seg_frame = seg_frame

                # Update dynamic lane boundaries
                path_filter.update(ll_mask, da_mask, 720, 1280)

                # ── Blind-Mode flag for strict safety overrides ──
                is_blind_mode = path_filter.is_blind

                # Track
                track_input = [
                    {
                        "bbox": d["bbox"],
                        "confidence": d["conf"],
                        "class_id": d["cls"],
                        "class_name": COCO_NAMES.get(d["cls"], "vehicle"),
                    }
                    for d in detections
                ]

                tracks = tracker.update(track_input)

                # Estimate
                tracked = []
                for t in tracks:
                    tid = t["track_id"]
                    x1 = int(t["bbox"][0])
                    y1 = int(t["bbox"][1])
                    x2 = int(t["bbox"][2])
                    y2 = int(t["bbox"][3])
                    cid = t.get("class_id", 2)
                    dist = estimator.estimate_distance(tid, x1, y1, x2, y2, cid)
                    spd = estimator.estimate_speed(tid, dist)
                    dire = estimator.estimate_direction(tid, x1, y1, x2, y2, orig_h)

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
                last_tracks = tracked
                ego_spd = estimator.estimate_ego_motion(frame)

                # Safety
                feasible, reason = analyzer.analyze(
                    orig_w, orig_h, tracked, path_filter
                )

                enriched = []
                for v in tracked:
                    bbox = v["bbox"]

                    # Dynamic Right-Side Filter: skip vehicles right of corridor
                    if path_filter.is_right_of_corridor(bbox, orig_h, orig_w):
                        continue

                    # Area-Voting Gating: 5x5 drivable mask check
                    on_road = path_filter.is_on_drivable(bbox, da_mask, orig_h, orig_w)

                    # Dynamic Zone Classification using lane boundaries
                    if v["direction"] == "oncoming":
                        zone = "oncoming_lane"
                    else:
                        zone = path_filter.classify_zone(bbox, orig_h, orig_w)
                        
                    # TTC Weighted Safety for Ego Lane
                    # Blind-Mode: raise MANEUVER_GAP from 25m to 30m
                    is_closing_fast = v["rel_speed_kmh"] > 10.0
                    base_gap = 30.0 if is_blind_mode else 25.0
                    gap_threshold = 30.0 if is_closing_fast else base_gap

                    enriched.append({
                        "track_id": v["id"],
                        "bbox": v["bbox"],
                        "distance_m": v["distance"],
                        "speed_kph": v["rel_speed_kmh"],
                        "direction": v["direction"],
                        "class_name": v["class_name"],
                        "is_oncoming": v["direction"] == "oncoming",
                        "is_relevant": on_road,
                        "is_too_close": v["distance"] < gap_threshold and zone == "ego_lane",
                        "is_critical": v["distance"] < 8.0,
                        "is_parked": False,
                        "approach_rate": (
                            v["rel_speed_kmh"] / 3.6
                            if v["direction"] == "oncoming"
                            else 0.0
                        ),
                        "zone": zone,
                    })

                ttc_dec = ttc.evaluate(enriched, None)

                critical_vehicles = [v for v in enriched if v.get("is_critical")]
                too_close_ego = [v for v in enriched if v.get("is_too_close")]

                if critical_vehicles:
                    last_safety = SafetyLevel.UNSAFE
                    last_reason = "CRITICAL | Brake Now"
                elif not feasible:
                    last_safety = SafetyLevel.UNSAFE
                    last_reason = f"NO OVERTAKE | {reason}"
                elif ttc_dec.level == SafetyLevel.UNSAFE:
                    last_safety = SafetyLevel.UNSAFE
                    last_reason = ttc_dec.reason
                elif too_close_ego:
                    last_safety = SafetyLevel.RISKY
                    last_reason = "CAUTION | Gap too small for maneuver"
                elif ttc_dec.level == SafetyLevel.RISKY:
                    last_safety = SafetyLevel.RISKY
                    last_reason = ttc_dec.reason
                else:
                    # ── Blind-Mode Strict Safety Override ───────────
                    if is_blind_mode:
                        # Check if ANY vehicle is in the overtake lane
                        overtake_lane_occupied = any(
                            v.get("zone") == "overtake_lane"
                            for v in enriched
                        )
                        if overtake_lane_occupied:
                            # Force RISKY (CAUTION) instead of UNSAFE to reduce panicky behavior
                            last_safety = SafetyLevel.RISKY
                            last_reason = "CAUTION | Blind Driving | Reduced Confidence"
                        else:
                            # Cap SAFE → CAUTION when lane lines are missing
                            last_safety = SafetyLevel.RISKY
                            last_reason = "CAUTION | Blind Driving | Reduced Confidence"
                    else:
                        last_safety = SafetyLevel.SAFE
                        last_reason = "SAFE TO OVERTAKE"

                estimator.cleanup({t["id"] for t in last_tracks})

            # ── Render (FIXED SCALING) ────────────────────
            if last_seg_frame is not None:
                display = last_seg_frame.copy()
            else:
                display = cv2.resize(frame, (1280, 720))

            # Display image ki actual width aur height lein (Jo 1280x720 hai)
            disp_h, disp_w = display.shape[:2]

            # Scale bboxes to ACTUAL display size (1280x720) instead of config width
            sx = disp_w / max(orig_w, 1)
            sy = disp_h / max(orig_h, 1)

            scaled = []
            for v in last_tracks:
                sv = v.copy()
                b = v["bbox"]
                # Ab boxes image ke saath perfectly align honge
                sv["bbox"] = [
                    int(b[0] * sx),
                    int(b[1] * sy),
                    int(b[2] * sx),
                    int(b[3] * sy),
                ]
                scaled.append(sv)

            display = hud.render(
                display, fps, scaled, ego_spd, last_safety, last_reason, cfg,
                lane_mode=path_filter.mode_display,
                confidence=path_filter.confidence_score
            )
            cv2.imshow("YOLOPv2 Overtaking Safety", display)

        # FPS sync
        elapsed = time.time() - loop_start
        if isinstance(video_path, int):
            wait_ms = 1  # Live camera controls its own framerate (no artificial delay needed)
        else:
            wait_ms = max(1, int((frame_delay - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            cap.release()
            cv2.destroyAllWindows()
            return False

        elif key in [ord("r"), ord("R")]:
            cap.release()
            cv2.destroyAllWindows()
            return True

        elif key == ord(" "):
            paused = not paused
            print("Paused" if paused else "Resumed")

        elif key in [ord("d"), ord("D")]:
            show_da = not show_da
            print(f"Drivable: {'ON' if show_da else 'OFF'}")

        elif key in [ord("l"), ord("L")]:
            show_ll = not show_ll
            print(f"Lanes: {'ON' if show_ll else 'OFF'}")

        elif key in [ord('r'), ord('R')]:
            if isinstance(video_path, str): # Sirf video files ke liye restart allow karein
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print("Restart not available for live camera.")

    cap.release()
    cv2.destroyAllWindows()
    return True

"""
if __name__ == "__main__":
    cfg = Config()
    print("=" * 55)
    print("  YOLOPv2 Overtaking Safety System")
    print("=" * 55)

    while True:
        print("\nSelect video file...")
        video = select_video(cfg)
        if not video:
            print("No video selected. Exiting.")
            break
        print(f"Selected: {Path(video).name}")
        if not run(video, cfg):
            break

    print("\nGoodbye!")
    cv2.destroyAllWindows()"""


# main.py ke bottom block ko isse replace karein
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, help='Directly start with Camera index (e.g. 1 or 2)')
    args = parser.parse_args()

    cfg = Config()
    print("=" * 55)
    print("  YOLOPv2 Overtaking Safety System - LIVE")
    print("=" * 55)

    if args.cam is not None:
        video = args.cam
    else:
        mode = input("Choose Mode: [1] Video File | [2] Laptop Cam | [3] Phone Cam (DroidCam/Iriun): ")

        if mode == '1':
            video = select_video(cfg) # Purana file selection
        elif mode == '2':
            video = 0 # Laptop cam is usually 0
        elif mode == '3':
            # Phone connected via USB/Wi-Fi using DroidCam/Iriun appears as a local camera (index 1 or 2)
            cam_idx = input("Enter Phone Camera Index (Usually 1 or 2, default is 1): ")
            try:
                video = int(cam_idx)
            except ValueError:
                print("Invalid input, defaulting to index 1.")
                video = 1
        else:
            print("Invalid choice. Exiting.")
            sys.exit()

    if video is not None:
        # run() function int aur string dono handle karta hai
        run(video, cfg) 

    print("\nGoodbye!")
    cv2.destroyAllWindows()