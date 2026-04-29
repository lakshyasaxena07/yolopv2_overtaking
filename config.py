# config.py — YOLOPv2 Overtaking Safety System

class Config:

    # ── YOLOPv2 Model ──────────────────────────────────────────
    MODEL_PATH       = r"E:\Minor 2\Most_Stable\Claude\yolopv2\models\yolopv2.pt"
    ONNX_MODEL_PATH  = r"E:\Minor 2\Most_Stable\Claude\yolopv2\models\yolopv2_fp16.onnx"
    USE_ONNX         = True
    DEVICE           = "cuda"
    CONF_THRESH      = 0.25
    IOU_THRESH       = 0.45
    IMG_SIZE         = 384   # 640 → 384 — VRAM bachega, speed badh jaayegi

    # COCO vehicle classes
    VEHICLE_CLASSES  = [1, 2, 3, 5, 7]
    # 1=bicycle, 2=car, 3=motorcycle/vehicle, 5=bus, 7=truck

    # ── Camera / Video ─────────────────────────────────────────
    CAMERA_SOURCE    = 0
    FRAME_WIDTH      = 1280
    FRAME_HEIGHT     = 720
    TARGET_FPS       = 0            # 0 = no cap, sync with video
    SKIP_FRAMES      = 3            # 2 → 3 — GPU load kam hoga

    # ── Output ─────────────────────────────────────────────────
    OUTPUT_WIDTH     = 1100
    OUTPUT_HEIGHT    = 720

    # ── Camera Calibration ─────────────────────────────────────
    FOCAL_LENGTH_PX  = 2850.0        # Tuned for accurate distance
    REAL_CAR_WIDTH_M    = 1.8
    REAL_TRUCK_WIDTH_M  = 2.5
    REAL_BUS_WIDTH_M    = 2.6
    REAL_BIKE_WIDTH_M   = 0.7

    # ── TTC Thresholds ─────────────────────────────────────────
    TTC_SAFE         = 6.0          # Seconds
    TTC_RISKY        = 3.5          # Seconds

    # ── Driving Mode ───────────────────────────────────────────
    DRIVING_MODE     = "india"      # "india" ya "international"

    # ── YOLOPv2 Overlay ────────────────────────────────────────
    SHOW_DRIVABLE    = True         # Green drivable area
    SHOW_LANES       = True         # Red lane lines
    DRIVABLE_ALPHA   = 0.4          # Transparency (0-1)
    LANE_THICKNESS   = 3            # Lane line thickness

    # ── Overtaking Logic ───────────────────────────────────────
    # Drivable area ka kitna % overtake lane mein hona chahiye
    # Agar kam hai toh overtake unsafe
    MIN_OVERTAKE_CLEAR = 0.15       # 15% minimum clear area

    # ── Videos ─────────────────────────────────────────────────
    VIDEOS_FOLDER    = "experiments/videos"

    # ── Debug ──────────────────────────────────────────────────
    SHOW_EGO_SPEED       = True
    SHOW_DIRECTION_DEBUG = False
    SHOW_DRIVABLE_DEBUG  = False    # Alag debug window