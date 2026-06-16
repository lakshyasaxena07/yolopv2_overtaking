# config.py — YOLOPv2 Overtaking Safety System (Production Config)

class Config:

    # ── YOLOPv2 Model ──────────────────────────────────────────
    MODEL_PATH       = r"E:\Minor 2\Most_Stable\Claude\yolopv2\models\yolopv2.pt"
    ONNX_MODEL_PATH  = r"E:\Minor 2\Most_Stable\Claude\yolopv2\models\yolopv2_fp16.onnx"
    USE_ONNX         = True
    USE_TENSORRT     = False # onnxruntime 1.24.4 + CUDA 13.0 = TRT segfault. CUDA EP is fast enough.
    DEVICE           = "cuda"
    CONF_THRESH      = 0.25
    IOU_THRESH       = 0.45
    IMG_SIZE         = 384   # Matches your static ONNX model exactly

    # BDD100k dataset vehicle classes
    VEHICLE_CLASSES  = [1, 2, 3, 5, 7]
    # 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck


    # ── Camera / Video ─────────────────────────────────────────
    CAMERA_SOURCE    = 0
    FRAME_WIDTH      = 1280
    FRAME_HEIGHT     = 720
    SYNC_VIDEO       = True         # Fixes 'fast video' issue

    # ── Fault Tolerance ────────────────────────────────────────
    WATCHDOG_TIMEOUT_MS = 300       # ms without frame → SENSOR_FAILURE

    # ── Output ─────────────────────────────────────────────────
    OUTPUT_WIDTH     = 1100
    OUTPUT_HEIGHT    = 720

    # ── Camera Calibration ─────────────────────────────────────
    FOCAL_LENGTH_PX         = 2850.0    # Default — override via calibration
    FOCAL_CALIBRATION_ENABLED = False   # Set True to run auto-calibration helper
    REAL_CAR_WIDTH_M        = 1.8
    REAL_TRUCK_WIDTH_M      = 2.5
    REAL_BUS_WIDTH_M        = 2.6
    REAL_BIKE_WIDTH_M       = 0.7       # bicycle
    REAL_MOTORCYCLE_WIDTH_M = 0.7       # SAFETY-CRITICAL: class 3 = motorcycle, NOT car

    # ── TTC Thresholds (Kinematic Safety) ──────────────────────
    TTC_SAFE             = 6.0      
    TTC_RISKY            = 3.5      
    TTC_UNSAFE_SAME_DIR  = 4.0      
    TTC_UNSAFE_ONCOMING  = 6.0      

    # ── Ego Motion (Downsampled Optical Flow) ──────────────────
    EGO_FLOW_WIDTH   = 320          
    EGO_FLOW_HEIGHT  = 180          

    # ── Safety History ─────────────────────────────────────────
    SAFETY_HISTORY_WINDOW_S = 0.5   

    # ── Driving Mode ───────────────────────────────────────────
    DRIVING_MODE     = "india"      

    # ── YOLOPv2 Overlay ────────────────────────────────────────
    SHOW_DRIVABLE    = True         
    SHOW_LANES       = True         
    DRIVABLE_ALPHA   = 0.4          
    LANE_THICKNESS   = 3            

    # ── Overtaking Logic ───────────────────────────────────────
    MIN_OVERTAKE_CLEAR = 0.15       

    # ── Videos ─────────────────────────────────────────────────
    VIDEOS_FOLDER    = r"D:\Overtaking_safety\Test\Custom_videos"
    # VIDEOS_FOLDER    = r"D:\1001_0\1001"


    # ── Debug ──────────────────────────────────────────────────
    SHOW_EGO_SPEED       = True
    SHOW_DIRECTION_DEBUG = False
    SHOW_DRIVABLE_DEBUG  = False    


    TARGET_FPS       = 30           # Matches normal.mp4
