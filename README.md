# YOLOPv2 Overtaking Safety Detection System

A real-time computer vision system for detecting safe overtaking opportunities on Indian and international roads, powered by **YOLOPv2** (You Only Look Once Panoptic v2) multi-task perception with **ONNX Runtime / TensorRT** acceleration and **Time-To-Collision (TTC)** safety analysis.

> **Status:** Stable · **Python:** 3.8+ · **CUDA:** Optional (11.8+ recommended) · **Last Updated:** April 2026

---

## 🚗 Overview

Overtaking on two-lane roads is one of the most dangerous driving maneuvers, especially in India where traffic is heterogeneous and unpredictable. This project provides a **real-time decision support system** that:

- **Detects vehicles** (cars, trucks, buses, motorcycles, bicycles) using YOLOPv2's multi-head architecture
- **Segments the road** into drivable area (green overlay) and lane lines (red overlay)
- **Tracks multiple vehicles** across frames using Kalman Filter–based SORT
- **Estimates distance, speed, and direction** of each tracked vehicle
- **Calculates Time-To-Collision (TTC)** for safety assessment
- **Classifies zones dynamically** (ego lane / overtake lane / right shoulder) using detected lane boundaries
- **Handles degraded conditions** with a multi-stage Blind-Mode fallback (Memory → Virtual → Fallback)
- **Provides a real-time HUD** with safety status: 🟢 SAFE · 🟡 CAUTION · 🔴 UNSAFE
- **Supports India and International** driving modes (left-hand / right-hand traffic)
- **Live camera support** with threaded frame grabbing (DroidCam / Iriun / Laptop webcam)

---

## ✨ Key Features

### 1. YOLOPv2 Multi-Task Perception
- **Single-shot inference** for detection + drivable area + lane lines
- ONNX Runtime with **TensorRT → CUDA → CPU** provider cascade
- FP16 quantized model for reduced VRAM and faster inference
- PyTorch JIT fallback when ONNX is unavailable
- Confidence filtering (default: 0.25) and NMS (IoU: 0.45)

### 2. SORT Vehicle Tracking
- **Kalman Filter** for smooth trajectory prediction (8D state space)
- **Hungarian Algorithm** for optimal track-to-detection assignment
- IoU-based distance metric with persistent track IDs across frames
- Automatic track creation, propagation, and termination

### 3. Distance & Speed Estimation
- **Pinhole camera model**: `Distance = (Real Width × Focal Length) / Pixel Width`
- Per-class real-world widths (Car: 1.8m, Truck: 2.5m, Bus: 2.6m, Motorcycle: 0.7m)
- **Moving average** (window of 5) for distance smoothing
- **EMA-smoothed** relative speed computation from distance history
- Real-time **ego vehicle speed** estimation via Lucas-Kanade optical flow

### 4. Direction Detection
- Three-signal voting system:
  1. **Width change rate** — growing bbox → oncoming
  2. **Y-center movement** — descending → oncoming
  3. **Vertical frame position** — upper third → oncoming
- Multi-frame temporal smoothing (60% majority over 8 frames)

### 5. Dynamic Lane-Based Path Filtering
- **Contour-based ego-lane detection** from YOLOPv2 lane-line mask
- `cv2.fitLine` slope projection for horizon-aware boundaries
- **Right-side corridor filtering** — auto-ignores vehicles right of the right lane boundary
- **5×5 drivable-area gating** — filters detections not on the road
- **Dynamic zone classification** (ego / overtake / right shoulder)

### 6. Blind-Mode Fallback (Degraded Conditions)
When lane lines are lost (rain, night, poor markings), the system degrades gracefully:

| Mode | Source | Confidence | Behavior |
|------|--------|------------|----------|
| **OPTICAL** | Live lane-line contours | 100% | Normal operation |
| **MEMORY** | Last valid boundaries | Decays linearly over 45 frames | Progressive widening (3%/frame) |
| **VIRTUAL** | Drivable-area mask edges | 0% | Green-mask proxy boundaries |
| **FALLBACK** | Fixed pixel ratios | 0% | Absolute last resort |

- Forces **CAUTION** state when no lane data is available
- Raises gap thresholds from 25m → 30m in Blind-Mode

### 7. TTC Safety Engine
- `TTC = Distance / Approach_Rate`
- Size-based penalties: Bus/Truck ×1.4, Car ×1.0, Motorcycle ×0.9
- Multi-criteria evaluation:
  1. Overtake feasibility check
  2. Oncoming threat TTC calculation
  3. Ego-lane congestion detection
  4. Overtake lane blockage detection
- **5-frame majority vote** decision stabilization
- Three safety levels:
  - 🟢 **SAFE** — Clear to overtake
  - 🟡 **CAUTION** — TTC 2.5s–5.0s or gap too small
  - 🔴 **UNSAFE** — TTC < 2.5s, critical distance (< 8m), or oncoming threat

### 8. Live Camera Support
- **ThreadedCamera** class for background frame grabbing (prevents GPU starvation)
- Software-based 720p upsampling (avoids green-screen issues with virtual camera drivers)
- Supports laptop webcam, DroidCam, and Iriun via USB/Wi-Fi
- Command-line `--cam` argument for direct camera index selection

---

## 📁 Project Structure

```
yolopv2/
├── main.py                              # Main application entry point & HUD renderer
├── config.py                            # All configuration parameters
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── .gitignore                           # Git ignore rules
├── models/
│   ├── yolopv2.pt                       # PyTorch JIT weights (not tracked)
│   └── yolopv2_fp16.onnx               # ONNX FP16 model (not tracked)
├── modules/
│   ├── __init__.py
│   ├── yolopv2_detector.py              # YOLOPv2 inference (ONNX + PyTorch)
│   ├── tracker.py                       # SORT tracker (Kalman + Hungarian)
│   ├── estimator.py                     # Distance / speed / direction estimation
│   ├── ttc_engine.py                    # Safety decision engine
│   └── lane_path_filter.py             # Dynamic lane-based path filtering + Blind-Mode
├── scripts/
│   ├── export_onnx.py                   # PyTorch → ONNX FP16 export script
│   └── test_onnx.py                     # ONNX validation script
└── experiments/                         # Scratchpad, logs, and outputs (not tracked)
```

---

## 🛠️ Module Details

### `main.py`
- **Application controller** with video/camera source selection (file / laptop cam / phone cam)
- **OvertakingAnalyzer** — dynamic lane-based overtake feasibility analysis
- **HUDRenderer** — real-time heads-up display with safety status, FPS, tracks, ego speed, and mode badge
- **ThreadedCamera** — background frame grabber for live camera sources
- Frame-by-frame processing pipeline with configurable skip-frames

**Keyboard Controls:**
| Key | Action |
|-----|--------|
| **D** | Toggle drivable area overlay |
| **L** | Toggle lane lines overlay |
| **Space** | Pause / Resume |
| **R** | Restart video |
| **Q / Esc** | Quit |

### `config.py`
All tunable parameters in one place:
- Model paths (PyTorch JIT + ONNX), device selection, inference size (384)
- Detection thresholds (confidence, NMS, vehicle classes)
- Camera calibration (focal length, per-class vehicle widths)
- TTC safety thresholds (SAFE: 6.0s, RISKY: 3.5s)
- Display settings (overlay opacity, lane thickness, output resolution)
- Driving mode (`india` / `international`)

### `modules/yolopv2_detector.py`
- **Dual-backend inference**: ONNX Runtime (TensorRT/CUDA) → PyTorch JIT fallback
- Letterbox preprocessing with stride-aligned padding
- FP16 support for reduced VRAM usage
- Segmentation mask extraction with bilinear interpolation to 720×1280
- Z-order rendering: drivable area → lane lines (ensures lane visibility at horizon)

### `modules/tracker.py`
- **KalmanBoxTracker**: 8D state `[x1, y1, x2, y2, vx, vy, vw, vh]`, 4D measurement
- **SORTTracker**: Multi-object coordination with IoU distance metric
- Hungarian algorithm for optimal track assignment
- Automatic track lifecycle management

### `modules/estimator.py`
- **Distance**: Pinhole camera model + 5-frame moving average
- **Speed**: Relative speed from distance history + EMA smoothing (α = 0.35)
- **Direction**: 3-signal voting + 8-frame temporal smoothing
- **Ego Speed**: Lucas-Kanade optical flow with median filtering

### `modules/ttc_engine.py`
- TTC computation with size-based approach rate penalties
- Multi-level threat assessment (oncoming, too-close, overtake-blocked)
- `SafetyDecision` dataclass with color, text, and reason output
- 5-frame history for majority-vote decision stabilization

### `modules/lane_path_filter.py`
- **Optical mode**: Real-time contour detection → `cv2.fitLine` slope projection
- **Memory mode**: Temporal buffer (45 frames) with progressive widening
- **Virtual mode**: Drivable-area edge proxy boundaries
- **Fallback mode**: Fixed pixel ratios as absolute last resort
- Confidence score decay and mode badge for HUD display

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 11.8+ (recommended for real-time performance)
- OpenCV 4.5+

### Step 1: Clone Repository
```bash
git clone https://github.com/lakshyasaxena07/yolopv2_overtaking.git
cd yolopv2_overtaking
```

### Step 2: Install PyTorch (Device-Specific)
Install PyTorch according to your hardware **before** other dependencies:

#### 🟢 Option A: NVIDIA GPU (CUDA) — Recommended
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 🔵 Option B: Intel GPU (Intel Arc / Iris Xe)
```bash
python -m pip install torch torchvision torchaudio --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
> *Ensure Intel graphics drivers are up to date.*

#### ⚪ Option C: CPU Only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Weights
The model weights are **not included** in the repository due to size constraints:

1. **PyTorch weights** (`yolopv2.pt`, ~149MB): Download from the [YOLOPv2 GitHub](https://github.com/hustvl/YOLOP)
2. **ONNX FP16 model** (`yolopv2_fp16.onnx`): Export using `scripts/export_onnx.py` or download pre-exported weights
3. Place the downloaded files in the `models/` directory

---

## 🎮 Usage

### Video File Mode
```bash
python main.py
# Select mode [1] → Choose video via file dialog
```

### Laptop Webcam
```bash
python main.py
# Select mode [2]
```

### Phone Camera (DroidCam / Iriun)
```bash
python main.py
# Select mode [3] → Enter camera index (usually 1 or 2)
```

### Direct Camera Launch
```bash
python main.py --cam 1
```

### Processing Pipeline
```
Video / Camera Frame
        ↓
YOLOPv2 Detection (ONNX/PyTorch)
  → vehicles, drivable area, lane lines
        ↓
LanePathFilter Update
  → ego boundaries, zone classification, blind-mode cascade
        ↓
SORT Tracking
  → persistent track IDs across frames
        ↓
Estimator
  → distance, speed, direction per vehicle
        ↓
OvertakingAnalyzer + TTC Engine
  → safety evaluation with multi-criteria assessment
        ↓
HUD Rendering
  → bounding boxes, status bar, mode badge, FPS
        ↓
Display (1280×720)
```

---

## 📊 Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_PATH       = r"path/to/yolopv2.pt"
ONNX_MODEL_PATH  = r"path/to/yolopv2_fp16.onnx"
USE_ONNX         = True              # ONNX Runtime (faster) or PyTorch JIT
DEVICE           = "cuda"            # "cuda" or "cpu"
CONF_THRESH      = 0.25              # Detection confidence threshold
IMG_SIZE         = 384               # Model input size (384 for speed, 640 for accuracy)

# Camera Calibration
FOCAL_LENGTH_PX  = 2850.0            # Tuned focal length (adjust per camera)
REAL_CAR_WIDTH_M = 1.8               # Physical car width in meters

# Safety Thresholds
TTC_SAFE         = 6.0               # Safe TTC (seconds)
TTC_RISKY        = 3.5               # Risky TTC (seconds)

# Driving Mode
DRIVING_MODE     = "india"           # "india" (right-side overtake) or "international"

# Display
SHOW_DRIVABLE    = True              # Green drivable area overlay
SHOW_LANES       = True              # Red lane lines overlay
DRIVABLE_ALPHA   = 0.4               # Transparency (0–1)
SKIP_FRAMES      = 3                 # Run detection every N frames
```

---

## 📈 HUD Display

### Top Bar
- **FPS** — Frame processing rate
- **Tracks** — Active tracked vehicle count
- **Ego Speed** — Estimated vehicle speed (km/h)
- **Mode Badge** — Current lane detection mode (OPTICAL / MEMORY / VIRTUAL / FALLBACK)

### Bounding Boxes
- 🟢 **Green** — Same-lane or following vehicles
- 🔴 **Red** — Oncoming vehicles
- Labels show: `#ID class distance speed`

### Bottom Bar
- Safety status: **SAFE** / **CAUTION** / **UNSAFE**
- Reason text with detailed assessment

---

## 🔧 Algorithm Details

### Distance Estimation (Pinhole Camera Model)
```
Distance = (Real_Width × Focal_Length) / Pixel_Width
```
| Vehicle | Real Width |
|---------|-----------|
| Car | 1.8m |
| Motorcycle | 0.7m |
| Bus | 2.6m |
| Truck | 2.5m |

### Relative Speed (EMA Smoothed)
```
Raw_Speed = (Distance[t-n] - Distance[t]) / Δt
Smoothed  = 0.35 × Raw + 0.65 × Previous
```

### Time-To-Collision
```
TTC = Distance / Approach_Rate
Effective_Rate = Approach_Rate × Size_Penalty
```

### Safety Decision Logic
```
1. Check CRITICAL distance (< 8m) → UNSAFE
2. Check overtake feasibility (OvertakingAnalyzer) → UNSAFE if blocked
3. Evaluate TTC for oncoming threats → UNSAFE / CAUTION
4. Check ego-lane gap (< 25m normal, < 30m blind) → CAUTION
5. Blind-Mode override → force CAUTION when lanes are lost
6. All clear → SAFE TO OVERTAKE
```

---

## 🎯 Supported Vehicle Classes

From COCO dataset:
| ID | Class | Tracked |
|----|-------|---------|
| 0 | Person | — |
| 1 | Bicycle | ✓ |
| 2 | Car | ✓ |
| 3 | Motorcycle | ✓ |
| 5 | Bus | ✓ |
| 7 | Truck | ✓ |

---

## 🐛 Known Limitations

1. **Model Size**: Weights (149MB+ for PT, varies for ONNX) must be downloaded separately
2. **Camera Calibration**: Focal length requires tuning for different cameras
3. **Speed Estimation**: Relative speed only — requires calibration for absolute speed
4. **Nighttime**: Reduced accuracy in low light (Blind-Mode activates)
5. **Weather**: Reduced accuracy in heavy rain/fog (Blind-Mode compensates)
6. **Single Camera**: Monocular depth estimation — no stereo baseline

---

## 📚 References

- **YOLOPv2**: [You Only Look at Once for Panoptic Driving Perception](https://github.com/hustvl/YOLOP)
- **SORT**: [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
- **Kalman Filter**: [FilterPy Documentation](https://filterpy.readthedocs.io/)
- **ONNX Runtime**: [Microsoft ONNX Runtime](https://onnxruntime.ai/)

---

## 📝 License

This project is provided as-is for educational and research purposes.

## 👨‍💻 Author

Developed by [Lakshya Saxena](https://github.com/lakshyasaxena07) as a vehicle safety analysis system for overtaking scenarios.

## 💬 Support

For issues, feature requests, or improvements, please [open an issue](https://github.com/lakshyasaxena07/yolopv2_overtaking/issues) on GitHub.
