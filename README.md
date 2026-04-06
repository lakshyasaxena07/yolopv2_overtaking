# YOLOPv2 Overtaking Safety Detection System

A real-time computer vision system for detecting safe overtaking opportunities on roads using YOLOPv2 (You Only Look Once Panoptic v2) object detection and Time-To-Collision (TTC) analysis.

## 🚗 Overview

This project is an intelligent vehicle overtaking decision support system that:
- Detects vehicles and analyzes their positions in real-time
- Tracks multiple vehicles using Kalman Filter-based tracking (SORT)
- Estimates distance, speed, and direction of vehicles
- Calculates Time-To-Collision (TTC) for safety assessment
- Provides real-time HUD visualization with safety status
- Supports **India and International** driving modes

## ✨ Features

### 1. **YOLOPv2-based Detection**
   - Real-time vehicle, bus, truck, motorcycle detection
   - Confidence-based filtering (default: 0.3)
   - Non-Maximum Suppression (NMS) at 0.45 IoU threshold
   - Drivable area segmentation (green overlay)
   - Lane line detection (red overlay)

### 2. **Vehicle Tracking (SORT)**
   - Kalman Filter for smooth trajectory prediction
   - Hungarian Algorithm for optimal track-to-detection assignment
   - IoU-based distance metric
   - Persistent track IDs across frames

### 3. **Distance & Speed Estimation**
   - Pinhole camera model-based distance calculation
   - Relative speed computation from distance history
   - Exponential Moving Average (EMA) smoothing
   - Real-time ego vehicle speed estimation via optical flow

### 4. **Direction Detection**
   - Lateral movement analysis
   - Oncoming vs. same-lane vehicle classification
   - Multi-frame voting for robust direction detection

### 5. **TTC-based Safety Engine**
   - Time-To-Collision calculation
   - Three safety levels:
     - 🟢 **SAFE** - TTC > 6.0 seconds
     - 🟡 **RISKY** - 3.5s < TTC < 6.0s
     - 🔴 **UNSAFE** - TTC < 3.5 seconds
   - Threat counting and analysis
   - Size-based penalties for different vehicle types

### 6. **Overtaking Lane Analysis**
   - India mode: Right-side overtake lane detection
   - International mode: Left-side overtake lane detection
   - Road drivability assessment
   - Oncoming vehicle threat detection

## 📁 Project Structure

```
yolopv2/
├── main.py                           # Main application entry point
├── config.py                         # Configuration parameters
├── README.md                         # This file
├── models/
│   └── yolopv2.pt                   # YOLOPv2 model weights (not tracked)
├── modules/
│   ├── __init__.py
│   ├── yolopv2_detector.py          # YOLOPv2 inference wrapper
│   ├── tracker.py                    # SORT tracker (Kalman + Hungarian)
│   ├── estimator.py                  # Distance/speed/direction estimation
│   └── ttc_engine.py                 # Safety decision engine
└── Test/
    └── videos/                       # Test video directory
```

## 🛠️ Modules

### `main.py`
- Main application controller
- Video playback management (pause, restart controls)
- Frame-by-frame processing pipeline
- HUD rendering with real-time metrics
- FPS calculation and display

**Keyboard Controls:**
- **D** - Toggle drivable area overlay
- **L** - Toggle lane lines overlay
- **Space** - Pause/Resume
- **R** - Restart video
- **Q** - Quit

### `config.py`
Configuration parameters:
- Model path and device (CUDA/CPU)
- Detection thresholds (confidence, NMS)
- Camera calibration (focal length, vehicle widths)
- TTC safety thresholds
- Display settings (drivable area opacity, lane thickness)
- Driving mode (India/International)

### `modules/yolopv2_detector.py`
YOLOPv2 inference pipeline:
- Model loading with TorchScript JIT compilation
- Frame preprocessing (letterboxing, normalization)
- Panoptic segmentation output processing
- Vehicle bounding box extraction and scaling
- Drivable area and lane line masking

### `modules/tracker.py`
SORT tracking implementation:
- `KalmanBoxTracker`: Individual vehicle state tracking
  - 8D state space: [x1, y1, x2, y2, vx, vy, vw, vh]
  - 4D measurement space: bounding box coordinates
  - Kalman prediction and update
- `SORTTracker`: Multi-object tracking coordination
  - IoU-based distance metric
  - Hungarian algorithm for optimal assignment
  - Track management (creation, termination, propagation)

### `modules/estimator.py`
Per-vehicle metric estimation:
- **Distance**: Pinhole camera model with real object widths
- **Speed**: Relative speed from distance history with EMA smoothing
- **Direction**: Lateral motion analysis + voting system
- **Ego Speed**: Optical flow (Lucas-Kanade) based estimation

### `modules/ttc_engine.py`
Safety decision engine:
- Time-To-Collision calculation: TTC = distance / approach_rate
- Multi-criteria safety evaluation:
  1. Overtake feasibility check
  2. Oncoming threat TTC calculation
  3. Congestion detection (vehicles too close)
  4. Decision stabilization (5-frame history)
- Size-based penalties for different vehicle categories
- `SafetyDecision` dataclass with color and text output

## 🚀 Installation

### Prerequisites
- Python 3.8+
- OpenCV 4.5+
- NumPy, SciPy
- FilterPy (Kalman Filter implementation)
- Device-specific PyTorch installation (see Step 2)

### Step 1: Clone Repository
```bash
git clone https://github.com/lakshyasaxena07/yolopv2_overtaking.git
cd yolopv2_overtaking
```

### Step 2: Ensure Proper Device Setup (PyTorch)
This project supports multiple hardware accelerators. Before installing the rest of the dependencies, install PyTorch according to your device:

#### 🟢 Option A: NVIDIA GPU (CUDA)
For maximum performance using NVIDIA graphics cards.
```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 🔵 Option B: Intel GPU (Intel Arc / Iris Xe)
For hardware acceleration on Intel GPUs using Intel Extension for PyTorch.
```bash
# Install PyTorch with Intel XPU support
python -m pip install torch torchvision torchaudio --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
*(Note for Intel GPU users: Make sure your Intel graphics drivers are up to date).*

#### ⚪ Option C: CPU Only
For laptops, older devices, or running without a dedicated supported graphics card.
```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Remaining Dependencies
```bash
# Install OpenCV, SciPy, FilterPy, etc.
pip install -r requirements.txt
```

### Step 4: Download Model Weights
The YOLOPv2 model weights (`models/yolopv2.pt`) are not included in the repository due to size constraints (149MB). 
1. Download `yolopv2.pt` from the [YOLOP GitHub repository](https://github.com/hustvl/YOLOP) or equivalent weights.
2. Place the downloaded file in the `models/` folder inside the project.

## 🎮 Usage

### Basic Usage
```bash
python main.py
```
This will:
1. Launch file dialog to select a video
2. Initialize YOLOPv2 detector on GPU/CPU
3. Start real-time analysis and display

### Processing Flow
```
Video Frame
    ↓
YOLOPv2 Detection (vehicles, lanes, drivable area)
    ↓
SORT Tracking (assign detections to tracks)
    ↓
Estimator (distance, speed, direction per vehicle)
    ↓
TTC Engine (safety evaluation)
    ↓
HUD Rendering (visualization with metrics)
    ↓
Display with FPS counter
```

## 📊 Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_PATH = r"path/to/yolopv2.pt"
DEVICE = "cuda"  # "cuda" or "cpu"
CONF_THRESH = 0.3  # Detection confidence threshold
IMG_SIZE = 640  # Model input size

# Camera Calibration
FOCAL_LENGTH_PX = 900.0  # Tuned focal length
REAL_CAR_WIDTH_M = 1.8   # Physical car width

# Safety Thresholds
TTC_SAFE = 6.0   # Safe TTC (seconds)
TTC_RISKY = 3.5  # Risky TTC (seconds)

# Driving Mode
DRIVING_MODE = "india"  # "india" or "international"

# Display
SHOW_DRIVABLE = True
SHOW_LANES = True
DRIVABLE_ALPHA = 0.4  # Transparency (0-1)
```

## 📈 Output Metrics

### HUD Display
- **FPS**: Frame processing rate
- **Tracks**: Active vehicle count
- **Ego Speed**: Vehicle speed (km/h)
- **Vehicle Info**: ID, class, distance (m), relative speed (km/h)
- **Safety Status**: SAFE / CAUTION / UNSAFE
- **Reason**: Detailed safety evaluation reason

### Bounding Boxes
- 🟢 **Green** - Same-lane or following vehicles
- 🔴 **Red** - Oncoming vehicles (danger)

## 🔧 Algorithm Details

### Distance Estimation
```
Distance = (Real Width × Focal Length) / Pixel Width
```
Real object widths configured for:
- Car: 1.8m
- Motorcycle: 0.8m
- Bus: 2.6m
- Truck: 2.5m

### Speed Calculation (Relative)
```
Speed = (Distance[t-n] - Distance[t]) / Δt
Smoothed = EMA_ALPHA × Raw + (1 - EMA_ALPHA) × Previous
```

### Time-To-Collision
```
TTC = Distance / Approach_Rate
```
Where approach_rate is the closing speed (m/s)

### Safety Decision Logic
```
1. Check drivable area feasibility
2. Calculate TTC for oncoming threats
3. Check congestion (vehicles in overtake zone)
4. Apply 5-frame history stabilization
5. Return SafetyLevel (SAFE/RISKY/UNSAFE)
```

## 🎯 Supported Vehicle Classes

From COCO dataset:
- 0: Person
- 1: Bicycle
- 2: **Car** ✓
- 3: **Motorcycle** ✓
- 5: **Bus** ✓
- 7: **Truck** ✓

## 🐛 Known Limitations

1. **Model Size**: YOLOPv2.pt (149MB) must be downloaded separately
2. **Camera Calibration**: Focal length requires tuning for different cameras
3. **Speed Estimation**: Relative speed only (requires calibration for absolute speed)
4. **Nighttime Performance**: Limited performance in low light conditions
5. **Weather**: Reduced accuracy in heavy rain/fog

## 📚 References

- **YOLOPv2**: [YOLOP - GitHub](https://github.com/hustvl/YOLOP)
- **SORT Tracking**: [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
- **Kalman Filtering**: [FilterPy Documentation](https://filterpy.readthedocs.io/)

## 📝 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Developed as a vehicle safety analysis system for overtaking scenarios.

## 💬 Support

For issues, feature requests, or improvements, please open an issue on GitHub.

---

**Last Updated**: March 2026  
**Status**: Stable  
**Python**: 3.8+  
**CUDA**: Optional (11.8+ recommended)
