import cv2
import torch
import numpy as np
import sys
from pathlib import Path

# Path configuration
YOLOPV2_DIR = Path(r"E:\Minor 2\claude\YOLOPv2")
if str(YOLOPV2_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOPV2_DIR))

from utils.utils import (
    non_max_suppression,
    scale_coords,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    show_seg_result,
)

class YOLOPv2Detector:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != "cpu"
        self.stride = 32
        self.img_size = config.IMG_SIZE
        self._load_model()

    def _load_model(self):
        """Sirf model load aur warmup ke liye. Yahan 'frame' variable use nahi karna."""
        print(f"YOLOPv2 loading: {self.cfg.MODEL_PATH}")
        self.model = torch.jit.load(self.cfg.MODEL_PATH, map_location=self.device)
        if self.half:
            self.model.half()
        self.model.eval()
        
        # Warmup with dummy tensor
        dummy = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
        if self.half: dummy = dummy.half()
        with torch.no_grad():
            self.model(dummy)
        print(f"YOLOPv2 ready on {self.device}")

    def detect(self, frame, show_da=True, show_ll=True):
        """
        Main inference function. Frame yahan process hota hai.
        """
        orig_shape = frame.shape[:2] # (height, width)

        # 1. Letterbox - Optimized version that returns ratio and padding
        img_lb, ratio, (dw, dh) = self._letterbox(
            frame, 
            new_shape=(self.img_size, self.img_size), 
            stride=self.stride
        )

        # 2. Pre-process
        img = img_lb[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        # 3. Inference
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(tensor)

        # 4. Process Bounding Boxes (NMS + Correct Scaling)
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(
            pred, self.cfg.CONF_THRESH, self.cfg.IOU_THRESH,
            classes=self.cfg.VEHICLE_CLASSES, agnostic=False
        )

        # ── Step 4: Process Bounding Boxes (Final Version) ──
        detections = []
        for det in pred:
            if det is not None and len(det):
                # Manual calculation agar scale_coords dhoka de raha ho
                # (Raw coordinate - padding) / ratio
                det[:, [0, 2]] = (det[:, [0, 2]] - dw) / ratio
                det[:, [1, 3]] = (det[:, [1, 3]] - dh) / ratio
                
                # Boundary check
                det[:, 0].clamp_(0, orig_shape[1])
                det[:, 1].clamp_(0, orig_shape[0])
                det[:, 2].clamp_(0, orig_shape[1])
                det[:, 3].clamp_(0, orig_shape[0])

                for *xyxy, conf, cls in det:
                    detections.append({
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "conf": float(conf),
                        "cls":  int(cls)
                    })

        # 5. Segmentation Masks (Dynamic extraction based on padding)
        y_start, y_end = int(round(dh)), int(round(self.img_size - dh))
        x_start, x_end = int(round(dw)), int(round(self.img_size - dw))

        seg_img = cv2.resize(frame, (1280, 720)) # Final HUD size
        
        # Unconditionally compute masks for structural gating downstream
        da_valid = seg[:, :, y_start:y_end, x_start:x_end]
        da_mask = torch.nn.functional.interpolate(da_valid, size=(720, 1280), mode='bilinear', align_corners=False)
        _, da_mask = torch.max(da_mask, 1)
        _da = da_mask.int().squeeze().cpu().numpy()
        
        ll_valid = ll[:, :, y_start:y_end, x_start:x_end]
        ll_mask = torch.nn.functional.interpolate(ll_valid, size=(720, 1280), mode='bilinear', align_corners=False)
        ll_mask = torch.round(ll_mask).squeeze(1)
        _ll = ll_mask.int().squeeze().cpu().numpy()

        # Z-Order Rendering: Draw Drivable FIRST, Lane Lines SECOND
        if show_da:
            mask = _da == 1
            seg_img[mask] = seg_img[mask] * 0.6 + np.array([0, 255, 0], dtype=np.uint8) * 0.4
            
        if show_ll:
            mask = _ll == 1
            seg_img[mask] = seg_img[mask] * 0.2 + np.array([0, 0, 255], dtype=np.uint8) * 0.8

        return detections, seg_img, orig_shape, _da, _ll

    def get_vehicle_width(self, cls_id):
        width_map = {
            2: self.cfg.REAL_CAR_WIDTH_M,
            7: self.cfg.REAL_TRUCK_WIDTH_M,
            5: self.cfg.REAL_BUS_WIDTH_M,
            3: self.cfg.REAL_CAR_WIDTH_M,
            1: self.cfg.REAL_BIKE_WIDTH_M
        }
        return width_map.get(cls_id, self.cfg.REAL_CAR_WIDTH_M)

    @staticmethod
    def _letterbox(img, new_shape=(640, 640), stride=32, color=(114, 114, 114)):
        shape = img.shape[:2] 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2 
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)