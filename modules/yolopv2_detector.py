# modules/yolopv2_detector.py
# Demo.py ka exact flow follow karta hai

import cv2
import torch
import numpy as np
import sys
from pathlib import Path

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

# YOLOPv2 mask output is always 720x1280
# driving_area_mask: seg[:,:,12:372,:] → scale_factor=2
# → 360*2=720 height, 640*2=1280 width
MASK_H = 720
MASK_W = 1280


class YOLOPv2Detector:

    def __init__(self, config):
        self.cfg    = config
        self.device = torch.device(
            config.DEVICE
            if torch.cuda.is_available() else "cpu"
        )
        self.half   = self.device.type != "cpu"
        self.stride = 32
        self.img_size = config.IMG_SIZE
        self._load_model()

    def _load_model(self):
        print(f"YOLOPv2 loading: {self.cfg.MODEL_PATH}")
        self.model = torch.jit.load(
            self.cfg.MODEL_PATH,
            map_location=self.device
        )
        if self.half:
            self.model.half()
        self.model.eval()
        dummy = torch.zeros(
            1, 3, self.img_size, self.img_size
        ).to(self.device)
        dummy = dummy.half() if self.half \
            else dummy.float()
        with torch.no_grad():
            self.model(dummy)
        print(f"YOLOPv2 ready on {self.device}")

    def detect(self, frame, show_da=True, show_ll=True):
        """
        Demo.py ka exact flow:
          1. Frame → MASK_W x MASK_H (1280x720)
          2. Letterbox → img_size tensor
          3. Model inference
          4. show_seg_result(frame_1280x720, masks)
          5. Bboxes scale back to original

        Returns:
          detections  : list of {bbox, conf, cls}
                        in ORIGINAL frame coords
          seg_frame   : 1280x720 frame with overlay
          orig_shape  : (h, w) original
        """
        orig_h, orig_w = frame.shape[:2]

        # ── Step 1: Resize to mask output size ────────────
        # Demo.py LoadImages produces 1280x720 frames
        # Mask is always 720x1280 — must match!
        frame_resized = cv2.resize(
            frame, (MASK_W, MASK_H),
            interpolation=cv2.INTER_LINEAR
        )

        # ── Step 2: Letterbox for model input ─────────────
        img_lb = self._letterbox(
            frame_resized.copy(),
            new_shape=(self.img_size, self.img_size),
            stride=self.stride
        )

        # ── Step 3: BGR→RGB, HWC→CHW, normalize ──────────
        img = img_lb[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half \
            else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        # ── Step 4: Inference ─────────────────────────────
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = \
                self.model(tensor)

        # ── Step 5: Detections ────────────────────────────
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(
            pred,
            self.cfg.CONF_THRESH,
            self.cfg.IOU_THRESH,
            classes=self.cfg.VEHICLE_CLASSES,
            agnostic=False
        )

        # Scale coords: tensor → resized → original
        detections = []
        for det in pred:
            if det is not None and len(det):
                # Scale to resized frame first
                det[:, :4] = scale_coords(
                    tensor.shape[2:],
                    det[:, :4],
                    frame_resized.shape
                ).round()

                # Then scale to original
                sx = orig_w / MASK_W
                sy = orig_h / MASK_H

                for *xyxy, conf, cls in det:
                    x1 = int(max(0,
                        min(xyxy[0] * sx, orig_w)))
                    y1 = int(max(0,
                        min(xyxy[1] * sy, orig_h)))
                    x2 = int(max(0,
                        min(xyxy[2] * sx, orig_w)))
                    y2 = int(max(0,
                        min(xyxy[3] * sy, orig_h)))
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": float(conf),
                            "cls":  int(cls)
                        })

        # ── Step 5: Apply seg overlay ─────────────────────
        da_mask = driving_area_mask(seg)
        ll_mask = lane_line_mask(ll)
        mask_h, mask_w = da_mask.shape[:2]

        # Letterbox resize — aspect ratio maintain karo
        frame_for_seg = self._letterbox(
            frame.copy(),
            new_shape=(mask_h, mask_w)
        )
        # frame_for_seg = cv2.resize(
        #     frame, (1280, 720),
        #     interpolation=cv2.INTER_LINEAR
        # )
        if show_da or show_ll:
            _da = da_mask if show_da else \
                  np.zeros_like(da_mask)
            _ll = ll_mask if show_ll else \
                  np.zeros_like(ll_mask)
            show_seg_result(
                frame_for_seg,
                (_da, _ll),
                is_demo=True
            )

        # Resize back to 1280x720 for display
        seg_display = cv2.resize(
            frame_for_seg, (1280, 720),
            interpolation=cv2.INTER_LINEAR
        )

        return detections, seg_display, (orig_h, orig_w)

    def get_class_name(self, cls_id):
        names = {
            0: "person",  1: "bicycle",
            2: "car",     3: "vehicle",
            5: "bus",     7: "truck"
        }
        return names.get(cls_id, "vehicle")

    def get_vehicle_width(self, cls_id):
        if cls_id == 2: return self.cfg.REAL_CAR_WIDTH_M
        if cls_id == 7: return self.cfg.REAL_TRUCK_WIDTH_M
        if cls_id == 5: return self.cfg.REAL_BUS_WIDTH_M
        if cls_id == 3: return self.cfg.REAL_BIKE_WIDTH_M
        return self.cfg.REAL_CAR_WIDTH_M

    @staticmethod
    def _letterbox(img, new_shape=(640, 640),
                   stride=32, color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0],
                new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)),
                     int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(
                img, new_unpad,
                interpolation=cv2.INTER_LINEAR
            )
        top    = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left   = int(round(dw - 0.1))
        right  = int(round(dw + 0.1))
        img    = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )
        return img