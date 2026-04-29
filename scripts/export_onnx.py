# export_onnx.py — YOLOPv2 ONNX FP16 Export Script
# Strategy: Load model on GPU in FP16, then export directly to ONNX.
# This avoids SequenceConstruct conversion errors in onnxconverter_common.
#
# Usage:  conda activate overtaking_safety
#         python export_onnx.py
#
# Output: models/yolopv2_fp16.onnx

import torch
import numpy as np
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_PT    = BASE_DIR / "models" / "yolopv2.pt"
OUTPUT_FP16 = BASE_DIR / "models" / "yolopv2_fp16.onnx"

IMG_SIZE = 384

def export():
    print("=" * 55)
    print("  YOLOPv2 ONNX FP16 Direct GPU Export")
    print("=" * 55)

    if not torch.cuda.is_available():
        print("CUDA must be available to export FP16 directly. Exiting.")
        return

    device = torch.device('cuda:0')

    # ── Step 1: Load traced model ─────────────────────────────
    print(f"\n[1/3] Loading traced model: {MODEL_PT}")
    model = torch.jit.load(str(MODEL_PT), map_location=device)
    model = model.half()  # Convert to FP16
    model.eval()
    print("      Model loaded and converted to FP16 OK")

    # ── Step 2: Export to ONNX FP16 ───────────────────────────
    print(f"\n[2/3] Exporting directly to ONNX FP16: {OUTPUT_FP16}")
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device, dtype=torch.float16)
    
    # We let ONNX generate the output names automatically
    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_FP16),
        opset_version=12,
        input_names=['images'],
        dynamic_axes={'images': {0: 'batch'}},
        do_constant_folding=True,
        verbose=False,
    )

    # ── Step 3: Validate FP16 ─────────────────────────────────
    print("\n[3/3] Validating FP16 model...")
    import onnx
    model_onnx = onnx.load(str(OUTPUT_FP16))
    onnx.checker.check_model(model_onnx)
    fp16_size = OUTPUT_FP16.stat().st_size / (1024 * 1024)
    print(f"      FP16 validation OK — {fp16_size:.1f} MB, {len(model_onnx.graph.output)} outputs")

    print(f"\n{'=' * 55}")
    print(f"  SUCCESS: {OUTPUT_FP16.name}")
    print(f"  Size: {fp16_size:.1f} MB")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    export()
