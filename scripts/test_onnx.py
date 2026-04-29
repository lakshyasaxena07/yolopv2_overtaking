import onnxruntime as ort
import numpy as np
from pathlib import Path

model_path = str(Path(__file__).parent.parent / 'models' / 'yolopv2_fp16.onnx')
sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
dummy = np.random.randn(1,3,384,384).astype(np.float16)
outs = sess.run(None, {'images': dummy})
print('Outputs returned:', len(outs))
for i, o in enumerate(outs):
    if hasattr(o, 'shape'):
        print(f'out[{i}] shape: {o.shape}')
    elif isinstance(o, list):
        print(f'out[{i}] list of length {len(o)}, first element type: {type(o[0])}')
        if hasattr(o[0], 'shape'):
            print(f'out[{i}] first element shape: {o[0].shape}')
    else:
        print(f'out[{i}] type: {type(o)}')
