import onnxruntime as ort
import onnx
import numpy as np

img = np.random.rand(1, 3, 832, 832).astype(np.float32)
providers = [('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}), 'CPUExecutionProvider']

def run_model(path, name):
    model = onnx.load(path)
    model.graph.name = name
    for node in model.graph.node:
        if node.name:
            node.name = node.name + "_" + name
    sess = ort.InferenceSession(model.SerializeToString(), providers=providers)
    return sess.run(None, {sess.get_inputs()[0].name: img})[0]

out1 = run_model('../models/small_baseline.onnx', 'm1')
out2 = run_model('../models/small_feature_kd.onnx', 'm2')
print("Outputs match?", np.allclose(out1, out2))
