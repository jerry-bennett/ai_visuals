import onnxruntime as ort

session = ort.InferenceSession("yolov8n.onnx")
input_shape = session.get_inputs()[0].shape
print("Model input shape:", input_shape)
