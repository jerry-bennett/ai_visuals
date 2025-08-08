from ultralytics import YOLO

det_model = YOLO("yolov8n.pt")
det_model.export(format="onnx", imgsz=320)

pose_model = YOLO("yolov8n-pose.pt")
pose_model.export(format="onnx", imgsz=320)
