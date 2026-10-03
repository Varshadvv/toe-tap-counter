from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # detection model, not -pose
model.export(format="tflite", imgsz=640)