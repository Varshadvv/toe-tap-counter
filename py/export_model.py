from ultralytics import YOLO

# Load the pose model
model = YOLO("yolo26n-pose.pt")

# Export to TensorFlow Lite
model.export(
    format="tflite",
    imgsz=640
)