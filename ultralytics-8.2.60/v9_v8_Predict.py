from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("runs/detect/train2/weights/best.pt")


results = model(source = "datasets/wave/train/images/4.png",save=True)