from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weights and train
model = YOLO("yolov9s.pt")
results = model.train(data="datasets/wave/data.yaml", epochs=100, imgsz=640, workers=0,  batch=2, patience=20,device=0,)