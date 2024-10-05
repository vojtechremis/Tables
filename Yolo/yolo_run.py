from ultralytics import YOLO
# Load a YOLOv8 model (you can also use yolov8n.pt, yolov8m.pt, etc. depending on the size of the model)
model = YOLO('yolov8x.pt') #n,s,m,l,x jsou přípony modelů
model.info()
# Train the model
#je potřeba změnit adresy v data.yaml
model.train(data='/mnt/lustre/helios-home/remisvoj/Projects/Tables/Yolo/data.yaml', epochs=100, imgsz=640)