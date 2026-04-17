from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Export the model to TensorRT
model.export(format="engine", device=0, half=True)  # dla:0 or dla:1 corresponds to the DLA cores

# Load the exported TensorRT model
trt_model = YOLO("yolo26n.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")