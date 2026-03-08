import ultralytics.nn.modules as modules
import inspect



print(f"DEBUG: File BiFPN đang được load từ: {inspect.getfile(modules.BiFPN)}")
import ultralytics.nn.tasks as tasks
print(f"DEBUG: File tasks.py đang thực sự chạy tại: {tasks.__file__}")
from ultralytics import YOLO

model = YOLO(r'cfg/models/11/yolo11_test3.yaml')
model.info()

model.train(
        data=r'datasets/apple/data.yaml',
        imgsz=640,
        batch=8,                         # Ổn định hơn, ít dao động mAP
        epochs=10,
        cache=False,
        amp=False,                        # FP32 cho độ chính xác cao nhất
        optimizer='SGD',
        patience=10,
        save_period=10,
        seed=42,
        project='runs/train',
        name='exp',
        workers=0,
        device='cpu',
        val=True,
    )   
#
# # Load a pretrained YOLO11n model
# model = YOLO("yolo11n.pt")
#
# # Train the model on the COCO8 dataset for 100 epochs
# train_results = model.train(
#     data="coco8.yaml",  # Path to dataset configuration file
#     epochs=100,  # Number of training epochs
#     imgsz=640,  # Image size for training
#     device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
# )
#
# # Evaluate the model's performance on the validation set
# metrics = model.val()
#
# # Perform object detection on an image
# results = model("path/to/image.jpg")  # Predict on an image
# results[0].show()  # Display results
#
# # Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model