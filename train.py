import ultralytics.nn.modules as modules
import inspect



print(f"DEBUG: File BiFPN đang được load từ: {inspect.getfile(modules.BiFPN)}")
import ultralytics.nn.tasks as tasks
print(f"DEBUG: File tasks.py đang thực sự chạy tại: {tasks.__file__}")
from ultralytics import YOLO

model = YOLO(r'cfg/models/11/yolo11_CA_CBAM_SIMAM.yaml')
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
