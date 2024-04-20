import numpy as np
from supervision import Detections
from ultralytics import YOLO

from api.app.service.model_service import ModelService


class YOLOv8Service(ModelService):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = YOLO(model=self.model_path)

    def predict(self, image: np.ndarray, confidence: float = 0.25, iou: float = 0.7,
                image_size: int = 640) -> Detections:
        return Detections.from_ultralytics(
            self.model.predict(source=image, conf=confidence, iou=iou, imgsz=image_size)[0]
        )
