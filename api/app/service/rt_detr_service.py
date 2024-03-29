from pathlib import Path
from typing import Union

import numpy as np
from supervision import Detections
from ultralytics import RTDETR

from api.app.service.model_service import ModelService


class RTDETRService(ModelService):

    def __init__(self, model_path: Union[str, Path]):
        super().__init__(model_path)
        self.model = RTDETR(model=self.model_path)

    def predict(self, image: np.ndarray, confidence: float = 0.25, iou: float = 0.7,
                image_size: int = 640) -> Detections:
        return Detections.from_ultralytics(
            self.model.predict(source=image, conf=confidence, iou=iou, imgsz=image_size)[0]
        )
