from pathlib import Path
from typing import Union

from supervision import Detections
from ultralytics import FastSAM

from api.app.service.model_service import ModelService


class FastSAMService(ModelService):

    def __init__(self, model_path: Union[str, Path]):
        super().__init__(model_path)
        self.model = FastSAM(model=self.model_path)

    def predict(self, image, confidence=0.25, iou=0.7, image_size=640) -> Detections:
        return Detections.from_ultralytics(
            self.model.predict(source=image, conf=confidence, iou=iou, imgsz=image_size)[0]
        )
