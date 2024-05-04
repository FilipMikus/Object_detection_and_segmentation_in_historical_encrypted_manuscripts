from supervision import Detections
from ultralytics import FastSAM

from api.app.service.base.model_service import ModelService


class FastSAMService(ModelService):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = FastSAM(model=self.model_path)

    def predict(self, image, confidence=0.25, iou=0.7, image_size=640) -> Detections:
        return Detections.from_ultralytics(
            self.model.predict(source=image, conf=confidence, iou=iou, imgsz=image_size)[0]
        )
