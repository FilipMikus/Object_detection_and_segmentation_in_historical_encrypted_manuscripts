from typing import Tuple

import numpy as np
from supervision import Detections, InferenceSlicer
from ultralytics import YOLO

from api.app.service.sahi_model_service import SAHIModelService


class SAHIYOLOv8Service(SAHIModelService):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = YOLO(model=self.model_path)

    def predict(
            self,
            image: np.ndarray, confidence: float = 0.25, iou: float = 0.7, image_size: int = 640,
            slice_size: Tuple[int, int] = (320, 320), slice_overlap_ratio: Tuple[float, float] = (0.2, 0.2)
    ) -> Detections:
        sahi_slicer = InferenceSlicer(
            callback=lambda x: Detections.from_ultralytics(
                self.model.predict(x, conf=confidence, iou=iou, imgsz=image_size)[0]
            ),
            slice_wh=slice_size,
            overlap_ratio_wh=slice_overlap_ratio
        )
        return sahi_slicer(image=image)
