from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from supervision.detection.core import Detections


class ModelService(ABC):
    model_path: str
    model: Any

    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def predict(self, image: np.ndarray, confidence: float = 0.25, iou: float = 0.7,
                image_size: int = 640) -> Detections:
        pass
