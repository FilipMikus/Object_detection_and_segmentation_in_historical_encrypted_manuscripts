from abc import abstractmethod, ABC
from typing import Tuple, Any

import numpy as np
from supervision import Detections


class SAHIModelService(ABC):
    model_path: str
    model: Any

    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def predict(
            self, image: np.ndarray, confidence: float = 0.25, iou: float = 0.7, image_size: Tuple[int, int] = 640,
            slice_size: Tuple[int, int] = (320, 320), slice_overlap_ratio: Tuple[float, float] = (0.2, 0.2)
    ) -> Detections:
        pass
