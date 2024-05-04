from abc import abstractmethod, ABC
from typing import List

from supervision import Detections


class TranscriptorService(ABC):
    detections: Detections
    separated_detections_tuples: List

    def __init__(self, detections: Detections):
        self.detections = detections

    @abstractmethod
    def separate_lines(self):
        pass

    @abstractmethod
    def transcript_class_id(self) -> str:
        if self.separated_detections_tuples is None:
            raise ValueError("Lines must be separated before transcribing. Call separate_lines() method first.")
        pass

    @abstractmethod
    def transcript_class_name(self) -> str:
        if self.separated_detections_tuples is None:
            raise ValueError("Lines must be separated before transcribing. Call separate_lines() method first.")
        pass

    @abstractmethod
    def transcript_lines_class_id(self) -> List:
        if self.separated_detections_tuples is None:
            raise ValueError("Lines must be separated before transcribing. Call separate_lines() method first.")
        pass

    @abstractmethod
    def transcript_lines_class_name(self) -> List:
        if self.separated_detections_tuples is None:
            raise ValueError("Lines must be separated before transcribing. Call separate_lines() method first.")
        pass
