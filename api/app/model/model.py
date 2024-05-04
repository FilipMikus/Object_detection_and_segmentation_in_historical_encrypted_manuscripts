from typing import Optional, List

from pydantic import BaseModel


class DetectionsModel(BaseModel):
    xyxy: List
    mask: Optional[List] = None
    confidence: Optional[List] = None
    class_id: Optional[List] = None
    class_name: Optional[List] = None


class TranscriptionsModel(BaseModel):
    xyxy: List
    confidence: List
    class_id: List
    class_name: List
    transcription_class_id: str
    transcription_class_name: str
    lines_transcription_class_id: List
    lines_transcription_class_name: List


class ExplorationsModel(BaseModel):
    split: str
    image_file: List
    vector: List
    class_id: List
    class_name: List
    xyxy: Optional[List] = None
    mask: Optional[List] = None
    distance: Optional[List] = None
