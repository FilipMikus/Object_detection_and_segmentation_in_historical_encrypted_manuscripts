from typing import Union, Literal, Any

import cv2
import numpy as np
import pandas as pd
from fastapi import UploadFile, File, HTTPException
from supervision import Detections

from api.app.model.model import DetectionsModel, ExplorationsModel


async def image_bytes_to_array(image_file: Union[UploadFile, File]) -> np.ndarray:
    contents = await image_file.read()
    image_array = np.fromstring(contents, np.uint8)
    image_array = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    return image_array


def validate_file_type(file: Union[UploadFile, File], valid_extensions: list[str]):
    if file.filename.rsplit('.', 1)[1].lower() not in valid_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Valid file types: " + str(valid_extensions))


def transform_detections_to_detections_model(detections: Detections) -> DetectionsModel:
    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
    )


def transform_explorer_dataframe_to_exploration_model(
        explorer_df: pd.DataFrame,
        split: Literal["train", "test", "valid"]
) -> ExplorationsModel:
    return ExplorationsModel(
        split=split,
        image_file=[file_path.rsplit('/', 1)[1] for file_path in np_array_to_list(explorer_df["im_file"].to_numpy())],
        vector=np_array_to_list(explorer_df["vector"].to_numpy()),
        class_id=np_array_to_list(explorer_df["cls"].to_numpy()),
        class_name=np_array_to_list(explorer_df["labels"].to_numpy()),
        xyxy=np_array_to_list(explorer_df["bboxes"].to_numpy()),
        mask=np_array_to_list(explorer_df["bboxes"].to_numpy()),
        distance=np_array_to_list(explorer_df["_distance"].to_numpy()) if "_distance" in explorer_df.columns else None
    )


def np_array_to_list(array: np.ndarray) -> Any:
    if isinstance(array, np.ndarray):
        return np_array_to_list(array.tolist())
    elif isinstance(array, list):
        return [np_array_to_list(item) for item in array]
    elif isinstance(array, tuple):
        return tuple(np_array_to_list(item) for item in array)
    else:
        return array


def transform_detections_to_xyxy_class_id_class_name_tuple(detections: Detections) -> list[tuple]:
    return [
        (
            detections.xyxy[i][0], detections.xyxy[i][1], detections.xyxy[i][2], detections.xyxy[i][3],
            detections.class_id[i], (detections.data["class_name"])[i]
        ) for i in range(len(detections.xyxy))
    ]
