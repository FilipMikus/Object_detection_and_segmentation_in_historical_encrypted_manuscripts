import io
from typing import Union, Literal

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import UploadFile, File, HTTPException
from supervision import Detections

from api.app.model.model import DetectionsModel, ExplorationsModel


async def image_bytes_to_array(image_file: Union[UploadFile, File]) -> np.ndarray:
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)
    return image_array


def validate_file_type(file: Union[UploadFile, File], valid_extensions: list[str]):
    if file.filename.rsplit('.', 1)[1].lower() not in valid_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Valid file types: " + str(valid_extensions))


def transform_detections_to_detections_model(detections: Detections) -> DetectionsModel:
    return DetectionsModel(
        xyxy=np_array_to_list(detections.xyxy),
        mask=np_array_to_list(detections.mask),
        confidence=np_array_to_list(detections.confidence),
        class_id=np_array_to_list(detections.class_id),
        # TODO
        class_name_mapping={}
    )


def transform_explorer_dataframe_to_exploration_model(
        explorer_df: pd.DataFrame,
        split: Literal["train", "test", "valid"]
) -> ExplorationsModel:
    return ExplorationsModel(
        split=split,
        image_file=np_array_to_list(explorer_df["im_file"].to_numpy()),
        vector=np_array_to_list(explorer_df["vector"].to_numpy()),
        class_id=np_array_to_list(explorer_df["cls"].to_numpy()),
        class_name=np_array_to_list(explorer_df["labels"].to_numpy()),
        xyxy=np_array_to_list(explorer_df["bboxes"].to_numpy()),
        mask=np_array_to_list(explorer_df["bboxes"].to_numpy()),
        distance=np_array_to_list(explorer_df["_distance"].to_numpy()) if "_distance" in explorer_df.columns else None
    )


def np_array_to_list(array: np.ndarray) -> list:
    if isinstance(array, np.ndarray):
        return np_array_to_list(array.tolist())
    elif isinstance(array, list):
        return [np_array_to_list(item) for item in array]
    elif isinstance(array, tuple):
        return tuple(np_array_to_list(item) for item in array)
    else:
        return array
