from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image
import io

from api.app.model.model import DetectionsModel
from api.app.service.fast_sam_service import FastSAMService
from api.app.utils.app_config import AppConfig

app_config = AppConfig()
sam_router = APIRouter(prefix="/sam", tags=["sam"])


@sam_router.post("")
async def sam(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640)
) -> DetectionsModel:
    if image_file.filename.rsplit('.', 1)[1].lower() not in ["jpeg", "png", "jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file type")

    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = FastSAMService(model_path=app_config.fast_sam_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=detections.mask.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=None,
        class_name_mapping=None
    )
