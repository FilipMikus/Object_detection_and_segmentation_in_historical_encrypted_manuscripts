from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image
import io

from api.app.model.model import DetectionsModel
from api.app.service.rt_detr_service import RTDETRService
from api.app.utils.app_config import AppConfig

app_config = AppConfig()
rt_detr_router = APIRouter(prefix="/rt_detr", tags=["rt_detr"])


@rt_detr_router.post("/digits/")
async def rt_detr_digits(
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

    service = RTDETRService(model_path=app_config.rt_detr_digits_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=None,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name_mapping={}
    )


@rt_detr_router.post("/glyphs/")
async def rt_detr_glyphs(
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

    service = RTDETRService(model_path=app_config.rt_detr_glyphs_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=None,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name_mapping={}
    )
