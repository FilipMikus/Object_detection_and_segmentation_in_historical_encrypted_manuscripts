from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image
import io

from api.app.model.model import DetectionsModel
from api.app.service.sahi_rt_detr_service import SAHIRTDETRService
from api.app.service.sahi_yolo_v8_service import SAHIYOLOv8Service
from api.app.utils.app_config import AppConfig

app_config = AppConfig()
sahi_router = APIRouter(prefix="/sahi", tags=["sahi"])


@sahi_router.post("/yolo_v8/digits/")
async def sahi_yolo_v8_digits(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        slice_size_width_parameter: Annotated[int, Form] = Form(320),
        slice_size_height_parameter: Annotated[int, Form] = Form(320),
        slice_overlap_ratio_width_parameter: Annotated[float, Form] = Form(0.2),
        slice_overlap_ratio_height_parameter: Annotated[float, Form] = Form(0.2)
) -> DetectionsModel:
    if image_file.filename.rsplit('.', 1)[1].lower() not in ["jpeg", "png", "jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file type")

    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = SAHIYOLOv8Service(model_path=app_config.yolo_v8_digits_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter,
        slice_size=(slice_size_width_parameter, slice_size_height_parameter),
        slice_overlap_ratio=(slice_overlap_ratio_width_parameter, slice_overlap_ratio_height_parameter)
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=detections.mask.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name_mapping={}
    )


@sahi_router.post("/yolo_v8/glyphs/")
async def sahi_yolo_v8_glyphs(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        slice_size_width_parameter: Annotated[int, Form] = Form(320),
        slice_size_height_parameter: Annotated[int, Form] = Form(320),
        slice_overlap_ratio_width_parameter: Annotated[float, Form] = Form(0.2),
        slice_overlap_ratio_height_parameter: Annotated[float, Form] = Form(0.2)
) -> DetectionsModel:
    if image_file.filename.rsplit('.', 1)[1].lower() not in ["jpeg", "png", "jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file type")

    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = SAHIYOLOv8Service(model_path=app_config.yolo_v8_glyphs_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter,
        slice_size=(slice_size_width_parameter, slice_size_height_parameter),
        slice_overlap_ratio=(slice_overlap_ratio_width_parameter, slice_overlap_ratio_height_parameter)
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=detections.mask.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name_mapping={}
    )


@sahi_router.post("/rt_detr/digits/")
async def sahi_rt_detr_digits(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        slice_size_width_parameter: Annotated[int, Form] = Form(320),
        slice_size_height_parameter: Annotated[int, Form] = Form(320),
        slice_overlap_ratio_width_parameter: Annotated[float, Form] = Form(0.2),
        slice_overlap_ratio_height_parameter: Annotated[float, Form] = Form(0.2)
) -> DetectionsModel:
    if image_file.filename.rsplit('.', 1)[1].lower() not in ["jpeg", "png", "jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file type")

    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = SAHIRTDETRService(model_path=app_config.rt_detr_digits_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter,
        slice_size=(slice_size_width_parameter, slice_size_height_parameter),
        slice_overlap_ratio=(slice_overlap_ratio_width_parameter, slice_overlap_ratio_height_parameter)
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=detections.mask.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name_mapping={}
    )


@sahi_router.post("/rt_detr/glyphs/")
async def sahi_rt_detr_glyphs(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        slice_size_width_parameter: Annotated[int, Form] = Form(320),
        slice_size_height_parameter: Annotated[int, Form] = Form(320),
        slice_overlap_ratio_width_parameter: Annotated[float, Form] = Form(0.2),
        slice_overlap_ratio_height_parameter: Annotated[float, Form] = Form(0.2)
) -> DetectionsModel:
    if image_file.filename.rsplit('.', 1)[1].lower() not in ["jpeg", "png", "jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file type")

    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = SAHIRTDETRService(model_path=app_config.rt_detr_glyphs_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter,
        slice_size=(slice_size_width_parameter, slice_size_height_parameter),
        slice_overlap_ratio=(slice_overlap_ratio_width_parameter, slice_overlap_ratio_height_parameter)
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=detections.mask.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name_mapping={}
    )
