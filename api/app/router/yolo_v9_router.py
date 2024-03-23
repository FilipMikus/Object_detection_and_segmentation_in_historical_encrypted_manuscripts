from typing import Annotated

from fastapi import APIRouter, File, UploadFile, Form

from api.app.model.model import DetectionsModel
from api.app.service.yolo_v9_service import YOLOv9Service
from api.app.utils.app_config import AppConfig
from api.app.utils.utils import validate_file_type, image_bytes_to_array

app_config = AppConfig()
yolo_v9_router = APIRouter(prefix="/yolo_v9", tags=["yolo_v9"])


@yolo_v9_router.post("/digits/")
async def yolo_v9_digits(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640)
) -> DetectionsModel:
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

    service = YOLOv9Service(model_path=app_config.yolo_v9_digits_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
    )


@yolo_v9_router.post("/glyphs/")
async def yolo_v9_glyphs(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640)
) -> DetectionsModel:
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

    service = YOLOv9Service(model_path=app_config.yolo_v9_glyphs_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter
    )

    return DetectionsModel(
        xyxy=detections.xyxy.tolist(),
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
    )
