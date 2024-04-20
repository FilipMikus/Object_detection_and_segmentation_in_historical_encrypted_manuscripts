from typing import Annotated

from fastapi import APIRouter, File, UploadFile, Form

from api.app.model.model import DetectionsModel
from api.app.service.fast_sam_service import FastSAMService
from api.app.utils.app_config import AppConfig
from api.app.utils.utils import validate_file_type, image_bytes_to_array

app_config = AppConfig()
fast_sam_router = APIRouter(prefix="/fast_sam", tags=["fast_sam"])


@fast_sam_router.post("")
async def fast_sam(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640)
) -> DetectionsModel:
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

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
