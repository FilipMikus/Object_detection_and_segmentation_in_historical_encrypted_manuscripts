from typing import Annotated

from fastapi import APIRouter, File, UploadFile, Form

from api.app.model.model import DetectionsModel
from api.app.service.sahi_rt_detr_service import SAHIRTDETRService
from api.app.service.sahi_yolo_v8_service import SAHIYOLOv8Service
from api.app.service.sahi_yolo_v9_service import SAHIYOLOv9Service
from api.app.utils.app_config import AppConfig
from api.app.utils.utils import validate_file_type, image_bytes_to_array

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
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

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
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
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
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

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
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
    )


@sahi_router.post("/yolo_v9/digits/")
async def sahi_yolo_v9_digits(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        slice_size_width_parameter: Annotated[int, Form] = Form(320),
        slice_size_height_parameter: Annotated[int, Form] = Form(320),
        slice_overlap_ratio_width_parameter: Annotated[float, Form] = Form(0.2),
        slice_overlap_ratio_height_parameter: Annotated[float, Form] = Form(0.2)
) -> DetectionsModel:
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

    service = SAHIYOLOv9Service(model_path=app_config.yolo_v9_digits_source)
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
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
    )


@sahi_router.post("/yolo_v9/glyphs/")
async def sahi_yolo_v9_glyphs(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        slice_size_width_parameter: Annotated[int, Form] = Form(320),
        slice_size_height_parameter: Annotated[int, Form] = Form(320),
        slice_overlap_ratio_width_parameter: Annotated[float, Form] = Form(0.2),
        slice_overlap_ratio_height_parameter: Annotated[float, Form] = Form(0.2)
) -> DetectionsModel:
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

    service = SAHIYOLOv9Service(model_path=app_config.yolo_v9_glyphs_source)
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
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
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
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

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
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
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
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

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
        mask=detections.mask.tolist() if detections.mask is not None else detections.mask,
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"]
    )
