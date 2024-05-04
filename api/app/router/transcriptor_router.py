from typing import Annotated

from fastapi import APIRouter, File, UploadFile, Form

from api.app.model.model import TranscriptionsModel
from api.app.service.sahi_yolo_v8_service import SAHIYOLOv8Service
from api.app.service.transcriptor_dbscan_service import TranscriptorDBSCANService
from api.app.service.yolo_v8_service import YOLOv8Service
from api.app.utils.app_config import AppConfig
from api.app.utils.utils import validate_file_type, image_bytes_to_array

app_config = AppConfig()
transcriptor_router = APIRouter(prefix="/transcriptor", tags=["transcriptor"])


@transcriptor_router.post("/yolo_v8/digits/")
async def transcriptor_yolo_v8_digits(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        number_of_lines_parameter: Annotated[int, Form] = Form(3)
) -> TranscriptionsModel:
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

    service = YOLOv8Service(model_path=app_config.yolo_v8_digits_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter
    )

    transcriptor_service = TranscriptorDBSCANService(detections=detections)
    transcriptor_service.separate_lines(number_of_lines=number_of_lines_parameter)

    return TranscriptionsModel(
        xyxy=detections.xyxy.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"],
        transcription_class_id=transcriptor_service.transcript_class_id(),
        transcription_class_name=transcriptor_service.transcript_class_name(),
        lines_transcription_class_id=transcriptor_service.transcript_lines_class_id(),
        lines_transcription_class_name=transcriptor_service.transcript_lines_class_name()
    )


@transcriptor_router.post("/yolo_v8/glyphs/")
async def transcriptor_yolo_v8_glyphs(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        number_of_lines_parameter: Annotated[int, Form] = Form(3)
) -> TranscriptionsModel:
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])

    image_array = await image_bytes_to_array(image_file)

    service = YOLOv8Service(model_path=app_config.yolo_v8_glyphs_source)
    detections = service.predict(
        image=image_array,
        confidence=confidence_parameter,
        iou=iou_parameter,
        image_size=image_size_parameter
    )

    transcriptor_service = TranscriptorDBSCANService(detections=detections)
    transcriptor_service.separate_lines(number_of_lines=number_of_lines_parameter)

    return TranscriptionsModel(
        xyxy=detections.xyxy.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"],
        transcription_class_id=transcriptor_service.transcript_class_id(),
        transcription_class_name=transcriptor_service.transcript_class_name(),
        lines_transcription_class_id=transcriptor_service.transcript_lines_class_id(),
        lines_transcription_class_name=transcriptor_service.transcript_lines_class_name()
    )


@transcriptor_router.post("/sahi/yolo_v8/digits/")
async def transcriptor_sahi_yolo_v8_digits(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        slice_size_width_parameter: Annotated[int, Form] = Form(320),
        slice_size_height_parameter: Annotated[int, Form] = Form(320),
        slice_overlap_ratio_width_parameter: Annotated[float, Form] = Form(0.2),
        slice_overlap_ratio_height_parameter: Annotated[float, Form] = Form(0.2),
        number_of_lines_parameter: Annotated[int, Form] = Form(17)
) -> TranscriptionsModel:
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

    transcriptor_service = TranscriptorDBSCANService(detections=detections)
    transcriptor_service.separate_lines(number_of_lines=number_of_lines_parameter)

    return TranscriptionsModel(
        xyxy=detections.xyxy.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"],
        transcription_class_id=transcriptor_service.transcript_class_id(),
        transcription_class_name=transcriptor_service.transcript_class_name(),
        lines_transcription_class_id=transcriptor_service.transcript_lines_class_id(),
        lines_transcription_class_name=transcriptor_service.transcript_lines_class_name()
    )


@transcriptor_router.post("/sahi/yolo_v8/glyphs/")
async def transcriptor_sahi_yolo_v8_glyphs(
        image_file: Annotated[UploadFile, File()],
        confidence_parameter: Annotated[float, Form] = Form(0.25),
        iou_parameter: Annotated[float, Form] = Form(0.7),
        image_size_parameter: Annotated[int, Form] = Form(640),
        slice_size_width_parameter: Annotated[int, Form] = Form(320),
        slice_size_height_parameter: Annotated[int, Form] = Form(320),
        slice_overlap_ratio_width_parameter: Annotated[float, Form] = Form(0.2),
        slice_overlap_ratio_height_parameter: Annotated[float, Form] = Form(0.2),
        number_of_lines_parameter: Annotated[int, Form] = Form(17)
) -> TranscriptionsModel:
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

    transcriptor_service = TranscriptorDBSCANService(detections=detections)
    transcriptor_service.separate_lines(number_of_lines=number_of_lines_parameter)

    return TranscriptionsModel(
        xyxy=detections.xyxy.tolist(),
        confidence=detections.confidence.tolist(),
        class_id=detections.class_id.tolist(),
        class_name=detections.data["class_name"],
        transcription_class_id=transcriptor_service.transcript_class_id(),
        transcription_class_name=transcriptor_service.transcript_class_name(),
        lines_transcription_class_id=transcriptor_service.transcript_lines_class_id(),
        lines_transcription_class_name=transcriptor_service.transcript_lines_class_name()
    )
