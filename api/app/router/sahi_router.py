from typing import Annotated
import http

import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, Response
from PIL import Image
import io

from api.app.service.sahi_rt_detr_service import SAHIRTDETRService
from api.app.service.sahi_yolo_v8_service import SAHIYOLOv8Service

sahi_router = APIRouter(prefix="/sahi", tags=["sahi"])


@sahi_router.post("/yolo_v8/")
async def sahi_yolo_v8(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    return Response(content={}, media_type="application/json")


@sahi_router.post("/yolo_v8/digits/")
async def sahi_yolo_v8_digits(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = SAHIYOLOv8Service(model_path="api/resources/weights/yolo_v8_digits_best.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")


@sahi_router.post("/yolo_v8/glyphs/")
async def sahi_yolo_v8_glyphs(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = SAHIYOLOv8Service(model_path="api/resources/weights/yolo_v8_glyphs_best.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")


@sahi_router.post("/rt_detr/")
async def sahi_rt_detr(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    return Response(content={}, media_type="application/json")


@sahi_router.post("/rt_detr/digits/")
async def sahi_rt_detr_digits(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = SAHIRTDETRService(model_path="api/resources/weights/rt_detr_digits_best.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")


@sahi_router.post("/rt_detr/glyphs/")
async def sahi_rt_detr_glyphs(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = SAHIRTDETRService(model_path="api/resources/weights/rt_detr_glyphs_best.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")
