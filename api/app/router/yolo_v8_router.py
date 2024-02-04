from typing import Annotated
import http

import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, Response
from PIL import Image
import io

from api.app.service.yolo_v8_service import YOLOv8Service

yolo_v8_router = APIRouter(prefix="/yolo_v8", tags=["yolo_v8"])


@yolo_v8_router.post("")
async def yolo_v8(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    return Response(content={}, media_type="application/json")


@yolo_v8_router.post("/digits/")
async def yolo_v8_digits(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = YOLOv8Service(model_path="api/resources/weights/yolo_v8_digits_best.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")


@yolo_v8_router.post("/glyphs/")
async def yolo_v8_glyphs(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = YOLOv8Service(model_path="api/resources/weights/yolo_v8_glyphs_best.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")
