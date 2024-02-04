from typing import Annotated
import http

import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, Response
from PIL import Image
import io

from api.app.service.rt_detr_service import RTDETRService

rt_detr_router = APIRouter(prefix="/rt_detr", tags=["rt_detr"])


@rt_detr_router.post("")
async def rt_detr(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    return Response(content={}, media_type="application/json")


@rt_detr_router.post("/digits/")
async def rt_detr_digits(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = RTDETRService(model_path="api/resources/weights/rt_detr_digits_best.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")


@rt_detr_router.post("/glyphs/")
async def rt_detr_glyphs(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = RTDETRService(model_path="api/resources/weights/rt_detr_glyphs_best.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")
