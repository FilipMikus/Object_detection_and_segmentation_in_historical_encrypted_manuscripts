from typing import Annotated
import http

from fastapi import APIRouter, File, UploadFile, Form, Response
from ultralytics import RTDETR
from PIL import Image
import io

rt_detr_router = APIRouter(prefix="/rt_detr", tags=["rt_detr"])


@rt_detr_router.post("")
async def rt_detr(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    return Response(content={}, media_type="application/json")


@rt_detr_router.post("/digits/")
async def rt_detr_digits(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    model = RTDETR("api/resources/weights/rt_detr_digits_best.pt")
    detections = model.predict(image)

    return Response(content=detections[0].tojson(), media_type="application/json")


@rt_detr_router.post("/glyphs/")
async def rt_detr_glyphs(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    model = RTDETR("api/resources/weights/rt_detr_glyphs_best.pt")
    detections = model.predict(image)

    return Response(content=detections[0].tojson(), media_type="application/json")
