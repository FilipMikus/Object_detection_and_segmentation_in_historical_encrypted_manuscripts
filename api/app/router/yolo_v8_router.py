from typing import Annotated
import http

from fastapi import APIRouter, File, UploadFile, Form, Response
from ultralytics import YOLO
from PIL import Image
import io

yolo_v8_router = APIRouter(prefix="/yolo_v8", tags=["yolo_v8"])


@yolo_v8_router.post("")
async def yolo_v8(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    return Response(content={}, media_type="application/json")


@yolo_v8_router.post("/digits/")
async def yolo_v8_digits(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    model = YOLO("api/resources/weights/yolo_v8_digits_best.pt")
    detections = model.predict(image)

    return Response(content=detections[0].tojson(), media_type="application/json")


@yolo_v8_router.post("/glyphs/")
async def yolo_v8_glyphs(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    model = YOLO("api/resources/weights/yolo_v8_glyphs_best.pt")
    detections = model.predict(image)

    return Response(content=detections[0].tojson(), media_type="application/json")
