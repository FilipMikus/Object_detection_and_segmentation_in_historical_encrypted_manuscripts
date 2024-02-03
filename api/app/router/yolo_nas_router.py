from typing import Annotated
import http

from fastapi import APIRouter, File, UploadFile, Form, Response
from PIL import Image
import io

yolo_nas_router = APIRouter(prefix="/yolo_nas", tags=["yolo_nas"])


@yolo_nas_router.post("")
async def yolo_nas(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    return Response(content={}, media_type="application/json")


@yolo_nas_router.post("/digits/")
async def yolo_nas_digits(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    return Response(content={}, media_type="application/json")


@yolo_nas_router.post("/glyphs/")
async def yolo_nas_glyphs(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    return Response(content={}, media_type="application/json")
