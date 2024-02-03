from typing import Annotated
import http

from fastapi import APIRouter, File, UploadFile, Form, Response
from PIL import Image
import io

sam_router = APIRouter(prefix="/sam", tags=["sam"])


@sam_router.post("")
async def sam(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    return Response(content={}, media_type="application/json")
