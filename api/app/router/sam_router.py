from typing import Annotated
import http

import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, Response
from PIL import Image
import io

from api.app.service.fast_sam_service import FastSAMService

sam_router = APIRouter(prefix="/sam", tags=["sam"])


@sam_router.post("")
async def sam(file: Annotated[UploadFile, File()], data: Annotated[str, Form()]):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    service = FastSAMService(model_path="api/resources/weights/FastSAM-s.pt")
    detections = service.predict(image_array)

    return Response(content={}, media_type="application/json")
