from typing import Annotated
import http

from fastapi import APIRouter, File, UploadFile, Form, Response
import io

# TODO
data_converter_router = APIRouter(prefix="/data/converter", tags=["data_converter"])
