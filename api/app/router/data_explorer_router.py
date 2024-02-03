from typing import Annotated
import http

from fastapi import APIRouter, File, UploadFile, Form, Response
import io

data_explorer_router = APIRouter(prefix="/data/explorer", tags=["data_explorer"])
