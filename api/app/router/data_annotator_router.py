import os
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Annotated
from zipfile import ZipFile

import aiofiles
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse

from api.app.service.data_annotator_service import DataAnnotatorService
from api.app.utils.app_config import AppConfig
from api.app.utils.utils import validate_file_type

app_config = AppConfig()
data_annotator_router = APIRouter(prefix="/data/annotator", tags=["data_annotator"])


@data_annotator_router.post("/digits/")
async def data_annotator_digits(dataset_images_file: Annotated[UploadFile, File()]) -> FileResponse:
    validate_file_type(file=dataset_images_file, valid_extensions=["zip"])

    with TemporaryDirectory() as temp_dir:

        async with aiofiles.open(os.path.join(temp_dir, dataset_images_file.filename), "wb") as zipped_temp_file:
            await zipped_temp_file.write(await dataset_images_file.read())
            with ZipFile(os.path.join(temp_dir, dataset_images_file.filename), "r") as zip_file:
                zip_file.extractall(temp_dir)

                with NamedTemporaryFile(delete=False, suffix=".zip") as labels_temp_file:
                    with TemporaryDirectory() as labels_temp_dir:

                        service = DataAnnotatorService(
                            dataset_images_path=os.path.join(
                                temp_dir,
                                os.path.splitext(dataset_images_file.filename)[0]
                            ),
                            output_dataset_labels_path=os.path.join(labels_temp_dir),
                            model_path=app_config.yolo_v8_annotator_digits_source,
                            sam_model_path=app_config.sam_annotator_source
                        )
                        service.annotate()

                        with ZipFile(os.path.join(labels_temp_file.name), "w") as labels_zip_dir:
                            for folder_name, sub_folders, file_names in os.walk(os.path.join(labels_temp_dir)):
                                for file_name in file_names:
                                    file_path = os.path.join(folder_name, file_name)
                                    labels_zip_dir.write(file_path, os.path.basename(file_path))

                            return FileResponse(
                                os.path.join(labels_temp_file.name),
                                background=BackgroundTasks(
                                    labels_temp_file.close()
                                )
                            )


@data_annotator_router.post("/glyphs/")
async def data_annotator_glyphs(dataset_images_file: Annotated[UploadFile, File()]) -> FileResponse:
    validate_file_type(file=dataset_images_file, valid_extensions=["zip"])

    with TemporaryDirectory() as temp_dir:

        async with aiofiles.open(os.path.join(temp_dir, dataset_images_file.filename), "wb") as zipped_temp_file:
            await zipped_temp_file.write(await dataset_images_file.read())
            with ZipFile(os.path.join(temp_dir, dataset_images_file.filename), "r") as zip_file:
                zip_file.extractall(temp_dir)

                with NamedTemporaryFile(delete=False, suffix=".zip") as labels_temp_file:
                    with TemporaryDirectory() as labels_temp_dir:

                        service = DataAnnotatorService(
                            dataset_images_path=os.path.join(
                                temp_dir,
                                os.path.splitext(dataset_images_file.filename)[0]
                            ),
                            output_dataset_labels_path=os.path.join(labels_temp_dir),
                            model_path=app_config.yolo_v8_annotator_glyphs_source,
                            sam_model_path=app_config.sam_annotator_source
                        )
                        service.annotate()

                        with ZipFile(os.path.join(labels_temp_file.name), "w") as labels_zip_dir:
                            for folder_name, sub_folders, file_names in os.walk(os.path.join(labels_temp_dir)):
                                for file_name in file_names:
                                    file_path = os.path.join(folder_name, file_name)
                                    labels_zip_dir.write(file_path, os.path.basename(file_path))

                            return FileResponse(
                                os.path.join(labels_temp_file.name),
                                background=BackgroundTasks(
                                    labels_temp_file.close()
                                )
                            )
