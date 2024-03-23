import os
from tempfile import TemporaryDirectory
from typing import Annotated, Literal
from zipfile import ZipFile

import aiofiles
from fastapi import APIRouter, File, UploadFile, Form

from api.app.model.model import ExplorationsModel
from api.app.service.data_explorer_service import DataExplorerService
from api.app.utils.app_config import AppConfig
from api.app.utils.utils import transform_explorer_dataframe_to_exploration_model, validate_file_type, \
    image_bytes_to_array

app_config = AppConfig()
data_explorer_router = APIRouter(prefix="/data/explorer", tags=["data_explorer"])


@data_explorer_router.post("/similarity/")
async def data_explorer_similarity(
        image_file: Annotated[UploadFile, File()],
        dataset_file: Annotated[UploadFile, File()],
        split_parameter: Annotated[Literal["train", "test", "valid"], Form] = Form("train")
) -> ExplorationsModel:
    validate_file_type(file=image_file, valid_extensions=["jpeg", "png", "jpg"])
    validate_file_type(file=dataset_file, valid_extensions=["zip"])

    with TemporaryDirectory() as temp_dir:
        async with aiofiles.open(os.path.join(temp_dir, dataset_file.filename), "wb") as zipped_temp_file:
            await zipped_temp_file.write(await dataset_file.read())
            with ZipFile(os.path.join(temp_dir, dataset_file.filename), "r") as zip_file:
                zip_file.extractall(temp_dir)

                image_array = await image_bytes_to_array(image_file)

                service = DataExplorerService(
                    dataset_yaml_path=os.path.join(temp_dir, "test_yolo_dataset", "data.yaml"),
                    model_path=app_config.yolo_v8_explorer_source
                )
                explorer_df = service.explore_by_similarity(
                    image=image_array,
                    split=split_parameter
                )

                return transform_explorer_dataframe_to_exploration_model(explorer_df=explorer_df, split=split_parameter)


@data_explorer_router.post("/query/")
async def data_explorer_query(
        dataset_file: Annotated[UploadFile, File()],
        query_parameter: Annotated[str, Form()],
        split_parameter: Annotated[Literal["train", "test", "valid"], Form] = Form("train")
) -> ExplorationsModel:
    validate_file_type(file=dataset_file, valid_extensions=["zip"])

    with TemporaryDirectory() as temp_dir:
        async with aiofiles.open(os.path.join(temp_dir, dataset_file.filename), "wb") as zipped_temp_file:
            await zipped_temp_file.write(await dataset_file.read())
            with ZipFile(os.path.join(temp_dir, dataset_file.filename), "r") as zip_file:
                zip_file.extractall(temp_dir)

                service = DataExplorerService(
                    dataset_yaml_path=os.path.join(temp_dir, "test_yolo_dataset", "data.yaml"),
                    model_path=app_config.yolo_v8_explorer_source
                )
                explorer_df = service.explore_by_query(
                    query=query_parameter,
                    split=split_parameter
                )

                return transform_explorer_dataframe_to_exploration_model(explorer_df=explorer_df, split=split_parameter)


@data_explorer_router.post("/prompt/")
async def data_explorer_prompt(
        dataset_file: Annotated[UploadFile, File()],
        prompt_parameter: Annotated[str, Form()],
        split_parameter: Annotated[Literal["train", "test", "valid"], Form] = Form("train")
) -> ExplorationsModel:
    validate_file_type(file=dataset_file, valid_extensions=["zip"])

    raise NotImplementedError("This endpoint is not implemented yet")

    with TemporaryDirectory() as temp_dir:
        async with aiofiles.open(os.path.join(temp_dir, dataset_file.filename), "wb") as zipped_temp_file:
            await zipped_temp_file.write(await dataset_file.read())
            with ZipFile(os.path.join(temp_dir, dataset_file.filename), "r") as zip_file:
                zip_file.extractall(temp_dir)

                service = DataExplorerService(
                    dataset_yaml_path=os.path.join(temp_dir, "test_yolo_dataset", "data.yaml"),
                    model_path=app_config.yolo_v8_explorer_source
                )
                explorer_df = service.explore_by_prompt(
                    prompt=prompt_parameter,
                    split=split_parameter
                )

                return transform_explorer_dataframe_to_exploration_model(explorer_df=explorer_df, split=split_parameter)
