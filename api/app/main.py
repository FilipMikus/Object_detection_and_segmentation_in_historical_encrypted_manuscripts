from fastapi import FastAPI

from api.app.router import (
    data_converter_router, data_explorer_router, rt_detr_router,
    sahi_router, sam_router, yolo_nas_router, yolo_v8_router
)


app = FastAPI()


app.include_router(data_converter_router.data_converter_router)
app.include_router(data_explorer_router.data_explorer_router)
app.include_router(rt_detr_router.rt_detr_router)
app.include_router(sahi_router.sahi_router)
app.include_router(sam_router.sam_router)
app.include_router(yolo_nas_router.yolo_nas_router)
app.include_router(yolo_v8_router.yolo_v8_router)
