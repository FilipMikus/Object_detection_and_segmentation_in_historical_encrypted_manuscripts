import uvicorn as uvicorn_server
from fastapi import FastAPI

from api.app.router import (data_annotator_router, data_explorer_router,
                            rt_detr_router, sahi_router, fast_sam_router,
                            yolo_v8_router, yolo_v9_router, yolo_world_v2_router)
from api.app.utils.app_config import AppConfig

app_config = AppConfig()
app = FastAPI(title=app_config.name, version=app_config.version)
app.include_router(data_annotator_router.data_annotator_router)
app.include_router(data_explorer_router.data_explorer_router)
app.include_router(rt_detr_router.rt_detr_router)
app.include_router(sahi_router.sahi_router)
app.include_router(fast_sam_router.fast_sam_router)
app.include_router(yolo_v8_router.yolo_v8_router)
app.include_router(yolo_v9_router.yolo_v9_router)
app.include_router(yolo_world_v2_router.yolo_world_v2_router)

if __name__ == '__main__':
    uvicorn_server.run(app, host=app_config.host, port=app_config.port)
