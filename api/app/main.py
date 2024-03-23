import uvicorn as uvicorn_server
from fastapi import FastAPI

from api.app.router import data_explorer_router, rt_detr_router, sahi_router, sam_router, yolo_v8_router, yolo_v9_router
from api.app.utils.app_config import AppConfig

app_config = AppConfig()
app = FastAPI(title=app_config.name, version=app_config.version)
app.include_router(data_explorer_router.data_explorer_router)
app.include_router(rt_detr_router.rt_detr_router)
app.include_router(sahi_router.sahi_router)
app.include_router(sam_router.sam_router)
app.include_router(yolo_v8_router.yolo_v8_router)
app.include_router(yolo_v9_router.yolo_v9_router)

if __name__ == '__main__':
    uvicorn_server.run(app, host=app_config.host, port=app_config.port)
