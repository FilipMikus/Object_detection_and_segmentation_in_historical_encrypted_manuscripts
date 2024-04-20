import json


class AppConfig:
    _instance = None

    name: str = "VEGA Historical Object Detection and Segmentation API"
    version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    yolo_v8_digits_source: str = "../resources/weights/yolo_v8_digits.pt"
    yolo_v8_glyphs_source: str = "../resources/weights/yolo_v8_glyphs.pt"
    yolo_v9_digits_source: str = "../resources/weights/yolo_v8_digits.pt"
    yolo_v9_glyphs_source: str = "../resources/weights/yolo_v8_glyphs.pt"
    yolo_world_v2_digits_source: str = "../resources/weights/yolo_world_v2_digits.pt"
    yolo_world_v2_glyphs_source: str = "../resources/weights/yolo_world_v2_glyphs.pt"
    yolo_v8_annotator_digits_source: str = "../resources/weights/yolo_v8_digits.pt"
    yolo_v8_annotator_glyphs_source: str = "../resources/weights/yolo_v8_glyphs.pt"
    rt_detr_digits_source: str = "../resources/weights/rt_detr_digits.pt"
    rt_detr_glyphs_source: str = "../resources/weights/rt_detr_glyphs.pt"
    yolo_v8_explorer_source: str = "../resources/weights/yolov8n.pt"
    fast_sam_source: str = "../resources/weights/FastSAM-s.pt"
    sam_annotator_source: str = "../resources/weights/mobile_sam.pt"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            try:
                with open("../config/config.json") as json_config_file:
                    config_data = json.load(json_config_file)
                    cls.name = config_data["name"]
                    cls.version = config_data["version"]
                    cls.host = config_data["host"]
                    cls.port = config_data["port"]
                    cls.yolo_v8_digits_source = config_data["yolo_v8_digits_source"]
                    cls.yolo_v8_glyphs_source = config_data["yolo_v8_glyphs_source"]
                    cls.yolo_v9_digits_source = config_data["yolo_v9_digits_source"]
                    cls.yolo_v9_glyphs_source = config_data["yolo_v9_glyphs_source"]
                    cls.yolo_world_v2_digits_source = config_data["yolo_world_v2_digits_source"]
                    cls.yolo_world_v2_glyphs_source = config_data["yolo_world_v2_glyphs_source"]
                    cls.yolo_v8_annotator_digits_source = config_data["yolo_v8_annotator_digits_source"]
                    cls.yolo_v8_annotator_glyphs_source = config_data["yolo_v8_annotator_glyphs_source"]
                    cls.rt_detr_digits_source = config_data["rt_detr_digits_source"]
                    cls.rt_detr_glyphs_source = config_data["rt_detr_glyphs_source"]
                    cls.yolo_v8_explorer_source = config_data["yolo_v8_explorer_source"]
                    cls.fast_sam_source = config_data["fast_sam_source"]
                    cls.sam_annotator_source = config_data["sam_annotator_source"]
            except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
                print(e)
                pass

        return cls._instance
