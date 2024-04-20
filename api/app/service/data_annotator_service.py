from ultralytics.data.annotator import auto_annotate


class DataAnnotatorService:

    def __init__(self, dataset_images_path: str, output_dataset_labels_path: str,
                 model_path: str, sam_model_path: str):
        self.dataset_images_path = dataset_images_path
        self.output_dataset_labels_path = output_dataset_labels_path
        self.model_path = model_path
        self.sam_model_path = sam_model_path

    def annotate(self) -> str:
        auto_annotate(
            data=self.dataset_images_path,
            det_model=self.model_path,
            sam_model=self.sam_model_path,
            output_dir=self.output_dataset_labels_path
        )
        return self.output_dataset_labels_path
