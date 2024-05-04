from typing import List

import numpy as np
from sklearn.cluster import DBSCAN

from api.app.service.base.transcriptor_service import TranscriptorService
from api.app.utils.utils import transform_detections_to_xyxy_class_id_class_name_tuple


class TranscriptorDBSCANService(TranscriptorService):

    def separate_lines(self, number_of_lines: int):
        detections_tuples = transform_detections_to_xyxy_class_id_class_name_tuple(detections=self.detections)

        y_center = np.array([[(detection[1] + detection[3]) / 2] for detection in detections_tuples])

        y_center_clusters = DBSCAN(
            eps=(
                    max(detections_tuple[3] for detections_tuple in detections_tuples) -
                    min(detection[1] for detection in detections_tuples)
                ) / (number_of_lines * 2),
            min_samples=1
        ).fit(y_center)
        y_center_clusters_labels = y_center_clusters.labels_

        detections_lines_dict = {}
        for label, detection in zip(y_center_clusters_labels, detections_tuples):
            if label in detections_lines_dict:
                detections_lines_dict[label].append(detection)
            else:
                detections_lines_dict[label] = [detection]

        detections_lines = [
            detections_lines_dict[key]
            for key in sorted(detections_lines_dict, key=lambda k: min(detections_lines_dict[k], key=lambda x: x[1])[1])
        ]
        self.separated_detections_tuples = [
            sorted(detections_line, key=lambda detection: (detection[0] + detection[2]) / 2)
            for detections_line in detections_lines
        ]

    def transcript_class_id(self) -> str:
        super().transcript_class_id()
        return "\n".join(["".join([str(box[4]) + " " for box in line]) for line in self.separated_detections_tuples])

    def transcript_class_name(self) -> str:
        super().transcript_class_name()
        return "\n".join(["".join([box[5] + " " for box in line]) for line in self.separated_detections_tuples])

    def transcript_lines_class_id(self) -> List:
        super().transcript_lines_class_id()
        return ["".join([str(box[4]) + " " for box in line]) for line in self.separated_detections_tuples]

    def transcript_lines_class_name(self) -> List:
        super().transcript_lines_class_name()
        return ["".join([box[5] + " " for box in line]) for line in self.separated_detections_tuples]
