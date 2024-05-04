from typing import Union, Literal, List

import numpy as np
import pandas as pd
from ultralytics import Explorer


class DataExplorerService:

    def __init__(self, dataset_yaml_path: str, model_path: str):
        self.explorer = Explorer(data=dataset_yaml_path, model=model_path)

    def explore_by_similarity(self, image: Union[np.ndarray, List[np.ndarray]],
                              split: Literal["train", "test", "val"] = "train") -> pd.DataFrame:
        self.explorer.create_embeddings_table(force=True, split=split)
        return self.explorer.get_similar(img=image)

    def explore_by_query(self, query: str, split: Literal["train", "test", "val"] = "train") -> pd.DataFrame:
        self.explorer.create_embeddings_table(force=True, split=split)
        return self.explorer.sql_query(query=query)

    def explore_by_prompt(self, prompt: str, split: Literal["train", "test", "val"] = "train") -> pd.DataFrame:
        self.explorer.create_embeddings_table(force=True, split=split)
        return self.explorer.ask_ai(query=prompt)
