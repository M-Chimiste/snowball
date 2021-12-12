import json
import os
import glob
import pandas as pd
import numpy as np
import collections

from utils import cosine_similarity, l2_norm
DIST_FUNC = ["l2", "cosine"]
class Snowball:
    def __init__(self, data, distance_function="l2") -> None:
        self.data = self._parse_data(data)
        
        if distance_function not in DIST_FUNC:
            raise Exception(f"{distance_function} is not a valid distance function.  Choose one of {DIST_FUNC}.")
        self.distance_function = distance_function
        
        pass
    
    def _parse_data(self, data):
        if type(data) is list:
            if len(data) < 2:
                raise Exception("Dataset doesn't contain enough items.")
            if type(data[0]) is list or type(data[0]) is np.ndarray:
                # convert into a dictionary with keys being numerical ID and value being vector
                parsed_data = []
                for idx, value in enumerate(data):
                    data_dict = {"id": idx,
                                "vector": value}
                    parsed_data.append(data_dict)
                return parsed_data
            if type(data[0]) is dict:
                if not data[0].get("vector"):
                    raise Exception(f"Dictionary {data[0]} has no key vector.")
                else:
                    return data
        
