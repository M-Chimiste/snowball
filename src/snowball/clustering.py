import json
import os
import glob
import pandas as pd
import numpy as np
import uuid
from tqdm import tqdm

from utils import cosine_similarity, l2_norm
DIST_FUNC = ["l2", "cosine"]
class Snowball:
    def __init__(self, data, distance_function="l2", extension=None, show_progress=True) -> None:
        self.data = self._parse_data(data)
        self._extension = extension
        self._progress = not show_progress
        self._data_type = None
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
                self._data_type = 'vect'
                return parsed_data
            
            if type(data[0]) is dict:
                if not data[0].get("vector"):
                    raise Exception(f"Dictionary {data[0]} has no key vector.")
                else:
                    self._data_type = "dict"
                    return data
            
            if type(data) is str:
                if not self._extension:
                    print("No file extension was found.  Will assume .json.")
                    self._extension = ".json"
                files = glob.glob(f"{data}*{self.extension}")
                if len(files) < 2:
                    raise Exception(f"Not enough files found.  Only found {len(files)}.")
                data = []
                for filename in tqdm(files, disable=self._progress):
                    file_content = self._read_json(filename)
                    if not file_content.get("vector"):
                        print(f"{filename} does not contain a vector key, skipping")
                        continue
                    
                    data.append(file_content)
                if len(data) < 2:
                    raise Exception(f"Not enough files found.  Only found {len(files)}.")
                self._data_type = "dict"
                return data


    def _read_json(self, filename):
        """Method will take a filename, read it into memory, and return it as a dictionary.

        Args:
            filename (str): Filename for the json file to be loaded.

        Returns:
            [data]: Dictionary representation of the file.
        """
        with open(filename, "r") as fp:
            data = json.load(fp)
        return data            
    

    def cluster(self):
        data = self.data
        temp_dicts = []
        all_uuids = []
        all_vectors = []
        if self._data_type is "dict":
            for i in tqdm(data, disable=self._progress):
                new_uuid = str(uuid.uuid4())
                i["snowball_uuid"] = new_uuid
                all_uuids.append(new_uuid)
                all_vectors.append(i["vector"])
        if self._data_type is "vect":
            for i in tqdm(data, disable=self._progress):
                new_uuid = str(uuid.uuid4())
                all_uuids.append(new_uuid)
                temp_dict = {"snowball_uuid": new_uuid,
                             "vector": i}
                temp_dicts.append(temp_dict)
            self.data = temp_dicts
        if self._data_type is None:
            raise Exception(f"An error has occurred, check your data and try again.  Data Type is {self._data_type}")
        
        # TODO Add Aggregate Clustering Logic.

                
        