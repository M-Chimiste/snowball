import json
import os
import glob
import pandas as pd
import numpy as np
import uuid
from tqdm import tqdm
from collections import defaultdict

from utils import cosine_similarity, l2_norm
DIST_FUNC = ["l2", "cosine"]
class Snowball:
    def __init__(self, data, distance_function="l2", extension=None, show_progress=True, cluster_size=3) -> None:
        self.data = self._parse_data(data)
        self._extension = extension
        self._progress = not show_progress
        self._data_type = None
        if distance_function not in DIST_FUNC:
            raise Exception(f"{distance_function} is not a valid distance function.  Choose one of {DIST_FUNC}.")
        self.distance_function = distance_function
        self.cluster_size = cluster_size
        
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
    

    def _check_distance(self, vector_1, vector_2, distance_function, threshold):
        """Method to check two vectors against a specific distance function and threshold and return a boolean
        if they meet the threshold criteria.

        Args:
            vector_1 (np.array): Numpy array of first vector.
            vector_2 (np.array): Numpy array of second vector.
            distance_function (str): The name of the distance function to be used.
            threshold (float): Distance threshold value to check against.

        Raises:
            Exception: Checks for a correct distance function.

        Returns:
            [bool]: Boolean value to determine if the two vectors are similar enough.
        """
        meets_threshold = False
        if distance_function is "l2":
            distance = abs(l2_norm(vector_1, vector_2))
                
        if distance_function is "cosine":
            distance = abs(cosine_similarity(vector_1, vector_2))
        try:
            if distance >= threshold:
                meets_threshold = True
        except:
            raise Exception("Unable to select correct distance function.")
        return meets_threshold


    def cluster(self, threshold, distance_function=None, cluster_size=None):
        """Function will take a threshold value, distance function, and cluster size to conduct
        an agglomerative clustering operation and return a dictionary of clusters.

        Args:
            threshold ([type]): [description]
            distance_function ([type], optional): [description]. Defaults to None.
            cluster_size ([type], optional): [description]. Defaults to None.

        Raises:
            Exception: Invalid distance function if not part of DIST_FUNC
            Exception: Invalid data issue.
        Returns:
            [dict]: Dictionary value with keys for clusters and associated records as items/values.
        """
        if not distance_function:
            distance_function = self.distance_function
        if distance_function not in DIST_FUNC:
            raise Exception(f"Invalid distance function. Must be one of {DIST_FUNC}")
        if cluster_size is None:
            cluster_size = self.cluster_size
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
        
        cluster_data = defaultdict(list)
        cluster_num = 0

        for item in tqdm(data, disable=self._progress):
            count = 1
            like_items = []
            like_items.append(item)
            index = len(data) - 1
            pop_index = []
            for i in (range(index)):
                i = i + 1
                try:                 
                    child = data[i]
                    parent_vector = item["vector"]
                    child_vector = child["vector"]
                    similarity_score = self._check_distance(parent_vector, child_vector, distance_function, threshold)
                    if similarity_score:
                        like_items.append(child)
                        pop_index.append(i)
                        count += 1
                    else:
                        count += 1
                except:
                    break
            # Trending Work Orders
            if len(like_items) >= cluster_size:
                cluster_data[cluster_num] = like_items
                count = 0
                cluster_num += 1
                for z in pop_index:
                    try:
                        if count > 0:
                            z = z - count
                        data.pop(z)
                        count += 1
                    except:
                        break
            else:
                continue
        return cluster_data


