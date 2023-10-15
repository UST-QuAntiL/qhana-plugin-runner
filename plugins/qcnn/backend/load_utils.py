# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.plugin_utils.entity_marshalling import load_entities, ensure_dict

from typing import List
import numpy as np


def get_point(ent: dict) -> np.array:
    dimension_keys = [k for k in ent.keys() if k not in ("ID", "href", "shape")]
    dimension_keys.sort()
    point = np.empty(len(dimension_keys))
    for idx, d in enumerate(dimension_keys):
        point[idx] = ent[d]
    return point


def get_entity_generator(entity_points_url: str):
    """
    Return a generator for the entity points, given an url to them.
    :param entity_points_url: url to the entity points
    """
    file_ = open_url(entity_points_url)
    file_.encoding = "utf-8"
    file_type = file_.headers["Content-Type"]
    entities_generator = load_entities(file_, mimetype=file_type)
    entities_generator = ensure_dict(entities_generator)
    for ent in entities_generator:
        yield {
            "ID": ent["ID"],
            "href": ent.get("href", ""),
            "point": get_point(ent).reshape(ent["shape"]),
        }


def get_ids_and_data_arr(data_url: str) -> (List, np.array):
    entity_points = list(get_entity_generator(data_url))
    id_list = []
    points_arr = []

    for ent in entity_points:
        if ent["ID"] in id_list:
            raise ValueError("Duplicate ID: ", ent["ID"])
        id_list.append(ent["ID"])
        points_arr.append(ent["point"])

    return id_list, np.array(points_arr)


def get_label_generator(entity_labels_url: str):
    """
    Return a generator for the entity labels, given an url to them.
    :param entity_labels_url: url to the entity labels
    """
    file_ = open_url(entity_labels_url)
    file_.encoding = "utf-8"
    file_type = file_.headers["Content-Type"]
    entities_generator = load_entities(file_, mimetype=file_type)
    entities_generator = ensure_dict(entities_generator)
    for ent in entities_generator:
        yield {"ID": ent["ID"], "href": ent.get("href", ""), "label": ent["label"]}


def get_label_arr(
    entity_labels_url: str,
    id_list: list,
    label_to_int: dict = None,
    int_to_label: list = None,
) -> (dict, List[List[float]]):
    entity_labels = list(get_label_generator(entity_labels_url))

    # Initialise label array
    labels = np.zeros(len(id_list), dtype=int)

    if label_to_int is None:
        label_to_int = dict()
    if int_to_label is None:
        int_to_label = list()

    id_to_idx = {value: idx for idx, value in enumerate(id_list)}
    for ent in entity_labels:
        label = ent["label"]
        label_str = str(label)
        if label_str not in label_to_int:
            label_to_int[label_str] = len(int_to_label)
            int_to_label.append(label)
        labels[id_to_idx[ent["ID"]]] = label_to_int[label_str]

    return labels, label_to_int, int_to_label
