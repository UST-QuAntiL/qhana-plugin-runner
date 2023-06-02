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

from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    load_entities,
    ensure_dict,
)
from qhana_plugin_runner.requests import open_url

from typing import List


def get_entity_generator(entity_url: str):
    """
    Return a generator for the entity points, given an url to them.
    :param entity_points_url: url to the entity points
    """
    file_ = open_url(entity_url)
    file_.encoding = "utf-8"
    file_type = file_.headers["Content-Type"]
    entities_generator = load_entities(file_, mimetype=file_type)
    entities_generator = ensure_dict(entities_generator)
    return entities_generator


def load_matrix_url(matrix_url: str) -> (List, List, List[List[float]]):
    """
    Returns a 2d List (matrix) from the given url.
    :param matrix_url: url to the matrix
    """
    matrix_generator = get_entity_generator(matrix_url)
    matrix = []
    id_list_row = []
    id_list_col = None
    for entity in matrix_generator:
        id_list_row.append(entity["ID"])
        if id_list_col is None:
            id_list_col = [k for k in entity.keys() if k not in ("ID", "href")]
        matrix.append([entity[id2] for id2 in id_list_col])
    return id_list_row, id_list_col, matrix
