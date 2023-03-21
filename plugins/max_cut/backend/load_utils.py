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
    matrix_generator = get_entity_generator(matrix_url)
    matrix = []
    id_list_row = []
    id_list_col = None
    for entity in matrix_generator:
        id_list_row.append(entity["ID"])
        if id_list_col is None:
            id_list_col = [k for k in entity.keys() if k != "ID"]
        matrix.append([entity[id2] for id2 in id_list_col])
    return id_list_row, id_list_col, matrix
