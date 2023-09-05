import numpy as np
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    load_entities,
    ensure_dict,
)
from qhana_plugin_runner.requests import open_url


def get_point(ent: dict) -> np.ndarray:
    dimension_keys = [k for k in ent.keys() if k not in ("ID", "href")]
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
        yield {"ID": ent["ID"], "href": ent.get("href", ""), "point": get_point(ent)}


def get_indices_and_point_arr(entity_points_url: str) -> (dict, np.array):
    entity_points = list(get_entity_generator(entity_points_url))
    id_list = []
    points_arr = []

    for ent in entity_points:
        if ent["ID"] in id_list:
            raise ValueError("Duplicate ID: ", ent["ID"])
        id_list.append(ent["ID"])
        points_arr.append(ent["point"])

    return id_list, np.array(points_arr)
