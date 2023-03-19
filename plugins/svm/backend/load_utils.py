from typing import List, Optional
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
    entity_labels_url: str, id_list: list, label_to_int=None, int_to_label=None
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


def get_id_list(id_to_idx: dict):
    id_list = [None] * len(id_to_idx)
    for id, idx in id_to_idx.items():
        id_list[id] = idx

    return id_list


def load_kernel_matrix(
    kernel_url: str, id_to_idx_X: Optional[dict] = None
) -> (dict, dict, np.array):
    """
    Loads in a kernel matrix, given its url
    :param kernel_url: url to the kernel matrix
    """
    kernel_json = open_url(kernel_url).json()
    not_provided = id_to_idx_X is None
    if not_provided:
        id_to_idx_X = {}
    id_to_idx_Y = {}
    idx_X = 0
    idx_Y = 0
    for entry in kernel_json:
        if entry["entity_1_ID"] not in id_to_idx_Y:
            id_to_idx_Y[entry["entity_1_ID"]] = idx_Y
            idx_Y += 1
        if not_provided:
            if entry["entity_2_ID"] not in id_to_idx_X:
                id_to_idx_X[entry["entity_2_ID"]] = idx_X
                idx_X += 1
    kernel_matrix = np.zeros((len(id_to_idx_Y), len(id_to_idx_X)))

    if id_to_idx_Y.keys() == id_to_idx_X.keys():
        id_to_idx_Y = id_to_idx_X

    for entry in kernel_json:
        ent_id_Y = entry["entity_1_ID"]
        ent_id_X = entry["entity_2_ID"]
        kernel = entry["kernel"]

        ent_idx_Y = id_to_idx_Y[ent_id_Y]
        ent_idx_X = id_to_idx_X[ent_id_X]

        kernel_matrix[ent_idx_Y, ent_idx_X] = kernel

    return id_to_idx_X, id_to_idx_Y, kernel_matrix
