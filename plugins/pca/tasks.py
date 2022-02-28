from tempfile import SpooledTemporaryFile

from typing import Optional

from json import loads
from celery.utils.log import get_task_logger

from plugins.pca import PCA
from plugins.pca.schemas import ParameterHandler, PCATypeEnum
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

from plugins.pca.pca_output import pca_to_output

import numpy as np

TASK_LOGGER = get_task_logger(__name__)


def get_points(ent):
    point = []
    d = 0
    while f"dim{d}" in ent.keys():
        point.append(ent[f"dim{d}"])
        d += 1
    return point


# Creates a generator for loading in entities
def get_entity_generator(entity_points_url: str, stream=False):
    file_ = open_url(entity_points_url, stream=stream)
    file_.encoding = "utf-8"
    if entity_points_url[-3:] == "csv":
        for ent in load_entities(file_, mimetype="text/csv"):
            ent = ent._asdict()
            point = get_points(ent)
            prepared_ent = {"ID": ent["ID"], "href": ent["href"], "point": point}
            yield prepared_ent
        # return load_entities(file_, mimetype="text/csv")
    else:
        for ent in load_entities(file_, mimetype="application/json"):
            yield ent


# loads in the complete dataset as entity_points and idx_to_id
def load_entityPoints_and_idxToId(entity_points_url: str):
    entity_generator = get_entity_generator(entity_points_url)
    id_to_idx = {}
    idx = 0

    ent = None
    for ent in entity_generator:
        ent = dict(ent)
        if ent["ID"] in id_to_idx:
            raise ValueError("Duplicate ID: ", ent["ID"])

        id_to_idx[ent["ID"]] = idx
        idx += 1

    points_cnt = len(id_to_idx)
    dimensions = len(ent["point"])
    points_arr = np.zeros((points_cnt, dimensions))

    entity_generator = get_entity_generator(entity_points_url)
    for ent in entity_generator:
        idx = id_to_idx[ent["ID"]]
        points_arr[idx] = ent["point"]

    return points_arr, id_to_idx


# This method returns a pca, depending on the input parameters input_params
def get_correct_pca(input_params: ParameterHandler):
    from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, KernelPCA

    pca_type = input_params.get("pcaType")

    # Result can't have dim <= 0. If this is entered, set to None.
    # If set to None all PCA types will compute as many components as possible
    # Exception for normal PCA we set n_components to 'mle', which automatically will choose the number of dimensions.
    n_components = input_params.get("dimensions")
    if n_components <= 0:
        n_components = None

    if pca_type == PCATypeEnum.normal.value:
        # Set to auto choose dimensions
        if n_components is None:
            n_components = 'mle'
        return PCA(
            n_components=n_components,
            svd_solver=input_params.get("solver"),
        )
    elif pca_type == PCATypeEnum.incremental.value:
        return IncrementalPCA(
            n_components=n_components,
            batch_size=input_params.get("batchSize"),
        )
    elif pca_type == PCATypeEnum.sparse.value:
        return SparsePCA(
            n_components=n_components,
            alpha=input_params.get("sparsityAlpha"),
            ridge_alpha=input_params.get("ridgeAlpha"),
        )
    elif pca_type == PCATypeEnum.kernel.value:
        return KernelPCA(
            n_components=n_components,
            kernel=input_params.get("kernel"),
            degree=input_params.get("degree"),
            gamma=input_params.get("kernelGamma"),
            coef0=input_params.get("kernelCoef")

        )
    raise ValueError(f"PCA with type {pca_type} not implemented!")


def get_entity_dict(ID, point):
    ent = {"ID": ID, "href": ""}
    for d in range(len(point)):
        ent[f"dim{d}"] = point[d]
    return ent


# This method is used, when we can avoid that the whole output data is in memory
# It creates a generator, transforming each input element into the output element
def prepare_stream_output(entity_generator, pca):
    for ent in entity_generator:
        point = ent["point"]
        transformed_ent = pca.transform([point])[0]
        yield get_entity_dict(ent["ID"], transformed_ent)


# This method is used, when the whole output data is in memory
def prepare_static_output(transformed_points, id_to_idx):
    entity_points = []
    for ID, i in id_to_idx.items():
        entity_points.append(get_entity_dict(ID, transformed_points[i]))
    return entity_points


# This is used, when the pca needs the whole dataset in memory to be fitted
# In this method the input data is completely loaded into memory
def complete_fitting(entity_points_url, pca):
    # load data from file
    (entity_points, id_to_idx) = load_entityPoints_and_idxToId(entity_points_url)
    pca.fit(entity_points)
    transformed_points = pca.transform(entity_points)
    return prepare_static_output(transformed_points, id_to_idx)


# This is used, when the pca can be incrementally fitted via batches
# In this method the input data is loaded in batchwise
def batch_fitting(entity_points_url, pca, batch_size):
    entity_generator = get_entity_generator(entity_points_url, stream=True)
    # get First element to get the correct array size
    el = next(entity_generator)["point"]
    batch = np.empty((batch_size, len(el)))
    batch[0] = el
    idx = 1
    for el in entity_generator:
        # check if we reached the batch_size
        if idx == batch_size:
            batch = np.array(batch)
            pca.partial_fit(batch)
            idx = 0
        # append element
        batch[idx] = el["point"].copy()
        idx += 1
    if idx != 0:
        batch = batch[:idx]
        pca.partial_fit(batch)
    return prepare_stream_output(
        get_entity_generator(entity_points_url, stream=True),
        pca,
    )


def save_outputs(db_id, pca_output, entity_points, attributes):
    # save pca
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(pca_output, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "pca_metadata.json",
            "pca-metadata",
            "application/json",
        )

    # save transformed points
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entity_points, output, "text/csv", attributes=attributes)
        STORE.persist_task_result(
            db_id,
            output,
            "transformed_entity_points.csv",
            "entity-points",
            "text/csv",
        )


@CELERY.task(name=f"{PCA.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters
    TASK_LOGGER.info(f"Starting new PCA calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: ParameterHandler = ParameterHandler(
        loads(task_data.parameters or "{}"), TASK_LOGGER
    )

    entity_points_url = input_params.get("entityPointsUrl")

    # Compute pca
    pca = get_correct_pca(input_params)

    # Since incremental pca uses batches, we want to load in the data in batches
    if input_params.get("pcaType") == PCATypeEnum.incremental.value:
        batch_size = input_params.get("batchSize")
        entity_points = batch_fitting(entity_points_url, pca, batch_size)
    # Since the other PCAs need the complete dataset at once, we load it in at once
    else:
        entity_points = complete_fitting(entity_points_url, pca)

    # Prepare output for pca file
    # dim = num features of output. Get dim here, since input params can be <= 0
    pca_output, dim = pca_to_output(pca)
    pca_output = [pca_output]

    TASK_LOGGER.info(f"\ndim = {dim}\n")

    # for each dimension we have an attribute, i.e. dimension 0 = dim0, dimension 1 = dim1, ...
    attributes = ["ID", "href"] + [f"dim{d}" for d in range(dim)]
    save_outputs(db_id, pca_output, entity_points, attributes)

    return "Result stored in file"
