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

import numpy as np

TASK_LOGGER = get_task_logger(__name__)


# Creates a generator for loading in entities
def get_entity_generator(entity_points_url: str, stream=False):
    file_ = open_url(entity_points_url, stream=stream)
    file_.encoding = "utf-8"
    if entity_points_url[-3:] == "csv":
        return load_entities(file_, mimetype="text/csv")
    else:
        return load_entities(file_, mimetype="application/json")


# loads in the complete dataset as entity_points and idx_to_id
def load_entityPoints_and_idxToId(entity_points_url: str):
    entity_generator = get_entity_generator(entity_points_url)
    id_to_idx = {}
    idx = 0

    ent = None
    for ent in entity_generator:
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
    if pca_type == PCATypeEnum.normal.value:
        return PCA(
            n_components=input_params.get("dimensions"),
            svd_solver=input_params.get("solver"),
        )
    elif pca_type == PCATypeEnum.incremental.value:
        return IncrementalPCA(
            n_components=input_params.get("dimensions"),
            batch_size=input_params.get("batchSize"),
        )
    elif pca_type == PCATypeEnum.sparse.value:
        return SparsePCA(
            n_components=input_params.get("dimensions"),
            alpha=input_params.get("sparsityAlpha"),
            ridge_alpha=input_params.get("ridgeAlpha"),
            max_iter=input_params.get("maxItr"),
        )
    elif pca_type == PCATypeEnum.kernel.value:
        return KernelPCA(
            n_components=input_params.get("dimensions"), kernel=input_params.get("kernel")
        )
    raise ValueError(f"PCA with type {pca_type} not implemented!")


# Compute min and max for minmax scaling
# Note that the maxima, when returned, are already scaled by the minima
def get_minmax_scaler(entity_points_url: str):
    entity_generator = get_entity_generator(entity_points_url, stream=True)
    ent = next(entity_generator)
    scale_min = np.array(ent["point"])
    scale_max = np.array(ent["point"])
    for ent in entity_generator:
        scale_min = np.minimum(scale_min, ent["point"])  # Takes minimum for each dimension separately
        scale_max = np.maximum(scale_max, ent["point"])  # Takes maximum for each dimension separately
    return scale_min, (scale_max - scale_min)


# This method is used, when we can avoid that the whole output data is in memory
# It creates a generator, transforming each input element into the output element
def prepare_stream_output(entity_generator, pca, minmax_scale, scale_min, scale_max):
    for ent in entity_generator:
        point = ent["point"]
        if minmax_scale:
            point = (point - scale_min) / scale_max
        transformed_ent = pca.transform([point])[0]
        yield {"ID": ent["ID"], "href": "", "point": transformed_ent}


# This method is used, when the whole output data is in memory
def prepare_static_output(
    transformed_points, pca, id_to_idx, minmax_scale, scale_min, scale_max
):
    entity_points = []
    for ID, i in id_to_idx.items():
        entity_points.append({"ID": ID, "href": "", "point": list(transformed_points[i])})
    return entity_points


# This is used, when the pca needs the whole dataset in memory to be fitted
# In this method the input data is completely loaded into memory
def complete_fitting(entity_points_url, pca, minmax_scale, scale_min, scale_max):
    # load data from file
    (entity_points, id_to_idx) = load_entityPoints_and_idxToId(entity_points_url)
    if minmax_scale:
        entity_points = (entity_points - scale_min) / scale_max
    pca.fit(entity_points)
    transformed_points = pca.transform(entity_points)
    return prepare_static_output(
        transformed_points, pca, id_to_idx, minmax_scale, scale_min, scale_max
    )

# This is used, when the pca can be incrementally fitted via batches
# In this method the input data is loaded in batchwise
def batch_fitting(entity_points_url, pca, batch_size, minmax_scale, scale_min, scale_max):
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
            if minmax_scale:
                batch = (batch - scale_min) / scale_max
            pca.partial_fit(batch)
            idx = 0
        # append element
        batch[idx] = el["point"].copy()
        idx += 1
    if idx != 0:
        batch = batch[:idx]
        if minmax_scale:
            batch = (batch - scale_min) / scale_max
        pca.partial_fit(batch)
    return prepare_stream_output(
        get_entity_generator(entity_points_url, stream=True),
        pca,
        minmax_scale,
        scale_min,
        scale_max,
    )


def save_outputs(db_id, pca_output, entity_points):
    # save pca
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(pca_output, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "pca.json",
            "principle-components",
            "application/json",
        )

    # save transformed points
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(
            entity_points, output, "text/csv", attributes=["ID", "href", "point"]
        )
        STORE.persist_task_result(
            db_id,
            output,
            "transformed_entity_points.csv",
            "entity",
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

    # Compute scaling factors, if needed
    minmax_scale = input_params.get("minmaxScale")
    (scale_min, scale_max) = (None, None)
    if minmax_scale:
        (scale_min, scale_max) = get_minmax_scaler(entity_points_url)

    # Compute pca
    pca = get_correct_pca(input_params)

    # Since incremental pca uses batches, we want to load in the data in batches
    if input_params.get("pcaType") == PCATypeEnum.incremental.value:
        batch_size = input_params.get("batchSize")
        entity_points = batch_fitting(
            entity_points_url, pca, batch_size, minmax_scale, scale_min, scale_max
        )
    # Since the other PCAs need the complete dataset at once, we load it in at once
    else:
        entity_points = complete_fitting(
            entity_points_url, pca, minmax_scale, scale_min, scale_max
        )

    # Prepare output for pca file
    pca_output = {"components": pca.components_.tolist(), "mean": pca.mean_.tolist()}
    if minmax_scale:
        pca_output["scalingMin"] = scale_min.tolist()
        pca_output["scalingMax"] = scale_max.tolist()
    pca_output["ref-transormed"] = "transformed_entity_points"
    pca_output = [pca_output]

    save_outputs(db_id, pca_output, entity_points)

    return "Result stored in file"
