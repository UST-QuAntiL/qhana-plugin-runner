from tempfile import SpooledTemporaryFile

from typing import Optional, List

from json import loads
from celery.utils.log import get_task_logger

from . import PCA
from .schemas import ParameterHandler, PCATypeEnum, KernelEnum
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

from .pca_output import pca_to_output

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
    # if entity_points_url[-3:] == "csv":
    file_type = file_.headers['Content-Type']
    if file_type == 'text/csv':
        for ent in load_entities(file_, mimetype=file_type):
            ent = ent._asdict()
            point = get_points(ent)
            prepared_ent = {"ID": ent["ID"], "href": ent["href"], "point": point}
            yield prepared_ent
        # return load_entities(file_, mimetype="text/csv")
    elif file_type == 'application/json':
        for ent in load_entities(file_, mimetype=file_type):
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


def load_kernel_matrix(kernel_url: str) -> (dict, dict, List[List[float]]):
    kernel_json = open_url(kernel_url).json()
    id_to_idx_X = {}
    id_to_idx_Y = {}
    idx_X = 0
    idx_Y = 0
    for entry in kernel_json:
        if entry["entity_1_ID"] not in id_to_idx_Y:
            id_to_idx_Y[entry["entity_1_ID"]] = idx_Y
            idx_Y += 1
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


# This method returns a pca, depending on the input parameters input_params
def get_correct_pca(input_params: ParameterHandler):
    from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, KernelPCA

    pca_type = input_params.get("pcaType")

    # Result can't have dim <= 0. If this is entered, set to None.
    # If set to None all PCA types will compute as many components as possible
    # Exception for normal PCA we set n_components to 'mle', which automatically will choose the number of dimensions.

    if pca_type == PCATypeEnum.normal.value:
        # For Debugging
        # TASK_LOGGER.info(f"\nNormal PCA with parameters:"
        #                  f"\n\tn_components={input_params.get('dimensions')}"
        #                  f"\n\tsvd_solver={input_params.get('solver')}"
        #                  f"\n\ttol={input_params.get('tol')}"
        #                  f"\n\titerated_power={input_params.get('iteratedPower')}")
        return PCA(
            n_components=input_params.get('dimensions'),
            svd_solver=input_params.get("solver"),
            tol=input_params.get('tol'),
            iterated_power=input_params.get('iteratedPower')
        )
    elif pca_type == PCATypeEnum.incremental.value:
        # For Debugging
        # TASK_LOGGER.info(f"\nIncremental PCA with parameters:"
        #                  f"\n\tn_components={input_params.get('dimensions')}"
        #                  f"\n\tbatch_size={input_params.get('batchSize')}")
        return IncrementalPCA(
            n_components=input_params.get('dimensions'),
            batch_size=input_params.get("batchSize"),
        )
    elif pca_type == PCATypeEnum.sparse.value:
        # For Debugging
        # TASK_LOGGER.info(f"\nSparse PCA with parameters:"
        #                  f"\n\tn_components={input_params.get('dimensions')}"
        #                  f"\n\talpha={input_params.get('sparsityAlpha')} (Sparsity Alpha)"
        #                  f"\n\tridge_alpha={input_params.get('ridgeAlpha')}"
        #                  f"\n\tmax_iter={input_params.get('maxItr')}"
        #                  f"\n\ttol={input_params.get('tol')}")
        return SparsePCA(
            n_components=input_params.get('dimensions'),
            alpha=input_params.get("sparsityAlpha"),
            ridge_alpha=input_params.get("ridgeAlpha"),
            max_iter=input_params.get('maxItr'),
            tol=input_params.get('tol')
        )
    elif pca_type == PCATypeEnum.kernel.value:
        eigen_solver = input_params.get("solver")
        if eigen_solver == 'full':
            eigen_solver = 'dense'
        # For Debugging
        # TASK_LOGGER.info(f"\nKernel PCA with parameters:"
        #                 f"\n\tn_components={input_params.get('dimensions')}"
        #                 f"\n\tkernel={input_params.get('kernel')}"
        #                 f"\n\tdegree={input_params.get('degree')}"
        #                 f"\n\tgamma={input_params.get('kernelGamma')}"
        #                 f"\n\tcoef0={input_params.get('kernelCoef')}"
        #                 f"\n\teigen_solver={eigen_solver}"
        #                 f"\n\ttol={input_params.get('tol')}")
        return KernelPCA(
            n_components=input_params.get('dimensions'),
            kernel=input_params.get("kernel"),
            degree=input_params.get("degree"),
            gamma=input_params.get("kernelGamma"),
            coef0=input_params.get("kernelCoef"),
            eigen_solver=eigen_solver,
            tol=input_params.get('tol'),
            # iterated_power=input_params.get('iteratedPower')  # This parameter is available in scikit-learn version ~= 1.0.2
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


def precomputed_kernel_fitting(kernel_url: str, pca):
    # load data from file
    id_to_idx_X, id_to_idx_Y, kernel_matrix = load_kernel_matrix(kernel_url)
    pca.fit(kernel_matrix)
    transformed_points = pca.transform(kernel_matrix)
    # Here we only allow kernel matrices between the same points. K(X, X)
    return prepare_static_output(transformed_points, id_to_idx_X)

# This is used, when the pca can be incrementally fitted via batches
# In this method the input data is loaded in batchwise
def batch_fitting(entity_points_url, pca, batch_size):
    entity_generator = get_entity_generator(entity_points_url, stream=True)
    # get First element to get the correct array size
    el = next(entity_generator)["point"]
    batch = np.empty((batch_size, len(el)))
    prev_batch = np.empty((batch_size, len(el)))
    prev_batch[0] = el
    idx = 1
    # Init prev_batch
    for el in entity_generator:
        # append element
        prev_batch[idx] = el["point"].copy()
        idx += 1
        # check if we reached the batch_size
        if idx == batch_size:
            idx = 0
            break
    for el in entity_generator:
        # check if we reached the batch_size
        if idx == batch_size:
            batch = np.array(batch)
            pca.partial_fit(prev_batch)
            prev_batch = batch.copy()
            idx = 0
        # append element
        batch[idx] = el["point"].copy()
        idx += 1
    if idx != 0:
        if idx < pca.n_components:
            batch = np.concatenate((prev_batch, batch[:idx]))
        else:
            pca.partial_fit(prev_batch)
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
    kernel_url = input_params.get("kernelUrl")

    # Compute pca
    pca = get_correct_pca(input_params)

    # Since incremental pca uses batches, we want to load in the data in batches
    if input_params.get("pcaType") == PCATypeEnum.incremental.value:
        batch_size = input_params.get("batchSize")
        entity_points = batch_fitting(entity_points_url, pca, batch_size)
    elif input_params.get("pcaType") == PCATypeEnum.kernel.value and input_params.get("kernel") == KernelEnum.precomputed.value:
        entity_points = precomputed_kernel_fitting(kernel_url, pca)
    # Since the other PCAs need the complete dataset at once, we load it in at once
    else:
        entity_points = complete_fitting(entity_points_url, pca)

    # Prepare output for pca file
    # dim = num features of output. Get dim here, since input params can be <= 0
    pca_output, dim = pca_to_output(pca)
    pca_output = [pca_output]

    # for each dimension we have an attribute, i.e. dimension 0 = dim0, dimension 1 = dim1, ...
    attributes = ["ID", "href"] + [f"dim{d}" for d in range(dim)]
    save_outputs(db_id, pca_output, entity_points, attributes)

    return "Result stored in file"
