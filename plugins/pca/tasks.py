# Copyright 2022 QHAna plugin runner contributors.
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
    """
    Return a generator for the entity points, given a url to them. This is useful, if not all points have to be loaded in
    at once, e.g. in IncrementalPCA.
    :param entity_points_url: url to the entity points
    """
    file_ = open_url(entity_points_url, stream=stream)
    file_.encoding = "utf-8"
    # if entity_points_url[-3:] == "csv":
    file_type = file_.headers["Content-Type"]
    if file_type == "text/csv":
        for ent in load_entities(file_, mimetype=file_type):
            ent = ent._asdict()
            point = get_points(ent)
            prepared_ent = {"ID": ent["ID"], "href": ent["href"], "point": point}
            yield prepared_ent
        # return load_entities(file_, mimetype="text/csv")
    elif file_type == "application/json":
        for ent in load_entities(file_, mimetype=file_type):
            yield ent


# loads in the complete dataset as entity_points and idx_to_id
def load_entity_points_and_idx_to_id(entity_points_url: str):
    """
    Loads in entity points, given their url.
    :param entity_points_url: url to the entity points
    """
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
    """
    Loads in a kernel matrix, given its url
    :param kernel_url: url to the kernel matrix
    """
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
    """
    Returns the correct pca model, given by the frontend's input paramters.
    :param input_params: ParameterHandler containing the frontend's input parameters
    """
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
            n_components=input_params.get("dimensions"),
            svd_solver=input_params.get("solver"),
            tol=input_params.get("tol"),
            iterated_power=input_params.get("iteratedPower"),
        )
    elif pca_type == PCATypeEnum.incremental.value:
        # For Debugging
        # TASK_LOGGER.info(f"\nIncremental PCA with parameters:"
        #                  f"\n\tn_components={input_params.get('dimensions')}"
        #                  f"\n\tbatch_size={input_params.get('batchSize')}")
        return IncrementalPCA(
            n_components=input_params.get("dimensions"),
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
            n_components=input_params.get("dimensions"),
            alpha=input_params.get("sparsityAlpha"),
            ridge_alpha=input_params.get("ridgeAlpha"),
            max_iter=input_params.get("maxItr"),
            tol=input_params.get("tol"),
        )
    elif pca_type == PCATypeEnum.kernel.value:
        eigen_solver = input_params.get("solver")
        if eigen_solver == "full":
            eigen_solver = "dense"
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
            n_components=input_params.get("dimensions"),
            kernel=input_params.get("kernel"),
            degree=input_params.get("degree"),
            gamma=input_params.get("kernelGamma"),
            coef0=input_params.get("kernelCoef"),
            eigen_solver=eigen_solver,
            tol=input_params.get("tol"),
            iterated_power=input_params.get('iteratedPower')  # This parameter is available in scikit-learn version ~= 1.0.2
        )
    raise ValueError(f"PCA with type {pca_type} not implemented!")


def get_entity_dict(ID, point):
    """
    Converts a point to the correct output format
    :param ID: The points ID
    :param point: The point
    :return: dict
    """
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
    """
    This method takes a set of points and prepares them for the final output. Each point gets converted to a dictionary
    with 'ID', 'href' and an entry for each dimension 'dim0', 'dim1', ...
    :param transformed_points: set of points
    :param id_to_idx: list converting a points id to the idx in the list transformed_points
    """
    entity_points = []
    for ID, i in id_to_idx.items():
        entity_points.append(get_entity_dict(ID, transformed_points[i]))
    return entity_points


# This is used, when the pca needs the whole dataset in memory to be fitted
# In this method the input data is completely loaded into memory
def complete_fitting(entity_points_url, pca):
    """
    This method loads in the entity points from the url. Then it fits the pca instance, transforms the entity points
    and returns a generator for the transformed entity points
    :param entity_points_url: url to the entity points
    :param pca: pca instance
    :return: list of the transformed data points
    """
    # load data from file
    (entity_points, id_to_idx) = load_entity_points_and_idx_to_id(entity_points_url)
    pca.fit(entity_points)
    transformed_points = pca.transform(entity_points)
    return prepare_static_output(transformed_points, id_to_idx)


def precomputed_kernel_fitting(kernel_url: str, pca):
    """
    This method takes a url to the kernel matrix and loads it. Afterwards it fits the pca model with the help of the
    kernel matrix and returns a generator of the already transformed data points.
    :param kernel_url: A url to the kernel matrix file
    :param pca: pca instance of sklearn
    :return: list of the transformed data points
    """
    # load data from file
    id_to_idx_X, id_to_idx_Y, kernel_matrix = load_kernel_matrix(kernel_url)
    # fit
    pca.fit(kernel_matrix)
    # transform
    transformed_points = pca.transform(kernel_matrix)
    # Here we only allow kernel matrices between the same points. K(X, X)
    return prepare_static_output(transformed_points, id_to_idx_X)


# This is used, when the pca can be incrementally fitted via batches
# In this method the input data is loaded in batchwise
def batch_fitting(entity_points_url, pca, batch_size):
    """
    This method loads in the entity points in batches of size batch_size from the entity_points_url and fits the
    pca with these batches. This is used for the IncrementalPCA class in sklearn.
    :param entity_points_url: url to the entity_points
    :param pca: a unfitted pca model
    :param batch_size: int how big the batchs for fitting should be.
    :return: None
    """
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


def plot_data(entity_points, dim, only_first_100):
    """
    This method creates a 1d, 2d or 3d plot of the given data. If the data is higher dimensional, then only the first three
    dimensions will be used to plot it.
    :param entity_points: List[dict]. Each point is a dictionary, with ID, href and its dimensions (e.g. dim0, dim1, ...).
    :param dim: int.
    :param only_first_100: If set to true, then only the first 100 points will be included in the plot.
    :return: plotly.express figure
    """
    import pandas as pd
    import plotly.express as px

    if dim >= 3:
        points_x = []
        points_y = []
        points_z = []
        ids = []
        if only_first_100:
            for _ in range(100):
                entity = next(entity_points)
                points_x.append(float(entity["dim0"]))
                points_y.append(float(entity["dim1"]))
                points_z.append(float(entity["dim2"]))
                ids.append(entity["ID"])
        else:
            for entity in entity_points:
                points_x.append(float(entity["dim0"]))
                points_y.append(float(entity["dim1"]))
                points_z.append(float(entity["dim2"]))
                ids.append(entity["ID"])
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "z": points_z,
                "ID": ids,
                "size": [10] * len(ids),
            }
        )
        return px.scatter_3d(df, x="x", y="y", z="z", hover_name="ID", size="size")
    elif dim == 2:
        points_x = []
        points_y = []
        ids = []
        if only_first_100:
            for _ in range(100):
                entity = next(entity_points)
                points_x.append(float(entity["dim0"]))
                points_y.append(float(entity["dim1"]))
                ids.append(entity["ID"])
        else:
            for entity in entity_points:
                points_x.append(float(entity["dim0"]))
                points_y.append(float(entity["dim1"]))
                ids.append(entity["ID"])
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "ID": ids,
                "size": [10] * len(ids),
            }
        )
        return px.scatter(df, x="x", y="y", hover_name="ID", size="size")
    else:
        points_x = []
        ids = []
        if only_first_100:
            for _ in range(100):
                entity = next(entity_points)
                points_x.append(float(entity["dim0"]))
                ids.append(entity["ID"])
        else:
            for entity in entity_points:
                points_x.append(float(entity["dim0"]))
                ids.append(entity["ID"])
        df = pd.DataFrame(
            {
                "x": points_x,
                "y": [0] * len(ids),
                "ID": ids,
                "size": [10] * len(ids),
            }
        )
        return px.scatter(df, x="x", y="y", hover_name="ID", size="size")


def save_outputs(
    db_id, pca_output, entity_points, attributes, entity_points_for_plot, dim
):
    """
    Saves the plugin's ouput into files. This includes a plot of the transformed data, the pca's parameters and
    the transformed points themselves.
    """
    TASK_LOGGER.info(
        f"entity_points_for_plot is not None = {entity_points_for_plot is not None}"
    )
    if entity_points_for_plot is not None:
        fig = plot_data(
            entity_points_for_plot,
            dim,
            only_first_100=(pca_output[0]["type"] == "incremental"),
        )

        with SpooledTemporaryFile(mode="wt") as output:
            html = fig.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                "plot.html",
                "plot",
                "text/html",
            )
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
        batch_fitting(entity_points_url, pca, batch_size)
        entity_points = prepare_stream_output(
            get_entity_generator(entity_points_url, stream=True), pca
        )
        entity_points_for_plot = prepare_stream_output(
            get_entity_generator(entity_points_url, stream=True), pca
        )
    elif (
        input_params.get("pcaType") == PCATypeEnum.kernel.value
        and input_params.get("kernel") == KernelEnum.precomputed.value
    ):
        entity_points = precomputed_kernel_fitting(kernel_url, pca)
        entity_points_for_plot = entity_points
    # Since the other PCAs need the complete dataset at once, we load it in at once
    else:
        entity_points = complete_fitting(entity_points_url, pca)
        entity_points_for_plot = entity_points

    # Prepare output for pca file
    # dim = num features of output. Get dim here, since input params can be <= 0
    pca_output, dim = pca_to_output(pca)
    pca_output = [pca_output]
    if dim > 3:
        entity_points_for_plot = None
    # for each dimension we have an attribute, i.e. dimension 0 = dim0, dimension 1 = dim1, ...
    attributes = ["ID", "href"] + [f"dim{d}" for d in range(dim)]
    save_outputs(
        db_id, pca_output, entity_points, attributes, entity_points_for_plot, dim
    )

    return "Result stored in file"
