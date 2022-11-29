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
from .schemas import InputParameters, InputParametersSchema, PCATypeEnum, KernelEnum
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
    ensure_dict,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

from .pca_output import pca_to_output, get_output_dimensionality

import numpy as np
from itertools import islice


TASK_LOGGER = get_task_logger(__name__)


def get_point(ent):
    dimension_keys = list(ent.keys())
    dimension_keys.remove("ID")
    dimension_keys.remove("href")

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
    return entities_generator


def load_entity_points_and_idx_to_id(entity_points_url: str):
    """
    Loads in entity points, given their url.
    :param entity_points_url: url to the entity points
    """
    entity_generator = get_entity_generator(entity_points_url)
    id_to_idx = {}
    idx = 0

    # Check for duplicates and set index for elements
    ent = None
    for ent in entity_generator:
        if ent["ID"] in id_to_idx:
            raise ValueError("Duplicate ID: ", ent["ID"])

        id_to_idx[ent["ID"]] = idx
        idx += 1

    # Set array with correct size
    points_cnt = len(id_to_idx)
    dimensions = len(ent.keys()-{"ID", "href"})
    points_arr = np.empty((points_cnt, dimensions))

    # Go through elements again and insert them at the correct index
    entity_generator = get_entity_generator(entity_points_url)
    for ent in entity_generator:
        idx = id_to_idx[ent["ID"]]
        points_arr[idx] = get_point(ent)

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


def get_pca(input_params: dict):
    """
    Returns the correct pca model, given by the frontend's input paramters.
    :param input_params: ParameterHandler containing the frontend's input parameters
    """
    from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, KernelPCA


    pca_type = input_params["pca_type"]

    # Result can't have dim <= 0. If this is entered, set to None.
    # If set to None all PCA types will compute as many components as possible
    # Exception for normal PCA we set n_components to 'mle', which automatically will choose the number of dimensions.

    if pca_type == PCATypeEnum.normal:
        return PCA(
            n_components=input_params["dimensions"],
            svd_solver=input_params["solver"].value,
            tol=input_params["tol"],
            iterated_power=input_params["iterated_power"],
        )
    elif pca_type == PCATypeEnum.incremental:
        return IncrementalPCA(
            n_components=input_params["dimensions"],
            batch_size=input_params["batch_size"],
        )
    elif pca_type == PCATypeEnum.sparse:
        return SparsePCA(
            n_components=input_params["dimensions"],
            alpha=input_params["sparsity_alpha"],
            ridge_alpha=input_params["ridge_alpha"],
            max_iter=input_params["max_itr"],
            tol=input_params["tol"],
        )
    elif pca_type == PCATypeEnum.kernel:
        eigen_solver = input_params["solver"].value
        if eigen_solver == "full":
            eigen_solver = "dense"

        return KernelPCA(
            n_components=input_params["dimensions"],
            kernel=input_params["kernel"].value,
            degree=input_params["degree"],
            gamma=input_params["kernel_gamma"],
            coef0=input_params["kernel_coef"],
            eigen_solver=eigen_solver,
            tol=input_params["tol"],
            iterated_power=input_params['iterated_power']
        )
    raise ValueError(f"PCA with type {pca_type} not implemented!")


def get_entity_dict(ID, point, dim_attributes):
    """
    Converts a point to the correct output format
    :param ID: The points ID
    :param point: The point
    :param dim_attributes: List of dimension attributes
    :return: dict
    """
    ent = {"ID": ID, "href": ""}
    for attr_key, value in zip(dim_attributes, point):
        ent[attr_key] = value
    return ent


def get_dim_attributes(dim):
    """
    Returns the attributes for each dimension, with the correct length.
    For each dimension we have an attribute, i.e. dimension 0 = dim0, dimension 1 = dim1, ...
    This method adds a zero padding to ensure that every dim<int> has the same length, e.g. dim00, dim01, ..., dim10, dim11
    :params dim: int number of dimensions
    :return: list[str]
    """
    zero_padding = len(str(dim - 1))
    dim_attributes = [f"dim{d:0{zero_padding}}" for d in range(dim)]
    return dim_attributes


def prepare_stream_output(entity_generator, pca, dim_attributes):
    """
    This method is a generator, preparing each entity point for the final output. This method is used, when using the
    incremental pca. Since the advantage of the incremental pca is that not every point has to be in memory at once,
    this method loads in each point one by one and therefore keeps the advantage of the incremental pca.
    :param entity_generator: generator of points
    :param pca: a fitted pca
    :param dim_attributes: List of dimension attributes
    """
    for ent in entity_generator:
        point = ent["point"]
        transformed_ent = pca.transform([point])[0]
        yield get_entity_dict(ent["ID"], transformed_ent, dim_attributes)


def prepare_static_output(transformed_points, id_to_idx, dim_attributes):
    """
    This method takes a set of points and prepares them for the final output. Each point gets converted to a dictionary
    with 'ID', 'href' and an entry for each dimension 'dim0', 'dim1', ...
    :param transformed_points: set of points
    :param id_to_idx: list converting a points id to the idx in the list transformed_points
    :param dim_attributes: List of dimension attributes
    """
    entity_points = []
    for ID, i in id_to_idx.items():
        entity_points.append(get_entity_dict(ID, transformed_points[i], dim_attributes))
    return entity_points


def complete_fitting(entity_points_url, pca):
    """
    This method loads in the entity points from the url. Then it fits the pca instance, transforms the entity points
    and returns a generator for the transformed entity points. Additionally, it returns a list of attributes for the
    output's dimensionality.
    :param entity_points_url: url to the entity points
    :param pca: pca instance
    :return: list of the transformed data points, list of dimension attributes, int of number of dimensions
    """
    # load data from file
    (entity_points, id_to_idx) = load_entity_points_and_idx_to_id(entity_points_url)
    pca.fit(entity_points)
    transformed_points = pca.transform(entity_points)

    dim = get_output_dimensionality(pca)
    dim_attributes = get_dim_attributes(dim)

    return prepare_static_output(transformed_points, id_to_idx, dim_attributes), dim_attributes


def precomputed_kernel_fitting(kernel_url: str, pca):
    """
    This method takes an url to the kernel matrix and loads it. Afterwards it fits the pca model with the help of the
    kernel matrix and returns a generator of the already transformed data points. Additionally, it returns a list of
    attributes for the output's dimensionality.
    :param kernel_url: A url to the kernel matrix file
    :param pca: pca instance of sklearn
    :return: list of the transformed data points, list of dimension attributes, int of number of dimensions
    """
    # load data from file
    id_to_idx_X, id_to_idx_Y, kernel_matrix = load_kernel_matrix(kernel_url)
    # fit
    pca.fit(kernel_matrix)
    # transform
    transformed_points = pca.transform(kernel_matrix)
    dim = get_output_dimensionality(pca)
    dim_attributes = get_dim_attributes(dim)
    # Here we only allow kernel matrices between the same points. K(X, X)
    return prepare_static_output(transformed_points, id_to_idx_X, dim_attributes), dim_attributes


def batch_fitting(entity_points_url, pca, batch_size):
    """
    This method loads in the entity points in batches of size batch_size from the entity_points_url and fits the
    pca with these batches. This is used for the IncrementalPCA class in sklearn. The method returns a list of
    attributes for the output's dimensionality.
    :param entity_points_url: url to the entity_points
    :param pca: an unfitted pca model
    :param batch_size: int how big the batchs for fitting should be.
    :return: list of dimension attributes
    """
    entity_generator = get_entity_generator(entity_points_url)
    # get First element to get the correct array size
    el = get_point(next(entity_generator)["point"])
    batch = np.empty((batch_size, len(el)))
    prev_batch = np.empty((batch_size, len(el)))
    prev_batch[0] = el
    idx = 1
    # Init prev_batch
    for el in entity_generator:
        # append element
        prev_batch[idx] = get_point(el["point"])
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
        batch[idx] = get_point(el["point"])
        idx += 1
    if idx != 0:
        if idx < pca.n_components:
            batch = np.concatenate((prev_batch, batch[:idx]))
        else:
            pca.partial_fit(prev_batch)
            batch = batch[:idx]
        pca.partial_fit(batch)

    dim = get_output_dimensionality(pca)
    dim_attributes = get_dim_attributes(dim)

    return dim_attributes


def plot_data(entity_points, dim_attributes, only_first_100):
    """
    This method creates a 1d, 2d or 3d plot of the given data. If the data is higher dimensional, then only the first three
    dimensions will be used to plot it.
    :param entity_points: List[dict]. Each point is a dictionary, with ID, href and its dimensions (e.g. dim0, dim1, ...).
    :param dim_attributes: List of dimension attributes
    :param only_first_100: If set to true, then only the first 100 points will be included in the plot.
    :return: plotly.express figure
    """
    import pandas as pd
    import plotly.express as px

    # if only_first 100 == True, then this ensures that only the first 100 entities are returned by entity_points
    # else it returns all the entity_points
    entity_points = islice(entity_points, 100 if only_first_100 else None)

    points = []
    ids = []
    for entity in entity_points:
        point = [float(entity[d]) for d in dim_attributes]
        points.append(point)
        ids.append(entity["ID"])

    points = np.array(points)

    if len(dim_attributes) >= 3:
        df = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "ID": ids,
                "size": [10] * len(ids),
            }
        )
        return px.scatter_3d(df, x="x", y="y", z="z", hover_name="ID", size="size")
    else:
        if len(dim_attributes) == 1:
            points_y = [0]*len(ids)
        else:
            points_y = points[:, 1]

        df = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points_y,
                "ID": ids,
                "size": [10] * len(ids),
            }
        )
        return px.scatter(df, x="x", y="y", hover_name="ID", size="size")


def save_outputs(
    db_id, pca_output, entity_points, attributes, entity_points_for_plot, dim_attributes
):
    """
    Saves the plugin's output into files. This includes a plot of the transformed data, the pca's parameters and
    the transformed points themselves.
    """
    TASK_LOGGER.info(
        f"entity_points_for_plot is not None = {entity_points_for_plot is not None}"
    )
    if entity_points_for_plot is not None:
        fig = plot_data(
            entity_points_for_plot,
            dim_attributes,
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


def prep_input_parameters(input_params: InputParameters) -> dict:
    input_params = input_params.__dict__
    # Set parameters to correct conditions
    # batch size needs to be at least the size of the dimensions
    input_params["batch_size"] = max(
        input_params["batch_size"], input_params["dimensions"]
    )
    # If dimensions <= 0, then dimensions should be chosen automatically
    if input_params["dimensions"] <= 0:
        input_params["dimensions"] = None
        if input_params["pca_type"] == PCATypeEnum.normal:
            input_params["dimensions"] = "mle"
    # If tolerance tol is set to <= 0, then we set it as follows
    if input_params["tol"] <= 0:
        # 1e-8 for sparse PCA
        if input_params["pca_type"] == PCATypeEnum.sparse:
            input_params["tol"] = 1e-8
        # 0 for normal and kernel PCA
        else:
            input_params["tol"] = 0
        # Incremental PCA does not use this parameter

    # If iterated power is set to <= 0, then it should be chosen automatically
    if input_params["iterated_power"] <= 0:
        input_params["iterated_ower"] = "auto"

    return input_params


@CELERY.task(name=f"{PCA.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters
    TASK_LOGGER.info(f"Starting new PCA calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)
    TASK_LOGGER.info(
        f"Loaded input parameters from db: {str(input_params)}"
    )
    input_params = prep_input_parameters(input_params)

    entity_points_url = input_params["entity_points_url"]
    kernel_url = input_params["kernel_url"]

    # Compute pca
    pca = get_pca(input_params)

    # Since incremental pca uses batches, we want to load in the data in batches
    if input_params["pca_type"] == PCATypeEnum.incremental.value:
        batch_size = input_params["batch_size"]
        dim_attributes = batch_fitting(entity_points_url, pca, batch_size)
        entity_points = prepare_stream_output(
            get_entity_generator(entity_points_url), pca, dim_attributes
        )
        entity_points_for_plot = prepare_stream_output(
            get_entity_generator(entity_points_url), pca, dim_attributes
        )
    elif (
        input_params["pca_type"] == PCATypeEnum.kernel.value
        and input_params["kernel"] == KernelEnum.precomputed.value
    ):
        entity_points, dim_attributes = precomputed_kernel_fitting(kernel_url, pca)
        entity_points_for_plot = entity_points
    # Since the other PCAs need the complete dataset at once, we load it in at once
    else:
        entity_points, dim_attributes = complete_fitting(entity_points_url, pca)
        entity_points_for_plot = entity_points

    # Prepare output for pca file
    # dim = num features of output. Get dim here, since input params can be <= 0
    pca_output = pca_to_output(pca)
    pca_output = [pca_output]

    attributes = ["ID", "href"] + dim_attributes
    save_outputs(
        db_id, pca_output, entity_points, attributes, entity_points_for_plot, dim_attributes
    )

    return "Result stored in file"
