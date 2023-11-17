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

from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

from . import Optics
from sklearn.cluster import OPTICS

from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)
from qhana_plugin_runner.storage import STORE

import numpy as np

from .backend.load_utils import get_indices_and_point_arr
from .backend.visualize import plot_data
from qhana_plugin_runner.requests import open_url
import re


TASK_LOGGER = get_task_logger(__name__)


def retrieve_filename_from_url(url) -> str:
    """
    Given an url to a file, it returns the name of the file
    :param url: str
    :return: str
    """
    response = open_url(url)
    fname = ""
    if "Content-Disposition" in response.headers.keys():
        fname = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
        if fname[0] == fname[-1] and fname[0] in {'"', "'"}:
            fname = fname[1:-1]
    else:
        fname = url.split("/")[-1]
    response.close()

    # Remove file type endings
    fname = fname.split(".")
    fname = fname[:-1]
    fname = ".".join(fname)

    return fname


@CELERY.task(name=f"{Optics.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new optics calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entity_points_url = input_params.entity_points_url
    min_samples = (
        int(input_params.min_samples)
        if input_params.min_samples > 1
        else input_params.min_samples
    )
    max_epsilon = np.inf if input_params.max_epsilon < 0 else input_params.max_epsilon
    metric_enum = input_params.metric_enum
    minkowski_p = input_params.minkowski_p
    method_enum = input_params.method_enum
    epsilon = None if input_params.epsilon < 0 else input_params.epsilon
    xi = input_params.xi
    predecessor_correction = input_params.predecessor_correction
    min_cluster_size = (
        None if input_params.min_cluster_size < 0 else input_params.min_cluster_size
    )
    algorithm_enum = input_params.algorithm_enum
    leaf_size = input_params.leaf_size
    visualize = input_params.visualize

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    id_list, points = get_indices_and_point_arr(entity_points_url)
    points = np.array(points)

    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_epsilon,
        metric=metric_enum.get_metric(),
        p=minkowski_p,
        # metric_params=None,
        cluster_method=method_enum.get_method(),
        eps=epsilon,
        xi=xi,
        predecessor_correction=predecessor_correction,
        min_cluster_size=min_cluster_size,
        algorithm=algorithm_enum.get_algorithm(),
        leaf_size=leaf_size,
        # n_jobs=n_jobs,
    )
    predictions = optics.fit_predict(points)
    labels = [
        {"ID": _id, "href": "", "label": int(_label)}
        for _id, _label in zip(id_list, predictions)
    ]

    fig = None
    if visualize:
        fig = plot_data(
            points,
            id_list,
            predictions,
            only_first_100=True,
            title=f"OPTICS Clusters",
        )

    file_name = retrieve_filename_from_url(entity_points_url)
    info_str = f"_method_{method_enum.value}_algorithm_{algorithm_enum.get_algorithm()}_metric_{metric_enum.get_metric()}_from_{file_name}"

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"labels{info_str}.json",
            "entity/label",
            "application/json",
        )

    if fig is not None:
        with SpooledTemporaryFile(mode="wt") as output:
            html = fig.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                f"cluster_plot{info_str}.html",
                "plot",
                "text/html",
            )

    return "Result stored in file"
