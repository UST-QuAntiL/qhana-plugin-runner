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

from . import ClassicalKMedoids

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

from .backend.load_utils import get_indices_and_point_arr
from .backend.visualize import plot_data

from sklearn_extra.cluster import KMedoids


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{ClassicalKMedoids.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new classical k medoids calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entity_points_url = input_params.entity_points_url
    num_clusters = input_params.num_clusters
    maxiter = input_params.maxiter
    init_enum = input_params.init_enum
    method_enum = input_params.method_enum
    visualize = input_params.visualize

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    id_list, points = get_indices_and_point_arr(entity_points_url)

    kmeans = KMedoids(
        n_clusters=num_clusters,
        method=method_enum.get_method(),
        init=init_enum.get_init(),
        random_state=0,
        max_iter=maxiter,
    )
    predictions = kmeans.fit_predict(points)
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
            title=f"Classical {num_clusters}-Medoids Clusters",
        )

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "labels.json",
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
                "cluster_plot.html",
                "plot",
                "text/html",
            )

    return "Result stored in file"