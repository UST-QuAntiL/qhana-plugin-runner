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

from typing import Iterator, Optional

import numpy as np
import requests
from celery.utils.log import get_task_logger

from plugins.optimizer.coordinator import Optimizer
from plugins.optimizer.shared.enums import InteractionEndpointType
from plugins.optimizer.shared.schemas import (
    ObjectiveFunctionPassData,
    ObjectiveFunctionPassDataResponse,
    ObjectiveFunctionPassDataResponseSchema,
    ObjectiveFunctionPassDataSchema,
)
from qhana_plugin_runner.api.tasks_api import TaskData, TaskStatusSchema
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)


def get_features(ent: dict, target_variable_name: str) -> np.ndarray:
    """
    Get the feature vector for a given entity.

    Args:
        ent: A dictionary representing an entity.
        target_variable_name: The name of the target variable.

    Returns:
        A numpy array representing the feature vector for the given entity.
    """
    dimension_keys = [
        k for k in ent.keys() if k not in ("ID", "href", target_variable_name)
    ]
    dimension_keys.sort()
    point = np.empty(len(dimension_keys))
    for idx, d in enumerate(dimension_keys):
        point[idx] = ent[d]
    return point


def get_entity_generator(
    entity_points_url: str, target_variable_name: str
) -> Iterator[dict]:
    """
    Return a generator for the entity points, given a URL to them.

    Args:
        entity_points_url: URL to the entity points
        target_variable_name: Name of the target variable

    Yields:
        A dictionary representing an entity.
    """
    with open_url(entity_points_url) as file_:
        file_.encoding = "utf-8"
        file_.headers["Content-Type"] = "text/csv"
        entities_generator = load_entities(file_, mimetype="text/csv")
        entities_generator = ensure_dict(entities_generator)
        for ent in entities_generator:
            yield {
                "ID": ent["ID"],
                "href": ent.get("href", ""),
                "features": get_features(ent, target_variable_name),
                "target": float(ent[target_variable_name]),
            }


def get_features_and_target(
    entity_points_url: str, target_variable_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return NumPy arrays for the entity points' features and target variable, given a URL to them.

    Args:
        entity_points_url: URL to the entity points
        target_variable_name: Name of the target variable

    Returns:
        A tuple containing NumPy arrays for the features and target variables
    """
    x = []
    y = []
    for ent in get_entity_generator(entity_points_url, target_variable_name):
        x.append(ent["features"])
        y.append(ent["target"])
    return np.array(x), np.array(y)


@CELERY.task(name=f"{Optimizer.instance.identifier}.echo_results", bind=True)
def echo_results(self, db_id: int) -> str:
    """
    Start a minimization task for the specified database id.

    Args:
        db_id: The database id of the task.

    Returns:
        A string representation of the weights obtained from the minimization process.

    Raises:
        A KeyError if task data with the specified id cannot be loaded to read parameters.
    """
    TASK_LOGGER.info(f"Starting the optimization task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    url = task_data.data.get("minimizer_result_url")
    minimizer_response = requests.get(url)
    response_data: TaskData = TaskStatusSchema().load(minimizer_response.json())
    for output in response_data.outputs:
        STORE.persist_task_result(
            task_db_id=db_id,
            file_=output["href"],
            file_type=output["data_type"],
            mimetype=output["content_type"],
            file_name=output["name"],
            storage_provider="url_file_store",
            commit=True,
        )

    return "Success"


@CELERY.task(name=f"{Optimizer.instance.identifier}.of_pass_data", bind=True)
def of_pass_data(self, db_id: int) -> str:
    """"""
    TASK_LOGGER.info(f"Starting passing data to of plugin with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    url = task_data.data.get(InteractionEndpointType.of_pass_data.value)
    input_file_url = task_data.data.get("input_file_url")
    target_variable = task_data.data.get("target_variable")
    hyperparameters = task_data.data.get("of_hyperparameters")

    X, y = get_features_and_target(input_file_url, target_variable)

    request_data = ObjectiveFunctionPassDataSchema().dump(
        ObjectiveFunctionPassData(
            x=X,
            y=y,
            hyperparameters=hyperparameters,
        )
    )

    response = requests.post(url, json=request_data)

    response_data: ObjectiveFunctionPassDataResponse = (
        ObjectiveFunctionPassDataResponseSchema().load(response.json())
    )
    task_data.data["number_weights"] = response_data.number_weights

    task_data.save(commit=True)

    return "Successfully passed data to of plugin"
