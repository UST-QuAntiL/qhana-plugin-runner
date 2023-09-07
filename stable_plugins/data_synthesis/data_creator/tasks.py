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

from . import DataCreator

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


TASK_LOGGER = get_task_logger(__name__)


def get_entity_dict(ID, point):
    """
    Converts a point to the correct output format
    :param ID: The points ID
    :param point: The point
    :return: dict
    """
    ent = {"ID": ID, "href": ""}
    for idx, value in enumerate(point):
        ent[f"dim{idx}"] = value
    return ent


@CELERY.task(name=f"{DataCreator.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new data creation calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    dataset_type = input_params.dataset_type

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    train_data, train_labels, test_data, test_labels = dataset_type.get_data(
        **input_params.__dict__
    )
    train_id = [str(i) for i in range(len(train_data))]
    test_id = [str(i) for i in range(len(test_data), len(train_data) + len(test_data))]
    train_data = [get_entity_dict(ID, point) for ID, point in zip(train_id, train_data)]
    train_labels = [
        {"ID": ID, "href": "", "label": label}
        for ID, label in zip(train_id, train_labels)
    ]
    test_data = [get_entity_dict(ID, point) for ID, point in zip(test_id, test_data)]
    test_labels = [
        {"ID": ID, "href": "", "label": label} for ID, label in zip(test_id, test_labels)
    ]

    info_str = f"_type_{dataset_type.name}"

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(train_data, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"train_data{info_str}_amount_{len(train_data)}.json",
            "entity/vector",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(train_labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"train_labels{info_str}_amount_{len(train_labels)}.json",
            "entity/label",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(test_data, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"test_data{info_str}_amount_{len(test_data)}.json",
            "entity/vector",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(test_labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"test_labels{info_str}_amount_{len(test_labels)}.json",
            "entity/label",
            "application/json",
        )

    return "Result stored in file"
