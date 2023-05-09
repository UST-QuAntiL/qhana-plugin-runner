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

import os
from tempfile import SpooledTemporaryFile

from typing import Optional, List

from celery.utils.log import get_task_logger

from . import QCNN

from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
    ensure_dict,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE


TASK_LOGGER = get_task_logger(__name__)





@CELERY.task(name=f"{QCNN.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new quantum cnn calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)
    
    train_data_url = input_params.train_data_url
    train_label_url = input_params.train_label_url
    test_data_url = input_params.test_data_url
    test_label_url = input_params.test_label_url
    randomly_shuffle = input_params.randomly_shuffle
    epochs = input_params.epochs
    optimizer = input_params.optimizer
    lr = input_params.lr
    qcnn_enum = input_params.qcnn_enum
    num_layers = input_params.num_layers
    batch_size = input_params.batch_size
    weight_init = input_params.weight_init
    weights_to_wiggle = input_params.weights_to_wiggle
    backend = input_params.backend
    shots = input_params.shots
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend

    if ibmq_token == "****":
        TASK_LOGGER.info("Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info("IBMQ_TOKEN environment variable not set")

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")
    
# hier könnte ihr Code stehen
    
    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        STORE.persist_task_result(
            db_id,
            output,
            "labels.json",
            "entity/label",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        STORE.persist_task_result(
            db_id,
            output,
            "plot.html",
            "plot",
            "text/html",
        )

    with SpooledTemporaryFile(mode="w") as output:
        STORE.persist_task_result(
            db_id,
            output,
            "confusion_matrix.html",
            "plot",
            "text/html",
        )

    with SpooledTemporaryFile(mode="w") as output:
        STORE.persist_task_result(
            db_id,
            output,
            "qnn-weights.json",
            "qnn-weights",
            "application/json",
        )


    return "Result stored in file"
