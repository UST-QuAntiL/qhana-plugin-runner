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

from . import DBManagerPlugin

from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{DBManagerPlugin.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new db manager calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    db_enum = input_params.db_enum
    db_host = input_params.db_host
    db_port = input_params.db_port
    db_user = input_params.db_user
    db_password = input_params.db_password
    db_database = input_params.db_database
    db_query = input_params.db_query

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")
    
    db_manager = db_enum.get_connected_db_manager(db_host, db_port, db_user, db_password, db_database)

    result = db_manager.execute_query(db_query)
    print(result)
    
    # Output data
    # with SpooledTemporaryFile(mode="w") as output:
    #     STORE.persist_task_result(
    #         db_id,
    #         output,
    #         "labels.json",
    #         "entity/label",
    #         "application/json",
    #     )


    return "Result stored in file"
