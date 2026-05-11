# Copyright 2026 QHAna plugin runner contributors.
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

import json
from json import loads
from tempfile import SpooledTemporaryFile
from typing import Optional

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE

from . import Plugin

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Plugin.instance.identifier}.store_texts_task", bind=True)
def store_texts_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting text-storing task with db_id={db_id}")

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(db_id)
    if not task_data:
        msg = f"No task data found for DB ID {db_id}!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    try:
        parameters = loads(task_data.parameters or "{}")
        qasm_code = parameters.get("qasmCode")
        if not qasm_code:
            raise ValueError("Missing required parameter: 'qasmCode'")
    except (json.JSONDecodeError, ValueError) as e:
        TASK_LOGGER.error(f"Error parsing task parameters: {e}")
        raise ValueError(f"Invalid parameters: {e}")

    try:
        with SpooledTemporaryFile(mode="w") as output:
            output.write(qasm_code)
            STORE.persist_task_result(
                db_id,
                output,
                "circuit.qasm",
                "executable/circuit",
                "text/x-qasm",
            )
        TASK_LOGGER.info("QASM file successfully stored.")
    except Exception as e:
        TASK_LOGGER.error(f"Error storing QASM file: {e}")
        raise
    TASK_LOGGER.info("Task completed successfully.")
    return "Files created successfully."
