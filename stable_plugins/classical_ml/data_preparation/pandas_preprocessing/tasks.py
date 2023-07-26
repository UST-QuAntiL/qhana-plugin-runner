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

from typing import Optional, List
from json import loads

from celery.utils.log import get_task_logger

from . import PDPreprocessing

from .schemas import (
    FirstInputParameters,
    FirstInputParametersSchema,
    SecondInputParameters,
    SecondInputParametersSchema,
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

from json import loads as json_load
from pandas import read_csv

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{PDPreprocessing.instance.identifier}.first_task", bind=True)
def first_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new pandas preprocessing calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: FirstInputParameters = FirstInputParametersSchema().loads(
        task_data.parameters
    )

    file_url = input_params.file_url

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    df = read_csv(file_url)

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        df.to_csv(output, index=False)
        task_file = STORE.persist_task_result(
            db_id,
            output,
            "file.csv",
            "entity",
            "text/csv",
        )
        print(f"task_file: {task_file}")
        task_data.data["file_url"] = loads(task_file.task.parameters)["fileUrl"]

    task_data.save(commit=True)

    return "Saved original csv file"


@CELERY.task(name=f"{PDPreprocessing.instance.identifier}.second_task", bind=True)
def second_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new pandas preprocessing calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: SecondInputParameters = SecondInputParametersSchema().loads(
        task_data.parameters
    )

    file_url: str = task_data.data["file_url"]
    TASK_LOGGER.info(f"file_url: {file_url}")
    preprocessing_enum = input_params.preprocessing_enum

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    df = read_csv(file_url)
    df = preprocessing_enum.preprocess_df(
        df, {k: v for k, v in input_params.__dict__.items() if k != "preprocessing_enum"}
    )

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        df.to_csv(output, index=False)
        STORE.persist_task_result(
            db_id,
            output,
            "preprocessed_file.csv",
            "entity",
            "text/csv",
        )

    return "Result stored in file"
