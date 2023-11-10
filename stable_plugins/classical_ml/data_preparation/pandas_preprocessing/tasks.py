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

from . import PDPreprocessing

from .schemas import (
    FirstInputParameters,
    FirstInputParametersSchema,
    SecondInputParameters,
    SecondInputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE

from pandas import read_csv
from pretty_html_table import build_table
from .backend.checkbox_list import get_checkbox_list_dict

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
            "entity",  # TODO keep original data type
            "text/csv",
        )
        task_data.data["file_url"] = task_file.file_storage_data
        table_html = df.to_html(max_rows=100, max_cols=100)
        if (
            len(str(table_html).encode("utf-8")) > 1000000
        ):  # Check if table requires more than 1MB
            table_html = df.to_html(max_rows=10, max_cols=10)
            if (
                len(str(table_html).encode("utf-8")) > 1000000
            ):  # Check if table requires more than 1MB
                table_html = "Table is too large to display"
        task_data.data["pandas_html"] = table_html
        task_data.data["columns_and_rows_html"] = get_checkbox_list_dict(
            {
                "columns": [str(el) for el in df.columns.tolist()],
                "rows": [str(el) for el in df.index.tolist()],
            }
        )
        task_data.data["columns_list"] = df.columns.tolist()

    task_data.save(commit=True)

    return "Saved original csv file"


@CELERY.task(name=f"{PDPreprocessing.instance.identifier}.preprocessing_task", bind=True)
def preprocessing_task(self, db_id: int, step_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new pandas preprocessing calculation task with db id '{db_id}' and step id '{step_id}'."
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
    if df is not None:
        file_id = step_id
        with SpooledTemporaryFile(mode="w") as output:
            df.to_csv(output, index=False)
            task_file = STORE.persist_task_result(
                db_id,
                output,
                f"preprocessed_file{file_id}.csv",
                "entity",  # TODO keep original data type
                "text/csv",
            )
            task_data.data["file_url"] = task_file.file_storage_data
            # task_data.data["pandas_html"] = df.to_html(max_rows=100)
            task_data.data["pandas_html"] = build_table(df, "grey_light")
            task_data.data["columns_and_rows_html"] = get_checkbox_list_dict(
                {
                    "columns": [str(el) for el in df.columns.tolist()],
                    "rows": [str(el) for el in df.index.tolist()],
                }
            )
            task_data.data["columns_list"] = df.columns.tolist()

    task_data.save(commit=True)

    return "Result stored in file"
