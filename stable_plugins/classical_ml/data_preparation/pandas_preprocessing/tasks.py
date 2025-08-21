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
from pandas import read_csv, read_json

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile
from qhana_plugin_runner.requests import (
    get_mimetype,
    open_url,
    open_url_as_file_like_simple,
    retrieve_attribute_metadata_url,
    retrieve_data_type,
    retrieve_filename,
)
from qhana_plugin_runner.storage import STORE

from . import PDPreprocessing
from .backend.checkbox_list import get_checkbox_list_dict
from .backend.pandas_preprocessing import PreprocessingEnum
from .schemas import (
    FirstInputParameters,
    FirstInputParametersSchema,
    SecondInputParameters,
    SecondInputParametersSchema,
)

TASK_LOGGER = get_task_logger(__name__)


def get_table_html(df) -> str:
    table_html = df.to_html(max_rows=100, max_cols=100)
    if (
        len(str(table_html).encode("utf-8")) > 1000000
    ):  # Check if table requires more than 1MB
        table_html = df.to_html(max_rows=10, max_cols=10)
        if (
            len(str(table_html).encode("utf-8")) > 1000000
        ):  # Check if table requires more than 1MB
            table_html = "Table is too large to display"
    return table_html


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

    try:
        with open_url_as_file_like_simple(file_url) as file_:
            df = read_json(file_, orient="records")
    except:
        with open_url_as_file_like_simple(file_url) as file_:
            df = read_csv(file_)

    file_name: str
    data_type: Optional[str]
    attr_metadata_url: Optional[str] = None

    with open_url(file_url, stream=True) as response:
        file_name = retrieve_filename(response)
        attr_metadata_url = retrieve_attribute_metadata_url(response)
        data_type = retrieve_data_type(response)

    if not data_type:
        data_type = "entity/list"

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        df.to_csv(output, index=False)
        task_file = STORE.persist_task_temp_file(
            db_id,
            output,
            "original_file.csv",
            "text/csv",
        )
        metadata = {}
        metadata["original_file_name"] = file_name
        metadata["original_data_type"] = data_type
        if attr_metadata_url:
            metadata["attribute_metadata_url"] = attr_metadata_url
        metadata["task_file_id"] = task_file.id

        table_html = get_table_html(df)
        metadata["pandas_html"] = table_html
        metadata["columns_and_rows_html"] = get_checkbox_list_dict(
            {
                "columns": [str(el) for el in df.columns.tolist()],
                "rows": [str(el) for el in df.index.tolist()],
            }
        )
        metadata["columns_list"] = df.columns.tolist()
        task_data.data = metadata

    task_data.save(commit=True)

    return "Saved original csv file"


@CELERY.task(name=f"{PDPreprocessing.instance.identifier}.preprocessing_task", bind=True)
def preprocessing_task(self, db_id: int, step_id: int, final: bool = False) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new pandas preprocessing calculation task with db id '{db_id}' and step id '{step_id}'."
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)
    assert isinstance(task_data.data, dict)  # type checker assert

    input_params: SecondInputParameters = SecondInputParametersSchema().loads(
        task_data.parameters
    )

    task_file_id: int = task_data.data["task_file_id"]
    TASK_LOGGER.info(f"task_file_id: {task_file_id}")
    preprocessing_enum = input_params.preprocessing_enum

    if preprocessing_enum == PreprocessingEnum.split_column:
        task_data.data["metadata_outdated"] = True

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    task_file = TaskFile.get_by_id(task_file_id)

    if task_file is None:
        raise KeyError(f"Could not find task file with id {task_file_id} in DB.")

    with STORE.open(task_file) as file_:
        df = read_csv(file_)

    df = preprocessing_enum.preprocess_df(
        df, {k: v for k, v in input_params.__dict__.items() if k != "preprocessing_enum"}
    )

    # Output data
    if df is not None:
        with SpooledTemporaryFile(mode="w") as output:
            df.to_csv(output, index=False)
            if final:
                file_name = f"{task_data.data['original_file_name']}.csv"
                if task_data.data.get("metadata_outdated"):
                    file_name = f"pd_preprocessed_{file_name}"
                task_file = STORE.persist_task_result(
                    db_id,
                    output,
                    file_name,
                    task_data.data["original_data_type"],
                    "text/csv",
                )
            else:
                task_file = STORE.persist_task_temp_file(
                    db_id,
                    output,
                    f"pd_preprocessed_{step_id}from_{task_data.data['original_file_name']}.csv",
                    "text/csv",
                )
            task_data.data["task_file_id"] = task_file.id

            task_data.data["pandas_html"] = get_table_html(df)
            task_data.data["columns_and_rows_html"] = get_checkbox_list_dict(
                {
                    "columns": [str(el) for el in df.columns.tolist()],
                    "rows": [str(el) for el in df.index.tolist()],
                }
            )
            task_data.data["columns_list"] = df.columns.tolist()

        attr_metadata_url = task_data.data["attribute_metadata_url"]
        if final and isinstance(attr_metadata_url, str):
            if not task_data.data.get("metadata_outdated"):
                # relay attribute metadata
                with open_url(attr_metadata_url) as response:
                    file_name = retrieve_filename(response)
                    mimetype = get_mimetype(response)
                if mimetype is not None:
                    STORE.persist_task_result(
                        db_id,
                        attr_metadata_url,
                        file_name=file_name,
                        file_type="entity/attribute-metadata",
                        mimetype=mimetype,
                        storage_provider="url_file_store",
                    )

    task_data.save(commit=True)

    return "Result stored in file"
