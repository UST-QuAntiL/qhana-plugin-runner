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

from typing import Optional, Tuple

import pandas as pd
from celery.utils.log import get_task_logger

from . import SQLLoaderPlugin

from .schemas import (
    FirstInputParameters,
    FirstInputParametersSchema,
    SecondInputParameters,
    SecondInputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE

from json import loads as json_load
from .backend.db_enum import DBEnum
from .backend.checkbox_list import get_checkbox_list_dict

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


def get_href(db_host: str, db_port: Optional[int], db_database: str):
    if db_port is not None and db_host:
        return f"{db_host}:{db_port}/{db_database}"
    if db_host:
        return f"{db_host}/{db_database}"
    return db_database


def prep_first_inputs(
    input_params: FirstInputParameters,
) -> Tuple[DBEnum, Optional[str], Optional[int], Optional[str], Optional[str], str]:
    db_enum = input_params.db_enum
    db_host = None if input_params.db_host == "" else input_params.db_host
    db_port = None if input_params.db_port == -1 else input_params.db_port
    db_user = None if input_params.db_user == "" else input_params.db_user
    db_password = None if input_params.db_password == "" else input_params.db_password
    db_database = input_params.db_database

    return db_enum, db_host, db_port, db_user, db_password, db_database


@CELERY.task(name=f"{SQLLoaderPlugin.instance.identifier}.first_task", bind=True)
def first_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new sql loader calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: FirstInputParameters = FirstInputParametersSchema().loads(
        task_data.parameters
    )

    db_enum, db_host, db_port, db_user, db_password, db_database = prep_first_inputs(
        input_params
    )

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    db_manager, db_type = db_enum.get_connected_db_manager(
        db_host, db_port, db_user, db_password, db_database
    )

    tables_and_columns = db_manager.get_tables_and_columns()
    db_manager.disconnect()
    checkbox_list = get_checkbox_list_dict(tables_and_columns)

    task_data.data.update(
        {
            "db_tables_and_columns": tables_and_columns,
            "checkbox_list": checkbox_list,
            "db_type": db_type,
            "db_host": db_host,
            "db_port": db_port,
            "db_user": db_user,
            "db_password": db_password,
            "db_database": db_database,
        }
    )

    task_data.save(commit=True)

    return "First step: checked database"


def retrieve_params_for_second_task(db_id: int, dumped_schema=None) -> dict:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # Previous parameters
    params = {
        "db_enum": DBEnum[task_data.data["db_type"]],
        "db_host": task_data.data["db_host"],
        "db_port": task_data.data["db_port"],
        "db_user": task_data.data["db_user"],
        "db_password": task_data.data["db_password"],
        "db_database": task_data.data["db_database"],
    }

    input_params: SecondInputParameters = SecondInputParametersSchema().loads(
        task_data.parameters if dumped_schema is None else dumped_schema
    )

    params.update(input_params.__dict__)
    params["columns_list"] = (
        ", ".join(json_load(input_params.columns_list)) if params["columns_list"] else []
    )

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    return params


def second_task_execution(
    db_enum: DBEnum,
    db_host: str,
    db_port: int,
    db_user: str,
    db_password: str,
    db_database: str,
    custom_query: bool,
    db_query: str,
    table_name: str,
    columns_list: str,
    id_attribute: str,
    limit: Optional[int] = None,
    retrieval_only: bool = False,
    **kwargs,
) -> pd.DataFrame:
    db_database = db_database.removeprefix("file://")

    db_manager, _ = db_enum.get_connected_db_manager(
        db_host, db_port, db_user, db_password, db_database
    )
    df = None
    if custom_query:
        db_query = db_query.strip()
        table_query = db_query if db_query.startswith("SELECT") else ""
    else:
        table_query = (
            f"SELECT {columns_list} FROM {table_name}" if columns_list != "" else ""
        )
        retrieval_only = True

    if table_query != "":
        if limit is not None and isinstance(limit, int) and limit > 0:
            table_query = table_query.rstrip(";")
            # Add \n's to avoid issues with comments
            # This does not avoid ';  -- end of line'
            table_query = f"SELECT * FROM (\n{table_query}\n) AS temp LIMIT {limit}"
        df = db_manager.get_query_as_dataframe(table_query)

        # Check if given attribute can be used as a unique identifier
        # If so, use attribute
        if id_attribute in df and len(df[id_attribute].unique()) == len(df[id_attribute]):
            df["ID"] = df[id_attribute]
        # If not, use indices
        else:
            df["ID"] = list(range(df.shape[0]))
        df["href"] = [get_href(db_host, db_port, db_database)] * df.shape[0]
    elif not retrieval_only:
        db_manager.execute_query(db_query)

    db_manager.disconnect()

    return df


@CELERY.task(name=f"{SQLLoaderPlugin.instance.identifier}.second_task", bind=True)
def second_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new sql loader calculation task with db id '{db_id}'")

    params = retrieve_params_for_second_task(db_id)
    df = second_task_execution(**params)

    # Delete password and credentials
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)
    for entry in ["db_password", "db_user"]:
        del task_data.data[entry]
        del params[entry]
    task_data.save(commit=True)

    # Output data
    if params["save_table"]:
        info_str = "custom_query" if params["custom_query"] else params["table_name"]
        if df is not None:
            with SpooledTemporaryFile(mode="w") as output:
                df.to_csv(output, index=False)
                STORE.persist_task_result(
                    db_id,
                    output,
                    f"entities_sql-loader_{info_str}.csv",
                    "entity",
                    "text/csv",
                )

    return "Second step: result stored in file"


@CELERY.task(
    name=f"{SQLLoaderPlugin.instance.identifier}.get_second_task_html", bind=True
)
def get_second_task_html(
    self,
    db_id: int,
    arguments: str,
) -> str:
    params = retrieve_params_for_second_task(db_id, dumped_schema=arguments)
    df = second_task_execution(**params, limit=10, retrieval_only=True)
    return get_table_html(df) if df is not None else ""
