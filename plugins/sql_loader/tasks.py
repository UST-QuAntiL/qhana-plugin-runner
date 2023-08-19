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

TASK_LOGGER = get_task_logger(__name__)


def get_href(db_host: str, db_port: Optional[int], db_database: str):
    result = db_host
    if db_port is not None:
        result += f":{db_port}"
    if db_port is not None or db_host != "":
        result += "/"
    return result + db_database


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

    TASK_LOGGER.info(f"Starting new db manager calculation task with db id '{db_id}'")
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

    db_database = db_database.removeprefix("file://")

    db_manager, db_type = db_enum.get_connected_db_manager(
        db_host, db_port, db_user, db_password, db_database
    )

    task_data.data.update(
        dict(
            db_tables_and_columns=db_manager.get_tables_and_columns(),
            db_type=db_type,
            db_host=db_host,
            db_port=db_port,
            db_user=db_user,
            db_password=db_password,
            db_database=db_database,
        )
    )

    task_data.save(commit=True)

    return "First step: checked database"


def retrieve_params_for_second_task(db_id: int, dumped_schema=None, debug_file=None) -> dict:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if debug_file is not None:
        with debug_file.open("a") as f:
            f.write(f"\n\ntask:\ntask_data.parameters: {task_data.parameters}\ntask_data.data: {task_data.data}")
            f.close()

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # Previous parameters
    previous_input = FirstInputParameters(
        db_enum=DBEnum[task_data.data["db_type"]],
        **{
            key: value
            for key, value in task_data.data.items()
            if key != "db_type" and key != "db_tables_and_columns"
        },
    )

    db_enum, db_host, db_port, db_user, db_password, db_database = prep_first_inputs(
        previous_input
    )
    params = dict(
        db_enum=db_enum,
        db_host=db_host,
        db_port=db_port,
        db_user=db_user,
        db_password=db_password,
        db_database=db_database,
    )

    TASK_LOGGER.info(f"Loaded input from previous step from db: {task_data.data}")
    print(f"Loaded input from previous step from db: {task_data.data}")


    input_params: SecondInputParameters = SecondInputParametersSchema().loads(
        task_data.parameters if dumped_schema is None else dumped_schema
    )

    params.update(input_params.__dict__)
    params["columns_list"] = ", ".join(json_load(input_params.columns_list))

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")
    print(f"Loaded input parameters from db: {str(input_params)}")

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
    **kwargs
) -> pd.DataFrame:

    db_database = db_database.removeprefix("file://")

    db_manager, _ = db_enum.get_connected_db_manager(
        db_host, db_port, db_user, db_password, db_database
    )
    df = None
    if custom_query:
        table_query = db_query
    else:
        table_query = f"SELECT {columns_list} FROM {table_name}"

    print(f"table_query: {table_query}")

    if table_query != "":
        if limit is not None:
            table_query = f"SELECT {table_query} LIMIT {limit}"
        df = db_manager.get_query_as_dataframe(table_query)

        # Check if given attribute can be used as a unique identifier
        # If so, use attribute
        if id_attribute in df and len(df[id_attribute].unique()) == len(df[id_attribute]):
            df["ID"] = df[id_attribute]
        # If not, use indices
        else:
            df["ID"] = list(range(df.shape[0]))
        df["href"] = [get_href(db_host, db_port, db_database)] * df.shape[0]
    else:
        db_manager.execute_query(db_query)

    return df


@CELERY.task(name=f"{SQLLoaderPlugin.instance.identifier}.second_task", bind=True)
def second_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new db manager calculation task with db id '{db_id}'")

    params = retrieve_params_for_second_task(db_id)
    df = second_task_execution(**params)

    # Output data
    if params["save_table"]:
        if df is not None:
            with SpooledTemporaryFile(mode="w") as output:
                df.to_csv(output, index=False)
                STORE.persist_task_result(
                    db_id,
                    output,
                    "entities.csv",
                    "entity",
                    "text/csv",
                )

    return "Second step: result stored in file"
