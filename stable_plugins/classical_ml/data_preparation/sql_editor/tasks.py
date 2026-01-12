import csv
import json
from json import loads
from tempfile import SpooledTemporaryFile
from typing import Optional

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE

from . import plugin
from .util import execute_sql, rows_to_records, serialize_rows

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{plugin.SQLEditor.instance.identifier}.process_sql",
    bind=True,
    ignore_result=True,
)
def process_sql(self, db_id: int) -> str:
    """Execute the stored SQL query and persist results in the chosen format."""
    TASK_LOGGER.info(f"Starting new SQL editor task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = loads(task_data.parameters or "{}")
    sql = params.get("sql", "")
    output_format = params.get("output_format", "csv")

    columns, rows = execute_sql(sql)
    file_name = "sql_result.csv"
    data_type = "entity/list"
    if output_format == "json":
        file_name = "sql_result.json"

    with SpooledTemporaryFile(mode="w", newline="") as output:
        if output_format == "json":
            json.dump(
                rows_to_records(columns, rows),
                output,
                ensure_ascii=True,
                default=str,
            )
            content_type = "application/json"
        else:
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(serialize_rows(rows))
            content_type = "text/csv"

        output.seek(0)
        STORE.persist_task_result(
            db_id,
            output,
            file_name,
            data_type,
            content_type,
        )

    return f"Result stored in {file_name}"
