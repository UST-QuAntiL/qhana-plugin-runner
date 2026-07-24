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

from json import loads
from tempfile import SpooledTemporaryFile
from typing import Iterator, List, Optional

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_array,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.plugin_utils.zip_utils import get_file_responses_from_zip
from qhana_plugin_runner.requests import get_mimetype, open_url
from qhana_plugin_runner.storage import STORE

from . import VectorConcatPlugin
from .schemas import ACCEPTED_CONTENT_TYPES

TASK_LOGGER = get_task_logger(__name__)


def _load_entities_from_zip(zip_bytes: bytes) -> Iterator[List]:
    """Yield one entity list per file contained in the zip archive.

    The content of each member may be JSON, JSON lines or CSV; the format is
    detected dynamically from the file name or its content.
    """
    for response in get_file_responses_from_zip(zip_bytes):
        mimetype = response.headers["Content-Type"]
        if not mimetype:
            raise ValueError(f"No Mimetype found for zip file '{response.url}'.")
        if mimetype not in ACCEPTED_CONTENT_TYPES:
            raise ValueError(
                f"Mimetype for zip file '{response.url}' is not one of {ACCEPTED_CONTENT_TYPES}"
            )
        yield list(
            ensure_array(
                load_entities(response, mimetype=mimetype),
                strict=True,
            )
        )


@CELERY.task(name=f"{VectorConcatPlugin.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new vector concat calculation task with db id {db_id}")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = loads(task_data.parameters or "{}")
    urls = params.get("urls", "").splitlines()
    TASK_LOGGER.info(f"URLS: {urls}")
    output_format = params.get("output_format", "csv")
    output_suffix = params.get("output_suffix", "").strip()
    output_name = f"concatenated_{output_suffix}" if output_suffix else "concatenated"

    entities_list = []
    for url in urls:
        with open_url(url, stream=True) as x:
            mimetype = get_mimetype(x)
            if not mimetype:
                raise ValueError(f"Could not determine mimetype of {url}!")
            if mimetype == "application/zip":
                entities_list.extend(_load_entities_from_zip(x.content))
            else:
                entities = list(
                    ensure_array(load_entities(x, mimetype=mimetype), strict=True)
                )
                entities_list.append(entities)

    combined = []
    for entities_tuple in zip(*entities_list, strict=True):
        e1 = entities_tuple[0]
        for e in entities_tuple[1:]:
            assert e.ID == e1.ID
            assert e.href == e1.href

        values = []
        for e in entities_tuple:
            values.extend(e.values)

        entity = {"ID": e1.ID, "href": e1.href}
        entity.update({f"dim{i}": v for i, v in enumerate(values)})
        combined.append(entity)

    with SpooledTemporaryFile(mode="w") as output:
        if output_format == "json":
            save_entities(combined, output, "application/json")
            STORE.persist_task_result(
                db_id,
                output,
                f"{output_name}.json",
                "entity/vector",
                "application/json",
            )
        elif output_format == "lines":
            save_entities(combined, output, "application/X-lines+json")
            STORE.persist_task_result(
                db_id,
                output,
                f"{output_name}.jsonl",
                "entity/vector",
                "application/X-lines+json",
            )
        else:
            attributes = ["ID", "href"] + [f"dim{i}" for i in range(len(combined[0]) - 2)]
            save_entities(combined, output, "text/csv", attributes=attributes)
            STORE.persist_task_result(
                db_id, output, f"{output_name}.csv", "entity/vector", "text/csv"
            )

    return "Result stored in file"
