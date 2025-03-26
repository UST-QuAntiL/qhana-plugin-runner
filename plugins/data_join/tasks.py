# Copyright 2025 QHAna plugin runner contributors.
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

from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Optional

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
)
from qhana_plugin_runner.requests import (
    open_url,
    retrieve_filename,
    get_mimetype,
    retrieve_data_type,
    retrieve_attribute_metadata_url,
)
from qhana_plugin_runner.storage import STORE

from . import DataJoin


TASK_LOGGER = get_task_logger(__name__)


def _get_file_metadata(entity_url: str) -> tuple[dict, list[str]]:
    with open_url(entity_url) as response:
        first_entity = next(
            ensure_dict(
                load_entities(response, get_mimetype(response)),
            )
        )
        attributes = list(first_entity.keys())

        metadata = {
            "data": entity_url,
            "attribute_metadata": retrieve_attribute_metadata_url(response),
            "name": retrieve_filename(response),
            "data_type": retrieve_data_type(response),
            "content_type": get_mimetype(response),
        }

        return metadata, attributes


@CELERY.task(name=f"{DataJoin.instance.identifier}.load_base", bind=True)
def load_base(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Load info from the base entity file for job '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = loads(task_data.parameters)

    data = task_data.data
    assert data is not None
    assert isinstance(data, dict)

    base_data_url = params["base"]
    data["base"], data["attributes"] = _get_file_metadata(base_data_url)

    task_data.save(commit=True)

    return "Loaded metadata from base data finished."


@CELERY.task(name=f"{DataJoin.instance.identifier}.add_data_to_join", bind=True)
def add_data_to_join(self, db_id: int, entity_url: str, join_attr: str):
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to update data!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    data = task_data.data
    if not data:
        data = {}
        task_data.data = data
    assert isinstance(data, dict)

    attributes = data.get("attributes")
    assert isinstance(attributes, (list, tuple, set))
    if join_attr not in attributes:
        msg = f"Cannot join to a nonexistant attribute {join_attr}! (Attributes: {attributes})"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    file_metadata, _ = _get_file_metadata(entity_url)
    file_metadata["join_attr"] = join_attr

    joins = data.setdefault("joins", [])
    assert isinstance(joins, list)
    joins.append(file_metadata)

    task_data.save(commit=True)

    # FIXME start a new substep

    return "Finished adding data to join!"


@CELERY.task(name=f"{DataJoin.instance.identifier}.join_data", bind=True)
def join_data(self, db_id: int, entity_url: str, join_attr: str):
    pass  # FIXME join data to base and save entities

    return "Finished joining data!"
