# Copyright 2024 QHAna plugin runner contributors.
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
from json import loads as json_load
from tempfile import SpooledTemporaryFile
from typing import Optional

import requests
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.registry_client import PLUGIN_REGISTRY_CLIENT

from requests import session

from . import M4MLoaderPlugin
from .muse_for_music_client import Muse4MusicClient
from .schemas import InputParameters, InputParametersSchema
from .util import (
    opus_to_entity,
    person_to_entity,
    part_to_entity,
    taxonomy_to_entity,
    OpusEntity,
    PersonEntity,
    PartEntity,
)

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{M4MLoaderPlugin.instance.identifier}.import_data", bind=True)
def import_data(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new sql loader calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    muse_for_music_url = None

    if muse_for_music_url is None:
        muse_for_music_url = get_muse_for_music_url_from_registry()

    client = Muse4MusicClient(muse_for_music_url)

    with requests.session():
        client.login(input_params.username, input_params.password)
        client.test_login()

        opuses = client.get_opuses()
        opus_entities = [opus_to_entity(o)._asdict() for o in opuses]

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                opus_entities, output, "text/csv", attributes=OpusEntity._fields
            )
            STORE.persist_task_result(
                db_id, output, "opuses.json", "entity/list", "text/csv"
            )

        people = client.get_people()
        person_entities = [person_to_entity(p)._asdict() for p in people]

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                person_entities, output, "text/csv", attributes=PersonEntity._fields
            )
            STORE.persist_task_result(
                db_id, output, "people.json", "entity/list", "text/csv"
            )

        parts = client.get_parts()
        part_entities = [part_to_entity(p)._asdict() for p in parts]

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                part_entities, output, "text/csv", attributes=PartEntity._fields
            )
            STORE.persist_task_result(
                db_id, output, "parts.json", "entity/list", "text/csv"
            )

        # TODO: subparts
        # TODO: taxonomies

        taxonomies = client.get_taxonomies()
        first_taxonomy = client.get_taxonomy(taxonomies[0])
        first_taxonomy_entity = taxonomy_to_entity(first_taxonomy)._asdict()

        with SpooledTemporaryFile(mode="w") as output:
            output.write(json.dumps(first_taxonomy_entity))
            STORE.persist_task_result(
                db_id, output, "taxonomy1.json", "graph/taxonomy", "application/json"
            )

    return "Import finished."


def get_muse_for_music_url_from_registry() -> Optional[str]:
    muse_for_music_url = None

    with session():
        service_result = PLUGIN_REGISTRY_CLIENT.search_by_rel(
            ["service", "collection"], query_params={"service-id": "muse-for-music"}
        )

        if service_result and service_result.data.get("collectionSize") == 1:
            service_api_link = service_result.data.get("items", [])[0]
            if service_result.embedded:
                service_object = service_result.embedded.get(service_api_link["href"])
            else:
                service_object = PLUGIN_REGISTRY_CLIENT.fetch_by_api_link(
                    service_api_link
                )
            muse_for_music_url = (
                service_object.data.get("url") if service_object else None
            )

    return muse_for_music_url
