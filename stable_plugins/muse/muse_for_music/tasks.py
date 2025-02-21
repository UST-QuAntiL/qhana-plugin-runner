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
from tempfile import SpooledTemporaryFile
from typing import Optional
from zipfile import ZipFile

import requests
from celery.utils.log import get_task_logger
from requests import session

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import tuple_serializer
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.registry_client import PLUGIN_REGISTRY_CLIENT
from qhana_plugin_runner.storage import STORE

from . import M4MLoaderPlugin
from .muse_for_music_client import Muse4MusicClient
from .schemas import InputParameters, InputParametersSchema
from .util import (
    OpusEntity,
    PartEntity,
    PersonEntity,
    SubpartEntity,
    VoiceEntity,
    get_attribute_metadata,
    opus_to_entity,
    part_to_entity,
    person_to_entity,
    subpart_to_entity,
    taxonomy_to_entity,
    voice_to_entity,
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

    muse_for_music_url: Optional[str] = None

    if input_params.muse_url:
        muse_for_music_url = input_params.muse_url

    if muse_for_music_url is None:
        muse_for_music_url = get_muse_for_music_url_from_registry()

    # TODO: load from env vars

    client = Muse4MusicClient(muse_for_music_url)
    metadata = get_attribute_metadata()

    with requests.session():
        client.login(input_params.username, input_params.password)
        client.test_login()

        opuses = client.get_opuses()
        serializer = tuple_serializer(OpusEntity._fields, metadata)
        serialized_opuses = [serializer(opus_to_entity(o)) for o in opuses]

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                serialized_opuses,
                output,
                "text/csv",
                attributes=OpusEntity._fields,
            )
            STORE.persist_task_result(
                db_id, output, "opuses.csv", "entity/list", "text/csv"
            )

        people = client.get_people()
        serializer = tuple_serializer(PersonEntity._fields, metadata)
        serialized_persons = [serializer(person_to_entity(p)) for p in people]

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                serialized_persons,
                output,
                "text/csv",
                attributes=PersonEntity._fields,
            )
            STORE.persist_task_result(
                db_id, output, "people.csv", "entity/list", "text/csv"
            )

        parts = client.get_parts()
        serializer = tuple_serializer(PartEntity._fields, metadata)
        mapped = [part_to_entity(p) for p in parts]
        serialized_parts = [serializer(p) for p in mapped]

        part_id_to_opus_id = {p.ID: p.opus for p in mapped}

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                serialized_parts, output, "text/csv", attributes=PartEntity._fields
            )
            STORE.persist_task_result(
                db_id, output, "parts.csv", "entity/list", "text/csv"
            )

        subparts = client.get_subparts()
        serializer = tuple_serializer(SubpartEntity._fields, metadata)
        mapped = [subpart_to_entity(p, part_id_to_opus_id) for p in subparts]
        serialized_subparts = [serializer(p) for p in mapped]

        subpart_id_to_part_id = {sp.ID: sp.part for sp in mapped}

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                serialized_subparts, output, "text/csv", attributes=SubpartEntity._fields
            )
            STORE.persist_task_result(
                db_id, output, "subparts.csv", "entity/list", "text/csv"
            )

        voices = []
        for subpart in mapped:
            voices.extend(client.get_voices(subpart.href))
        serializer = tuple_serializer(VoiceEntity._fields, metadata)
        mapped = [
            voice_to_entity(v, subpart_id_to_part_id, part_id_to_opus_id) for v in voices
        ]
        serialized_voices = [serializer(v) for v in mapped]

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                serialized_voices, output, "text/csv", attributes=VoiceEntity._fields
            )
            STORE.persist_task_result(
                db_id, output, "voices.csv", "entity/list", "text/csv"
            )

        # TODO: partial graphs? (opus to parts, parts to subparts, subparts to voices, citations, voice relations, etc.)
        # TODO: full graph? (all partial graphs merged)

        taxonomy_urls = client.get_taxonomies()
        tmp_zip_file = SpooledTemporaryFile(mode="wb")
        zip_file = ZipFile(tmp_zip_file, "w")

        for taxonomy_url in taxonomy_urls:
            taxonomy = client.get_taxonomy(taxonomy_url)
            taxonomy_entity = taxonomy_to_entity(taxonomy).to_dict()
            zip_file.writestr(
                taxonomy_entity["GRAPH_ID"] + ".json", json.dumps(taxonomy_entity)
            )

        zip_file.close()

        STORE.persist_task_result(
            db_id,
            tmp_zip_file,
            "taxonomies.zip",
            "graph/taxonomy",
            "application/zip",
        )

        metadata_entities = [metadata.to_dict() for metadata in metadata.values()]

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                metadata_entities,
                output,
                "application/json",
            )
            STORE.persist_task_result(
                db_id,
                output,
                "attribute_metadata.json",
                "entity/attribute-metadata",
                "application/json",
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
