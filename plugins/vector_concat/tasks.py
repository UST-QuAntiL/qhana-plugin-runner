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
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_array,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.plugin_utils.zip_utils import get_file_responses_from_zip
from qhana_plugin_runner.requests import get_mimetype, open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import VectorConcatPlugin
from .schemas import ACCEPTED_CONTENT_TYPES

TASK_LOGGER = get_task_logger(__name__)


class VectorSource(NamedTuple):
    name: str
    url: str
    zip_member: str | None
    dimensions: list[str]


def _source_dimensions(entities: list[Any]) -> list[str]:
    if not entities:
        return []

    first = entities[0]

    if isinstance(first, dict):
        return sorted(key for key in first if key not in ("ID", "href"))

    fields = list(first._fields)
    return fields[2:] if "href" in fields else fields[1:]


def _load_vector_entities(response, mimetype: str, url: str, zip_member: str | None):
    raw = list(load_entities(response, mimetype=mimetype))
    dimensions = _source_dimensions(raw)
    entities = list(ensure_array(iter(raw), strict=True))
    name = zip_member if zip_member is not None else retrieve_filename(response)

    return VectorSource(name, url, zip_member, dimensions), entities


def _load_entities_from_zip(
    zip_bytes: bytes, url: str
) -> Iterator[tuple[VectorSource, list]]:
    """Yield one source/entity list pair per file contained in the zip archive.

    The content of each member may be JSON, JSON lines or CSV; the format is
    detected dynamically from the file name or its content.
    """
    for response in get_file_responses_from_zip(zip_bytes):
        mimetype = get_mimetype(response)
        if not mimetype:
            raise ValueError(f"No Mimetype found for zip file '{response.url}'.")
        if mimetype not in ACCEPTED_CONTENT_TYPES:
            raise ValueError(
                f"Mimetype for zip file '{response.url}' is not one of {ACCEPTED_CONTENT_TYPES}"
            )
        yield _load_vector_entities(response, mimetype, url, response.url)


def _load_sources(urls: list[str]) -> tuple[list[VectorSource], list[list]]:
    """Load every input url, expanding zip archives into their members.

    Returns the sources and their entity lists in concatenation order.
    """
    sources = []
    entities_list = []

    for url in urls:
        with open_url(url, stream=True) as response:
            mimetype = get_mimetype(response)
            if not mimetype:
                raise ValueError(f"Could not determine mimetype of {url}!")
            if mimetype == "application/zip":
                loaded = list(_load_entities_from_zip(response.content, url))
            else:
                loaded = [_load_vector_entities(response, mimetype, url, None)]

        for source, entities in loaded:
            sources.append(source)
            entities_list.append(entities)

    return sources, entities_list


def _build_dimension_mapping(
    sources: list[VectorSource], dimension_count: int
) -> list[dict]:
    """Map every output dimension back to the input file it came from.

    Raises:
        ValueError: if the mapped columns do not add up to ``dimension_count``.
    """
    mapping = []

    for input_index, source in enumerate(sources):
        for source_dimension in source.dimensions:
            mapping.append(
                {
                    "ID": f"dim{len(mapping)}",
                    "href": "",
                    "inputIndex": input_index,
                    "source": source.name,
                    "sourceUrl": source.url,
                    "zipMember": source.zip_member or "",
                    "sourceDimension": source_dimension,
                }
            )

    if len(mapping) != dimension_count:
        raise ValueError(
            f"Could not map the output dimensions: the input files declare {len(mapping)} "
            f"columns but the concatenated entities have {dimension_count} dimensions!"
        )

    return mapping


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

    sources, entities_list = _load_sources(urls)

    combined = []
    dimension_count = 0
    for entity_index, entities_tuple in enumerate(zip(*entities_list, strict=True)):
        e1 = entities_tuple[0]
        for e in entities_tuple[1:]:
            assert e.ID == e1.ID
            assert e.href == e1.href

        values = []
        for e in entities_tuple:
            values.extend(e.values)

        if entity_index == 0:
            dimension_count = len(values)
        elif len(values) != dimension_count:
            raise ValueError(
                f"Entity '{e1.ID}' has {len(values)} dimensions but the first entity "
                f"has {dimension_count}! All entities must have the same dimensions."
            )

        entity = {"ID": e1.ID, "href": e1.href}
        entity.update({f"dim{i}": v for i, v in enumerate(values)})
        combined.append(entity)

    dimension_mapping = _build_dimension_mapping(sources, dimension_count)

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
            attributes = ["ID", "href"] + [f"dim{i}" for i in range(dimension_count)]
            save_entities(combined, output, "text/csv", attributes=attributes)
            STORE.persist_task_result(
                db_id, output, f"{output_name}.csv", "entity/vector", "text/csv"
            )

    with SpooledTemporaryFile(mode="w") as mapping_output:
        save_entities(dimension_mapping, mapping_output, "application/json")
        STORE.persist_task_result(
            db_id,
            mapping_output,
            f"{output_name}_dimension_mapping.json",
            "entity/dimension-mapping",
            "application/json",
        )

    return "Result stored in file"
