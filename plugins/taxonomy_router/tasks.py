from json import loads
from pathlib import PurePath
from typing import Optional

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    EntityTupleMixin,
    load_entities,
)
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import get_mimetype, open_url

from . import TaxonomyRouter
from .schemas import PIPELINE_FIELD_PREFIX

TASK_LOGGER = get_task_logger(__name__)


def _load_task(db_id: int) -> ProcessingTask:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)
    return task_data


def _taxonomy_ref(metadata: AttributeMetadata) -> Optional[str]:
    """Return the referenced taxonomy file name, or ``None`` for non-taxonomy attributes.

    Taxonomy references point into the taxonomies zip, e.g.
    ``taxonomies.zip:t_Gattung.json`` (the file name is the part after the
    colon). Other references target plain files such as ``people.csv`` and are
    excluded. The ``refTarget`` field is the reliable signal here. The ``type``
    field carries the attribute id in the sample data, not the data type.
    """
    ref_target = metadata.ref_target or ""
    if ".zip:" not in ref_target:
        return None
    return ref_target.split(":", 1)[1]


def _load_entity_attributes(entities_url: str) -> set:
    """Return the set of attribute names present in the entities file."""
    attributes: set = set()
    with open_url(entities_url) as response:
        mimetype = get_mimetype(response)
        for ent in load_entities(response, mimetype):
            if isinstance(ent, EntityTupleMixin):
                attributes.update(type(ent).entity_attributes)
                break
            attributes.update(ent.keys())
    return attributes


@CELERY.task(name=f"{TaxonomyRouter.instance.identifier}.preprocessing_task", bind=True)
def preprocessing_task(self, db_id: int) -> str:
    """Step 1 task: collect the taxonomy attributes for the second UI step."""
    TASK_LOGGER.info(f"Starting taxonomy router preprocessing with db id '{db_id}'")
    task_data = _load_task(db_id)

    params = loads(task_data.parameters or "{}")
    entities_url: Optional[str] = params.get("entities_url")
    if not entities_url:
        raise ValueError("No entities URL provided!")
    metadata_url: Optional[str] = params.get("entities_metadata_url")
    if not metadata_url:
        raise ValueError("No entities metadata URL provided!")
    taxonomies_zip_url: Optional[str] = params.get("taxonomies_zip_url")
    if not taxonomies_zip_url:
        raise ValueError("No taxonomies zip URL provided!")

    # The attribute metadata describes the full schema for all entity types. The
    # uploaded entities only contain a subset of those attributes, so collect the
    # attribute names actually present in the entities file.
    entity_attributes = _load_entity_attributes(entities_url)

    # Collect the taxonomy file names actually present in the uploaded zip.
    available_taxonomies = {
        PurePath(file_name).name
        for _, file_name in get_files_from_zip_url(taxonomies_zip_url, mode="t")
    }

    # Keep attributes that are present in the entities and whose referenced
    # taxonomy exists in the zip.
    taxonomy_attributes = []
    with open_url(metadata_url) as response:
        mimetype = get_mimetype(response)
        for element in load_entities(response, mimetype):
            metadata = AttributeMetadata.from_dict(element)
            if metadata.ID not in entity_attributes:
                continue
            ref = _taxonomy_ref(metadata)
            if ref and PurePath(ref).name in available_taxonomies:
                taxonomy_attributes.append(metadata.ID)

    TASK_LOGGER.info(
        f"Found {len(taxonomy_attributes)} taxonomy attribute(s) with a matching "
        f"taxonomy in the zip: {taxonomy_attributes}"
    )
    task_data.data["taxonomy_attributes"] = taxonomy_attributes
    task_data.save(commit=True)

    return f"Found {len(taxonomy_attributes)} taxonomy attribute(s)."


@CELERY.task(name=f"{TaxonomyRouter.instance.identifier}.processing_task", bind=True)
def processing_task(self, db_id: int) -> str:
    """Step 2 task: record the chosen pipeline routing.

    Calling the downstream pipelines is not implemented yet, so this only logs
    the selections and returns a summary.
    """
    TASK_LOGGER.info(f"Starting taxonomy router processing with db id '{db_id}'")
    task_data = _load_task(db_id)

    params = loads(task_data.parameters or "{}")
    selections = {
        key[len(PIPELINE_FIELD_PREFIX) :]: value
        for key, value in params.items()
        if key.startswith(PIPELINE_FIELD_PREFIX)
    }
    input_urls = {
        "entities_url": params.get("entities_url"),
        "entities_metadata_url": params.get("entities_metadata_url"),
        "taxonomies_zip_url": params.get("taxonomies_zip_url"),
    }

    TASK_LOGGER.info(f"Input parameters: {input_urls}")
    TASK_LOGGER.info(f"Selected pipeline routing: {selections}")

    return f"Routing recorded for {len(selections)} attribute(s): {selections}"
