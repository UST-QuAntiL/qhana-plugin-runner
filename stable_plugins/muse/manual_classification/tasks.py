from tempfile import SpooledTemporaryFile

from typing import Optional
from json import loads
import copy
import mimetypes

from celery.utils.log import get_task_logger

from . import ManualClassification
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.requests import open_url
from typing import Dict, Optional

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{ManualClassification.instance.identifier}.pre_render_classification_task",
    bind=True,
)
def pre_render_classification(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new entity filter task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    input_file_url: Optional[str] = params.get("input_file_url", None)

    TASK_LOGGER.info(
        f"Loaded input parameters from db: input_file_url='{input_file_url}'"
    )

    if input_file_url is None or not input_file_url:
        msg = "No input file URL provided!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    entities = []
    entity_annotation = {}
    attr_list = set()
    with open_url(input_file_url, stream=True) as url_data:
        try:
            mimetype = url_data.headers["Content-Type"]
        except KeyError:
            mimetype = mimetypes.MimeTypes().guess_type(url=input_file_url)[0]
        task_data.data["mimetype"] = mimetype
        input_entities = ensure_dict(load_entities(file_=url_data, mimetype=mimetype))

        if not input_entities:
            msg = "No entities could be loaded!"
            TASK_LOGGER.error(msg)
            raise ValueError(msg)

        # create data structure for annotations and extract attributes used in entities
        for entity in input_entities:
            entities.append(entity)
            entity_annotation[entity["ID"]] = []
            for attr in entity.keys():
                attr_list.add(attr)

    attr_list.remove("ID")
    # store in data of task_data
    task_data.data["entity_annotation"] = entity_annotation
    task_data.data["entity_list"] = entities
    task_data.data["attr_list"] = ["ID"] + list(attr_list)
    task_data.data["input_file_url"] = input_file_url
    task_data.save(commit=True)

    return "Classification pre-rendering successful."


@CELERY.task(
    name=f"{ManualClassification.instance.identifier}.add_class",
    bind=True,
)
def add_class(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new add class task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    class_identifier: Optional[str] = params.get("class_identifier", None)

    TASK_LOGGER.info(
        f"Loaded input parameters from db: class_identifier='{class_identifier}', params='{params}'"
    )

    if (
        class_identifier is None or not class_identifier
    ):  # should not happen because of form validation
        msg = "No class identified provided!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    entity_annotation = task_data.data["entity_annotation"]
    for id in entity_annotation.keys():
        if params.get(id):
            tmp = set(entity_annotation[id])
            tmp.add(class_identifier)
            entity_annotation[id] = list(tmp)

    # store in data of task_data
    # task_data.data["entity_annotation"] = entity_annotation # TODO check if changes happen
    task_data.save(commit=True)

    return "Adding new class successful."


@CELERY.task(
    name=f"{ManualClassification.instance.identifier}.save_classification",
    bind=True,
)
def save_classification(self, task_log: str, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new add class task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id}!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    annotated_entities = []
    entity_annotation = task_data.data["entity_annotation"]
    entity_list = task_data.data["entity_list"]
    for entity in entity_list:
        entity["classification"] = entity_annotation[entity["ID"]]
        annotated_entities.append(entity)

    with SpooledTemporaryFile(mode="w") as output:
        try:
            mimetype = task_data.data["mimetype"]
        except:
            mimetype = "application/json"
        save_entities(entities=annotated_entities, file_=output, mimetype=mimetype)

        if mimetype == "application/json":
            file_type = ".json"
        elif mimetype == "text/csv":
            file_type = ".csv"
        else:
            msg = f"Invalid mimetype {mimetype}!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)
        STORE.persist_task_result(
            db_id,
            output,
            "classified_entities" + file_type,
            "manual_classification_output",
            mimetype,
        )

    return "Manual classification successful."
