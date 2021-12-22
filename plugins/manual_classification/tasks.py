from tempfile import SpooledTemporaryFile

from typing import Optional
from json import dumps, loads
import mimetypes

from celery.utils.log import get_task_logger

from plugins.manual_classification import ManualClassification
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ResponseLike,
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

    input_entities = {}
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

        # extract relevant information to create selection fields for manual classification
        entity_classification_data = {}
        for entity in input_entities:
            # keys are entity id's and values are lists of annotated classes
            entity_classification_data[entity["ID"]] = []

    # store in data of task_data
    task_data.data["entity_data"] = entity_classification_data
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

    entity_data = task_data.data["entity_data"]
    for id in entity_data.keys():
        if params.get(id):
            tmp = set(entity_data[id])
            tmp.add(class_identifier)
            entity_data[id] = list(tmp)

    # store in data of task_data
    task_data.data["entity_data"] = entity_data
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

    try:
        input_file_url = task_data.data["input_file_url"]
    except:
        msg = f"No input_file_url in db! Somehow got lost during the plugin execution"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # TODO: somehow return output files with annotated data...
    annotated_entities = []
    for id in task_data.data["entity_data"].keys():
        annotated_entities.append(
            {"ID": id, "annotated_classes": task_data.data["entity_data"][id]}
        )

    with SpooledTemporaryFile(mode="w") as output:
        try:
            mimetype = task_data.data["mimetype"]
        except:
            mimetype = "application/json"
        save_entities(entities=annotated_entities, file_=output, mimetype=mimetype)

        if mimetype == "application/json":
            file_type = ".json"
        else:
            file_type = ".csv"
        STORE.persist_task_result(
            db_id,
            output,
            "filtered_entities" + file_type,
            "entity_filter_output",
            mimetype,
        )

    return "Adding new class successful."
