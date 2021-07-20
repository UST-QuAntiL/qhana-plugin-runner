import json
from typing import Optional

import flask
from celery.utils.log import get_task_logger
from qhana.backend.database import Database
from qhana.backend.entityService import EntityService

from plugins.costume_loader_pkg import CostumeLoader
from plugins.costume_loader_pkg.schemas import (
    InputParameters,
    InputParametersSchema,
    MuseEntitySchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{CostumeLoader.instance.identifier}.costume_loading_task", bind=True)
def costume_loading_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    param_schema = InputParametersSchema()
    input_params: InputParameters = param_schema.loads(task_data.parameters)

    es = EntityService()

    plan = [
        input_params.aggregator,
        input_params.transformer,
    ]
    plan.extend(
        [
            (
                input_params.attributes[i],
                input_params.element_comparers[i],
                input_params.attribute_comparers[i],
                input_params.empty_attribute_actions[i],
                input_params.filters[i],
            )
            for i in range(len(input_params.attributes))
        ]
    )

    es.add_plan(plan)

    app = flask.current_app

    db = Database()
    db.open_with_params(
        host=app.config.get("COSTUME_LOADER_DB_HOST"),
        user=app.config.get("COSTUME_LOADER_DB_USER"),
        password=app.config.get("COSTUME_LOADER_DB_PASSWORD"),
        database=app.config.get("COSTUME_LOADER_DB_DATABASE"),
    )

    es.create_subset(input_params.subset, db)

    entity_schema = MuseEntitySchema()

    entities = [entity_schema.dump(entity) for entity in es.allEntities]

    return json.dumps(entities)
