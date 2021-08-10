import json
from tempfile import SpooledTemporaryFile
from typing import Optional
from zipfile import ZipFile

import flask
from celery.utils.log import get_task_logger
from plugins.costume_loader_pkg.backend.attribute import Attribute
from plugins.costume_loader_pkg.backend.database import Database
from plugins.costume_loader_pkg.backend.entity import EntityFactory

from plugins.costume_loader_pkg import CostumeLoader
from plugins.costume_loader_pkg.backend.taxonomy import Taxonomy, TaxonomyType
from plugins.costume_loader_pkg.schemas import (
    InputParameters,
    InputParametersSchema,
    MuseEntitySchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)

base_element_attrs = [
    Attribute.basiselement,
    Attribute.design,
    Attribute.form,
    Attribute.trageweise,
    Attribute.zustand,
    Attribute.funktion,
    Attribute.material,
    Attribute.materialeindruck,
    Attribute.farbe,
    Attribute.farbeindruck,
]


@CELERY.task(name=f"{CostumeLoader.instance.identifier}.costume_loading_task", bind=True)
def costume_loading_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new costume loading task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    param_schema = InputParametersSchema()
    input_params: InputParameters = param_schema.loads(task_data.parameters)

    attributes = input_params.attributes

    app = flask.current_app

    db = Database()
    db.open_with_params(
        host=app.config.get("COSTUME_LOADER_DB_HOST"),
        user=app.config.get("COSTUME_LOADER_DB_USER"),
        password=app.config.get("COSTUME_LOADER_DB_PASSWORD"),
        database=app.config.get("COSTUME_LOADER_DB_DATABASE"),
    )

    entity_schema = MuseEntitySchema()

    entities = [
        entity_schema.dump(entity) for entity in EntityFactory.create(attributes, db)
    ]

    entities_json = json.dumps(entities)

    with SpooledTemporaryFile(mode="w") as output:
        output.write(entities_json)
        STORE.persist_task_result(
            db_id, output, "entities.json", "costume-loader-output", "application/json"
        )

    return "result: " + entities_json


@CELERY.task(name=f"{CostumeLoader.instance.identifier}.taxonomy_loading_task", bind=True)
def taxonomy_loading_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new taxnomoy loading task with db id '{db_id}'")
    app = flask.current_app

    db = Database()
    db.open_with_params(
        host=app.config.get("COSTUME_LOADER_DB_HOST"),
        user=app.config.get("COSTUME_LOADER_DB_USER"),
        password=app.config.get("COSTUME_LOADER_DB_PASSWORD"),
        database=app.config.get("COSTUME_LOADER_DB_DATABASE"),
    )

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for tax_type in TaxonomyType:
        if tax_type.get_database_table_name(tax_type) is not None:
            taxonomy = Taxonomy.create_from_db(tax_type, db)
            taxonomy_json = json.dumps(taxonomy)

            zip_file.writestr(tax_type.value + ".json", taxonomy_json)

    zip_file.close()

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        "taxonomies.zip",
        "taxonomy-output",
        "application/zip",
    )

    return "result stored in zip file"
