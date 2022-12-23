import json
from tempfile import SpooledTemporaryFile
from typing import Optional
from zipfile import ZipFile

import flask
from celery.utils.log import get_task_logger
from .backend.attribute import Attribute
from .backend.database import Database
from .backend.entity import EntityFactory

from . import CostumeLoader
from .backend.taxonomy import Taxonomy, TaxonomyType
from .schemas import InputParametersSchema, InputParameters, CostumeType
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)

costume_attrs = [
    Attribute.ortsbegebenheit,
    Attribute.dominanteFarbe,
    Attribute.stereotypRelevant,
    Attribute.dominanteFunktion,
    Attribute.dominanterZustand,
    Attribute.dominanteCharaktereigenschaft,
    Attribute.stereotyp,
    Attribute.geschlecht,
    Attribute.dominanterAlterseindruck,
    Attribute.genre,
    Attribute.rollenberuf,
    Attribute.dominantesAlter,
    Attribute.rollenrelevanz,
    Attribute.spielzeit,
    Attribute.tageszeit,
    Attribute.koerpermodifikation,
    Attribute.kostuemZeit,
    Attribute.familienstand,
    Attribute.charaktereigenschaft,
    Attribute.spielort,
    Attribute.spielortDetail,
    Attribute.alterseindruck,
    Attribute.alter,
]

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


@CELERY.task(name=f"{CostumeLoader.instance.identifier}.loading_task", bind=True)
def loading_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new loading task with db id '{db_id}'")
    app = flask.current_app

    #################
    # load costumes #
    #################

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    param_schema = InputParametersSchema()
    input_params: InputParameters = param_schema.loads(task_data.parameters)

    db = Database()
    db.open_with_params(
        host=input_params.db_host,
        user=input_params.db_user,
        password=input_params.db_password,
        database=input_params.db_database,
    )

    if input_params.costume_type == CostumeType.WITHOUT_BASE_ELEMENTS:
        entity_list = EntityFactory.create(costume_attrs, db)
    else:
        entity_list = EntityFactory.create(list(Attribute), db)

    entities = [entity.to_dataloader_dict() for entity in entity_list]

    entities_json = json.dumps(entities)

    with SpooledTemporaryFile(mode="w") as output:
        output.write(entities_json)
        STORE.persist_task_result(
            db_id, output, "entities.json", "entity/list", "application/json"
        )

    #############################
    # create attribute metadata #
    #############################

    attribute_metadata = []
    attrs_to_consider = (
        costume_attrs
        if input_params.costume_type == CostumeType.WITHOUT_BASE_ELEMENTS
        else list(Attribute)
    )

    for attr in attrs_to_consider:
        attr_type = (
            "integer" if attr in [Attribute.kostuemZeit, Attribute.alter] else "ref"
        )
        multiple = attr not in [
            Attribute.ortsbegebenheit,
            Attribute.dominanteFarbe,
            Attribute.stereotypRelevant,
            Attribute.dominanteFunktion,
            Attribute.dominanterZustand,
            Attribute.basiselement,
            Attribute.rollenberuf,
            Attribute.geschlecht,
            Attribute.dominanterAlterseindruck,
            Attribute.dominantesAlter,
            Attribute.rollenrelevanz,
            Attribute.kostuemZeit,
        ]
        tax_name = getattr(Attribute.get_taxonomy_type(attr), "value", None)
        ref_target = None if tax_name is None else "taxonomies.zip:" + tax_name + ".json"

        metadata = AttributeMetadata(
            attr.value,
            attr_type,
            attr.value,
            multiple=multiple,
            separator=";",
            ref_target=ref_target,
            extra={"taxonomy_name": tax_name},
        )

        attribute_metadata.append(metadata.to_dict())

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(attribute_metadata, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "attribute_metadata.json",
            "entity/attribute-metadata",
            "application/json",
        )

    ###################
    # load taxonomies #
    ###################

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
        "graphs",
        "application/zip",
    )

    return "result stored in output files"
