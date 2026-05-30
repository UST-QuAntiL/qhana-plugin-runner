import json
from zipfile import ZipFile
from tempfile import SpooledTemporaryFile
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url

from . import TaxonomyRouter
from .schemas import InputParametersSchema

TASK_LOGGER = get_task_logger(__name__)

@CELERY.task(
    name=f"{TaxonomyRouter.instance.identifier}.route_taxonomies_task", bind=True)
def route_taxonomies_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new routing task with db id '{db_id}'")
    task_data = ProcessingTask.get_by_id(id_=db_id)
    if not task_data:
        raise KeyError(f"Could not load task data with id {db_id}")

    params = InputParametersSchema().loads(task_data.parameters)
    taxonomies_url = params["taxonomies_zip_url"]
    entities_url = params["entities_url"]

    # 1. Pass-through the Entity List
    entity_response = open_url(entities_url)
    entity_response.raise_for_status()
    
    with SpooledTemporaryFile(mode="w+b") as tmp_entity_file:
        tmp_entity_file.write(entity_response.content)
        tmp_entity_file.seek(0)
        STORE.persist_task_result(
            db_id, tmp_entity_file, "routed_entities.csv", "entity/list", "text/csv"
        )

    # 2. Process and split the Taxonomies
    tmp_tree_zip = SpooledTemporaryFile(mode="wb")
    tmp_non_tree_zip = SpooledTemporaryFile(mode="wb")
    
    with ZipFile(tmp_tree_zip, "w") as tree_zip, \
         ZipFile(tmp_non_tree_zip, "w") as non_tree_zip:

        # get_files_from_zip_url safely handles the download and extraction
        for zipped_file, file_name in get_files_from_zip_url(taxonomies_url, mode="t"):
            if not file_name.endswith('.json'):
                continue
            
            tax_data = json.load(zipped_file)
            
            if is_tree_taxonomy(tax_data):
                tree_zip.writestr(file_name, json.dumps(tax_data))
            else:
                non_tree_zip.writestr(file_name, json.dumps(tax_data))

    STORE.persist_task_result(
        db_id, tmp_tree_zip, "tree_taxonomies.zip", "graph/taxonomy", "application/zip"
    )
    STORE.persist_task_result(
        db_id, tmp_non_tree_zip, "non_tree_taxonomies.zip", "graph/taxonomy", "application/zip"
    )
    
    return "Entities routed and taxonomies successfully separated."


def is_tree_taxonomy(taxonomy_data: dict) -> bool:
    """
    Determines if a taxonomy is a true hierarchical tree.
    A flat list taxonomy (depth 1) will have all relations originating from a single root node.
    A true tree will have multiple nodes acting as sources in the relations.
    """
    relations = taxonomy_data.get("relations", [])
    
    if not relations:
        return False 
        
    unique_sources = set(rel.get("source") for rel in relations if "source" in rel)
    return len(unique_sources) > 1