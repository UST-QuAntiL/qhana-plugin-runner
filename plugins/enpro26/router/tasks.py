from celery.utils.log import get_task_logger
import json
from zipfile import ZipFile
from tempfile import SpooledTemporaryFile
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from . import Router
# from .schemas import InputParametersSchema

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{Router.instance.identifier}.route_task", bind=True)
def route_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new routing task with db id '{db_id}'")
    task_data = ProcessingTask.get_by_id(id_=db_id)
    if not task_data:
        raise KeyError(f"Could not load task data with id {db_id}")

    # params = InputParametersSchema().loads(task_data.parameters)

    
    # entities_url = params["entities_url"]
    # entities_metadata_url = params["entities_metadata_url"]
    # taxonomies_url = params["taxonomies_zip_url"]
    # attributes = params["attributes"]
    # routing_options = params["routing_options"]
    # merging_point = params["merging_point"]
    # taxonomy_checkbox = params["taxonomy_checkbox"]

    # if routing_options.value == "Wu-Palmer":
    #     # Call the Wu-Palmer routing function (not implemented yet)
    #     TASK_LOGGER.info("Wu-Palmer routing option selected.")
    #     pass
    # elif routing_options.value == "One-Hot":
    #     # Call the One-Hot routing function (not implemented yet)
    #     TASK_LOGGER.info("One-Hot routing option selected.")
    #     pass
    # elif routing_options.value == "Numeric Mapping":
    #     # Call the Numeric Mapping routing function (not implemented yet)
    #     TASK_LOGGER.info("Numeric Mapping routing option selected.")
    #     pass

    return "Routing task completed successfully."