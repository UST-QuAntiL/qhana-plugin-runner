import requests
import traceback
import time
from celery.utils.log import get_task_logger
from urllib.parse import urljoin

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import Router
from .schemas import InputParametersSchema, InputParameters

TASK_LOGGER = get_task_logger(__name__)
WU_PALMER_URL = "http://localhost:5005/plugins/wu-palmer@v0-2-1/process/"
SMM_URL = "http://localhost:5005/plugins/sym-max-mean@v0-1-2/process/"
TRANSFORMER_URL = "http://localhost:5005/plugins/sim-to-dist-transformers@v0-2-1/process/"
AGG_URL = "http://localhost:5005/plugins/distance-aggregator@v0-2-1/process/"
MDS_URL = "http://localhost:5005/plugins/mds@v0-2-1/process/"

# --- HELPER FUNCTIONS ---

def subscribe_to_plugin(task_result_url: str, webhook_url: str):
    """Subscribes the webhook to a target plugin's task result updates."""

    webhook_url = webhook_url.replace("localhost", "127.0.0.1")
    response = requests.get(task_result_url).json()
    subscription_link = next((link["href"] for link in response.get("links", []) if link["type"] == "subscribe"), None)
    if not subscription_link:
        raise ValueError("Target plugin does not support subscriptions!")
    requests.post(subscription_link, json={"command": "subscribe", "event": "status", "webhookHref": webhook_url}).raise_for_status()


def extract_output_url(outputs: list, data_type: str) -> str:
    """Finds the URL of a specific output from a plugin result based on its dataType."""
    for output in outputs:
        if output.get("dataType") == data_type:
            return output["href"]
    raise ValueError(f"Could not find output with dataType {data_type}")


# --- Initial Routing Task Launcher ---
@CELERY.task(
    name=f"{Router.instance.identifier}.route_task", bind=True)
def route_task(self, db_id: int) -> str:
    task_data = ProcessingTask.get_by_id(id_=db_id)
    TASK_LOGGER.info(f"Starting routing task for db id '{db_id}'")
    if not task_data:
        raise KeyError(f"Could not load task data with id {db_id}")
    params: InputParameters = InputParametersSchema().loads(task_data.parameters)
    
    TASK_LOGGER.info("Starting Step 1: Wu-Palmer")
    payload = {
        "entitiesUrl": params.entities_url,
        "entitiesMetadataUrl": params.entities_metadata_url,
        "taxonomiesZipUrl": params.taxonomies_zip_url,
        "attributes": params.attributes,
        "root_is_part_of_hierarchy": str(params.root_is_part_of_hierarchy).lower()
    }

    response = requests.post(WU_PALMER_URL, data=payload, allow_redirects=False)
    task_url = urljoin(WU_PALMER_URL, response.headers["Location"])
    
    task_data.data["wu_palmer_url"] = task_url
    task_data.add_task_log_entry("Started Wu-Palmer.")
    task_data.save(commit=True)

    subscribe_to_plugin(task_url, task_data.data["webhook_url"])

# --- Webhook task handles task results ---
@CELERY.task(name=f"{Router.instance.identifier}.handle_webhook_task", bind=True)
def handle_webhook_task(self, db_id: int, source_url: str):
    task_data = ProcessingTask.get_by_id(db_id)
    
    # Identify which plugin triggered the webhook
    is_wp = (source_url == task_data.data.get("wu_palmer_url"))
    is_smm = (source_url == task_data.data.get("sym_max_mean_url"))
    is_trans = (source_url == task_data.data.get("transformers_url"))
    is_agg = (source_url == task_data.data.get("aggregators_url"))
    is_mds = (source_url == task_data.data.get("mds_url"))

    sub_task_result = requests.get(source_url).json()
    status = sub_task_result.get("status", "PENDING")

    if status == "SUCCESS":
        # Launch the dedicated Celery task for whichever step just finished!
        if is_wp: process_step_2_smm.delay(db_id, source_url)
        elif is_smm: process_step_3_transformers.delay(db_id, source_url)
        elif is_trans: process_step_4_aggregator.delay(db_id, source_url)
        elif is_agg: process_step_5_mds.delay(db_id, source_url)
        elif is_mds: finalize_pipeline.delay(db_id, source_url)


# --- Sym-Max-Mean Task ---
@CELERY.task(name=f"{Router.instance.identifier}.process_step_2_smm", bind=True, max_retries=10)
def process_step_2_smm(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 2: SymMaxMean")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        # 1. Download & Persist WP result
        outputs = requests.get(source_url).json().get("outputs", [])
        sims_url = extract_output_url(outputs, "custom/element-similarities")
        STORE.persist_task_result(
            db_id, open_url(sims_url).content, "wp_similarities.zip", "custom/element-similarities", "application/zip"
        )

        # 2. Launch SMM
        params: InputParameters = InputParametersSchema().loads(task_data.parameters)
        payload = {
            "entitiesUrl": params.entities_url,
            "elementSimilaritiesUrl": sims_url,
            "attributes": params.attributes
        }
        
        response = requests.post(SMM_URL, data=payload, allow_redirects=False)
        task_url = urljoin(SMM_URL, response.headers["Location"])
        
        task_data.data["sym_max_mean_url"] = task_url
        task_data.add_task_log_entry("Started Sym Max Mean.")
        task_data.save(commit=True)
        subscribe_to_plugin(task_url, task_data.data["webhook_url"])

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        # If Werkzeug connection drops, gracefully retry this step in 3 seconds!
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Step 1 to 2:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)


# --- TRANSFORMER TASK ---
@CELERY.task(name=f"{Router.instance.identifier}.process_step_3_transformers", bind=True, max_retries=10)
def process_step_3_transformers(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 3: Transformer")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        # 1. Persist SMM result
        outputs = requests.get(source_url).json().get("outputs", [])
        attr_sims_url = extract_output_url(outputs, "custom/attribute-similarities")
        STORE.persist_task_result(
            db_id, open_url(attr_sims_url).content, "smm_similarities.zip", "custom/attribute-similarities", "application/zip"
        )

        # 2. Launch Transformers
        params = InputParametersSchema().loads(task_data.parameters)
        payload = {
            "attributeSimilaritiesUrl": attr_sims_url,
            "attributes": params.attributes,
            "transformer": params.transformer.name
        }
        
        response = requests.post(TRANSFORMER_URL, data=payload, allow_redirects=False)
        task_url = urljoin(TRANSFORMER_URL, response.headers["Location"])
        
        task_data.data["transformers_url"] = task_url
        task_data.add_task_log_entry("Started Transformers.")
        task_data.save(commit=True)
        subscribe_to_plugin(task_url, task_data.data["webhook_url"])

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Step 2 to 3:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)

# --- AGGREGATOR TASK ---
@CELERY.task(name=f"{Router.instance.identifier}.process_step_4_aggregator", bind=True, max_retries=10)
def process_step_4_aggregator(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 4: Aggregator")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        attr_dists_url = extract_output_url(outputs, "custom/attribute-distances")
        STORE.persist_task_result(
            db_id, open_url(attr_dists_url).content, "transformer_distances.zip", "custom/attribute-distances", "application/zip"
        )

        params = InputParametersSchema().loads(task_data.parameters)
        payload = {
            "attributeDistancesUrl": attr_dists_url,
            "aggregator": params.aggregator.name, 
            "missingDataHandling": params.missing_data_handling.name 
        }
        
        response = requests.post(AGG_URL, data=payload, allow_redirects=False)
        task_url = urljoin(AGG_URL, response.headers["Location"])
        
        task_data.data["aggregators_url"] = task_url
        task_data.add_task_log_entry("Started Aggregator.")
        task_data.save(commit=True)
        subscribe_to_plugin(task_url, task_data.data["webhook_url"])

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Step 3 to 4:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)

# --- MDS TASK ---
@CELERY.task(name=f"{Router.instance.identifier}.process_step_5_mds", bind=True, max_retries=10)
def process_step_5_mds(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 5: MDS")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        entity_dists_url = extract_output_url(outputs, "custom/entity-distances")
        STORE.persist_task_result(
            db_id, open_url(entity_dists_url).content, "aggregator_distances.json", "custom/entity-distances", "application/json"
        )

        params = InputParametersSchema().loads(task_data.parameters)
        payload = {
            "entityDistancesUrl": entity_dists_url,
            "dimensions": params.dimensions, 
            "metric": params.metric.name,
            "nInit": params.n_init,
            "maxIter": params.max_iter,
        }
        
        response = requests.post(MDS_URL, data=payload, allow_redirects=False)
        task_url = urljoin(MDS_URL, response.headers["Location"])
        
        task_data.data["mds_url"] = task_url
        task_data.add_task_log_entry("Started MDS.")
        task_data.save(commit=True)
        subscribe_to_plugin(task_url, task_data.data["webhook_url"])

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Step 4 to 5:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)

# --- END OF PIPELINE ---
@CELERY.task(name=f"{Router.instance.identifier}.finalize_pipeline", bind=True, max_retries=10)
def finalize_pipeline(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 6: Finishing the Pipeline")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        final_dists_url = extract_output_url(outputs, "entity/vector")
        STORE.persist_task_result(
            db_id, open_url(final_dists_url).content, "mds_final_vectors.json", "entity/vector", "application/json"
        )
        
        # MANUALLY TRIGGER OFFICIAL COMPLETION - Fixes infinite pending bug!
        save_task_result.delay("Pipeline Completed Successfully!", db_id)

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Final Step:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)
