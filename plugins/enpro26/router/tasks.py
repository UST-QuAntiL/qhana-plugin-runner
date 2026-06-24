import requests
import time
import traceback
from flask import url_for
from flask.globals import current_app
from celery.utils.log import get_task_logger
from urllib.parse import urljoin


from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.tasks import TASK_STATUS_CHANGED

from . import Router
from .schemas import InputParametersSchema, InputParameters

TASK_LOGGER = get_task_logger(__name__)
WU_PALMER_URL = "http://localhost:5005/plugins/wu-palmer@v0-2-1/process/"
SMM_URL = "http://localhost:5005/plugins/sym-max-mean@v0-1-2/process/"
TRANSFORMER_URL = "http://localhost:5005/plugins/sim-to-dist-transformers@v0-2-1/process/"
AGG_URL = "http://localhost:5005/plugins/distance-aggregator@v0-2-1/process/"
MDS_URL = "http://localhost:5005/plugins/mds@v0-2-1/process/"

# --- HELPER FUNCTIONS ---

def subscribe_to_plugin(task_result_url: str, webhook_url: str) -> str:
    """Subscribes the webhook to a target plugin's task result updates."""

    webhook_url = webhook_url.replace("localhost", "127.0.0.1")
    response = requests.get(task_result_url).json()
    subscription_link = next((link["href"] for link in response.get("links", []) if link["type"] == "subscribe"), None)
    if not subscription_link:
        raise ValueError("Target plugin does not support subscriptions!")
    requests.post(subscription_link, json={"command": "subscribe", "event": "status", "webhookHref": webhook_url}).raise_for_status()
    status_check = requests.get(task_result_url).json()
    return status_check.get("status", "PENDING")

def extract_output_url(outputs: list, data_type: str) -> str:
    """Finds the URL of a specific output from a plugin result based on its dataType."""
    for output in outputs:
        if output.get("dataType") == data_type:
            return output["href"]
    raise ValueError(f"Could not find output with dataType {data_type}")


# --- TASK FUNCTION ---

@CELERY.task(
    name=f"{Router.instance.identifier}.route_task", bind=True, max_retries=None)
def route_task(self, db_id: int) -> str:
    task_data = ProcessingTask.get_by_id(id_=db_id)
    TASK_LOGGER.info(f"Supervisor checking pipeline state for db id '{db_id}'")
    if not task_data:
        raise KeyError(f"Could not load task data with id {db_id}")
    
    current_step = task_data.data.get("current_step")

    if current_step == "done":
        return "Pipeline completed successfully!"

    if current_step == "error":
        error_msg = task_data.data.get("error_msg", "Unknown error in pipeline")
        raise ValueError(f"Pipeline Failed: {error_msg}. Check the log entries above for the exact line number and traceback.")

    if current_step is None:
        TASK_LOGGER.info(f"Starting new external wu palmer task with db id '{db_id}'")
        params: InputParameters = InputParametersSchema().loads(task_data.parameters)

        if params.routing_options.value == "Wu-Palmer":
            TASK_LOGGER.info("Starting Step 1: Wu-Palmer")
            payload = {
                "entitiesUrl": params.entities_url,
                "entitiesMetadataUrl": params.entities_metadata_url,
                "taxonomiesZipUrl": params.taxonomies_zip_url,
                "attributes": params.attributes,
                "root_is_part_of_hierarchy": str(params.root_is_part_of_hierarchy).lower()
            }

            response = requests.post(WU_PALMER_URL, data=payload, allow_redirects=False)
            if response.status_code != 303:
                raise ValueError(f"Wu-Palmer request failed with status {response.status_code}: {response.text}")

            task_url = urljoin(WU_PALMER_URL, response.headers["Location"])

            task_data.data["wu_palmer_url"] = task_url
            task_data.data["current_step"] = 1 #Wu-Palmer
            task_data.save(commit=True)

            status = subscribe_to_plugin(task_url, task_data.data["webhook_url"])

            if status in ["SUCCESS", "ERROR", "FAILURE"]:
                handle_webhook_task.delay(db_id=db_id, source_url=task_url)
            else:
                handle_webhook_task.apply_async(kwargs={"db_id": db_id, "source_url": task_url}, countdown=5)
    
    TASK_LOGGER.info("Pipeline still running. Supervisor sleeping for 5 seconds...")
    raise self.retry(countdown=5)

@CELERY.task(name=f"{Router.instance.identifier}.handle_webhook_task", bind=True)
def handle_webhook_task(self, db_id: int, source_url: str):
    """Triggered whenever a sub-plugin finishes its task."""

    task_data = ProcessingTask.get_by_id(id_=db_id)
    current_step = task_data.data.get("current_step")

    if current_step in ["done", "error"]: return

    is_wu_palmer = (source_url == task_data.data.get("wu_palmer_url"))
    is_sym_max_mean = (source_url == task_data.data.get("sym_max_mean_url"))
    is_transformers = (source_url == task_data.data.get("transformers_url"))
    is_aggregators = (source_url == task_data.data.get("aggregators_url"))
    is_mds = (source_url == task_data.data.get("mds_url"))

    # Ignore Duplicates
    if is_wu_palmer and current_step != 1: return 
    if is_sym_max_mean and current_step != 2: return 
    if is_transformers and current_step != 3: return 
    if is_aggregators and current_step != 4: return 
    if is_mds and current_step != 5: return 

    if not any([is_wu_palmer, is_sym_max_mean, is_transformers, is_aggregators, is_mds]):
        TASK_LOGGER.warning(f"Unrecognized webhook source ignored: {source_url}")
        return "Unrecognized webhook"

    try:     
        sub_task_result = requests.get(source_url).json()
        status = sub_task_result.get("status","PENDING")

        if status == "PENDING":
            handle_webhook_task.apply_async(kwargs={"db_id": db_id, "source_url": source_url}, countdown=5)
            return
        
        elif status in ["ERROR", "FAILURE"]:
            raise ValueError(f"Sub-plugin failed during {current_step}")
        
        elif status == "SUCCESS":
            # If it succeeded, check which step just finished and start the NEXT step
            outputs = sub_task_result.get("outputs", [])
            params: InputParameters = InputParametersSchema().loads(task_data.parameters)
            my_webhook_url = task_data.data["webhook_url"]
            
            if current_step == 1 and is_wu_palmer:
                task_data.data["current_step"] = "transition_1_to_2"
                task_data.save(commit=True)
                
                TASK_LOGGER.info("Step 1 done. Waiting 2 seconds to protect server pool...")
                time.sleep(2) # Safe to sleep now, duplicates are blocked!
                
                element_sims_url = extract_output_url(outputs, "custom/element-similarities")
                
                payload = {
                    "entitiesUrl": params.entities_url,
                    "elementSimilaritiesUrl": element_sims_url,
                    "attributes": params.attributes
                }
                response = requests.post(SMM_URL, data=payload, allow_redirects=False)
                if response.status_code != 303:
                    raise ValueError(f"Sym-Max-Mean failed with status {response.status_code}: {response.text}")

                task_url = urljoin(SMM_URL, response.headers["Location"])
                
                # Save State
                task_data.data["sym_max_mean_url"] = task_url
                task_data.data["current_step"] = 2
                task_data.add_task_log_entry("Started Sym Max Mean.")
                task_data.save(commit=True)

                status2 = subscribe_to_plugin(task_url, my_webhook_url)
                if status2 in ["SUCCESS", "ERROR", "FAILURE"]:
                    handle_webhook_task.delay(db_id=db_id, source_url=task_url)
                else:
                    handle_webhook_task.apply_async(kwargs={"db_id": db_id, "source_url": task_url}, countdown=5)

            elif current_step == 2 and is_sym_max_mean:
                task_data.data["current_step"] = "transition_2_to_3"
                task_data.save(commit=True)

                TASK_LOGGER.info("Step 2 done. Waiting 2 seconds to protect server pool...")
                time.sleep(2)

                attr_sims_url = extract_output_url(outputs, "custom/attribute-similarities")
                payload = {
                    "attributeSimilaritiesUrl": attr_sims_url,
                    "attributes": params.attributes,
                    "transformer": params.transformer.name
                }

                response = requests.post(TRANSFORMER_URL, data=payload, allow_redirects=False)
                if response.status_code != 303:
                    raise ValueError(f"Transformers failed with status {response.status_code}: {response.text}")
                
                task_url = urljoin(TRANSFORMER_URL, response.headers["Location"])
                
                task_data.data["transformers_url"] = task_url
                task_data.data["current_step"] = 3
                task_data.add_task_log_entry("Started Transformers.")
                task_data.save(commit=True)

                status2 = subscribe_to_plugin(task_url, my_webhook_url)

                if status2 in ["SUCCESS", "ERROR", "FAILURE"]:
                    handle_webhook_task.delay(db_id=db_id, source_url=task_url)
                else:
                    handle_webhook_task.apply_async(kwargs={"db_id": db_id, "source_url": task_url}, countdown=5)

            elif current_step == 3 and is_transformers:
                task_data.data["current_step"] = "transition_3_to_4"
                task_data.save(commit=True)

                TASK_LOGGER.info("Step 3 done. Waiting 2 seconds to protect server pool...")
                time.sleep(2)

                attr_dists_url = extract_output_url(outputs, "custom/attribute-distances")
                
                payload = {
                    "attributeDistancesUrl": attr_dists_url,
                    "aggregator": params.aggregator.name, 
                    "missingDataHandling": params.missing_data_handling.name 
                }
                
                response = requests.post(AGG_URL, data=payload, allow_redirects=False)
                if response.status_code != 303:
                    raise ValueError(f"Aggregator failed with status {response.status_code}: {response.text}")

                task_url = urljoin(AGG_URL, response.headers["Location"])
                
                task_data.data["aggregators_url"] = task_url
                task_data.data["current_step"] = 4
                task_data.add_task_log_entry("Started Aggregator.")
                task_data.save(commit=True)

                status2 = subscribe_to_plugin(task_url, my_webhook_url)
                if status2 in ["SUCCESS", "ERROR", "FAILURE"]:
                    handle_webhook_task.delay(db_id=db_id, source_url=task_url)
                else:
                    handle_webhook_task.apply_async(kwargs={"db_id": db_id, "source_url": task_url}, countdown=5)
            
            elif current_step == 4 and is_aggregators:
                task_data.data["current_step"] = "transition_4_to_5"
                task_data.save(commit=True)

                TASK_LOGGER.info("Step 4 done. Waiting 2 seconds to protect server pool...")
                time.sleep(2)

                entity_dists_url = extract_output_url(outputs, "custom/entity-distances")
                
                payload = {
                    "entityDistancesUrl": entity_dists_url,
                    "dimensions": params.dimensions, 
                    "metric": params.metric.name,
                    "nInit": params.n_init,
                    "maxIter": params.max_iter,
                }
                
                response = requests.post(MDS_URL, data=payload, allow_redirects=False)
                if response.status_code != 303:
                    raise ValueError(f"MDS failed with status {response.status_code}: {response.text}")

                task_url = urljoin(MDS_URL, response.headers["Location"])
                
                task_data.data["mds_url"] = task_url
                task_data.data["current_step"] = 5
                task_data.add_task_log_entry("Started Aggregator.")
                task_data.save(commit=True)

                status2 = subscribe_to_plugin(task_url, my_webhook_url)
                if status2 in ["SUCCESS", "ERROR", "FAILURE"]:
                    handle_webhook_task.delay(db_id=db_id, source_url=task_url)
                else:
                    handle_webhook_task.apply_async(kwargs={"db_id": db_id, "source_url": task_url}, countdown=5)
    
            elif current_step == 5 and is_mds:
                task_data.data["current_step"] = "transition_5_to_done"
                task_data.save(commit=True)
                TASK_LOGGER.info("Step 5 done. Pipeline Complete!")
                
                final_dists_url = extract_output_url(outputs, "entity/vector")
                
                # Persist the final output back to our own router task so it shows in the UI
                STORE.persist_task_result(
                    db_id, 
                    open_url(final_dists_url).content, 
                    "final_entity_distances.json", 
                    "entity/vector", 
                    "application/json"
                )
                
                # Mark the whole pipeline as SUCCESSFUL
                task_data.data["current_step"] = "done"
                task_data.save(commit=True)

    except Exception as e:
        tb_str = traceback.format_exc()
        task_data.data["current_step"] = "error"
        task_data.data["error_msg"] = f"Webhook Task Error: Pipeline crashed:\nError Message:\n{str(e)}\n\nTraceback:\n{tb_str}"
        task_data.save(commit=True)