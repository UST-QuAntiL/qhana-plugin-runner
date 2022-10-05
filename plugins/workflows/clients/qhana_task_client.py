from __future__ import annotations

import http.client
import json
import re
from typing import TYPE_CHECKING, List, Optional, Sequence

import requests
from celery.utils.log import get_task_logger

from .. import Workflows
from ..datatypes.camunda_datatypes import ExternalTask
from ..datatypes.qhana_datatypes import (
    QhanaInput,
    QhanaOutput,
    QhanaPlugin,
    QhanaTask,
)
from ..util.helper import endpoint_found_simple, request_json

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)

if TYPE_CHECKING:
    from camunda_client import CamundaClient


class QhanaTaskClient:
    """
    Gets all available plugins and creates qhana plugin instances from camunda external tasks.
    Completes camunda external tasks and forwards results to the result store
    """

    def __init__(
        self,
        plugin_runner_endpoints: List[str],
    ):
        self.plugin_runner_endpoints = plugin_runner_endpoints
        self.plugins: List[QhanaPlugin] = []
        self.get_plugins_from_endpoints()

    def create_qhana_plugin_instance(
        self, camunda_client: CamundaClient, external_task: ExternalTask
    ):
        """
        Console specific qhana plugin instance creator
        :param camunda_client: The camunda_client to be used
        :param external_task: External task to use for creating the qhana plugin instance
        :return: Qhana task
        """

        plugin_name = ""
        if "." in external_task.topic_name:
            plugin_name = external_task.topic_name.split(".", 1)[1]
        else:
            TASK_LOGGER.warning(
                f"No plugin extracted from external task topic {external_task}"
            )

        # Check if requested plugin exists
        plugin = self.resolve(plugin_name)

        if plugin:
            local_variables = camunda_client.get_task_local_variables(external_task)
            try:
                parameters = self.collect_input(
                    external_task, camunda_client, local_variables
                )
            except ValueError:
                return

            qhana_task: QhanaTask = self.invoke_qhana_task(
                camunda_client, external_task, plugin, parameters
            )

            if qhana_task is None:
                raise Exception

            return qhana_task

    def invoke_qhana_task(
        self,
        camunda_client: CamundaClient,
        external_task: ExternalTask,
        plugin: QhanaPlugin,
        params,
    ):
        """
        Starts a QHAna task
        :param external_task: The external task that resulted in the QHAna task
        :param camunda_client: Camunda client
        :param plugin: The plugin to run
        :param params: Parameters for running the plugin
        """
        TASK_LOGGER.info(f"Call plugin process endpoint '{plugin.process_endpoint}'")
        response = requests.post(plugin.process_endpoint, data=params)
        url = response.url

        if response.status_code == http.client.OK:
            response = response.json()
            db_id = re.search("/\d+/", url).group(0)[
                1:-1
            ]  # FIXME use flask to handle url parsing and pass it via g or via explicit parameters!
            TASK_LOGGER.info(f"Started QHAna plugin {plugin.identifier}")

            return QhanaTask.deserialize(response, db_id, external_task, plugin)
        elif response.status_code == http.client.UNPROCESSABLE_ENTITY:
            TASK_LOGGER.warning(f"Received unprocessable entity on endpoint {url}")

            camunda_client.external_task_bpmn_error(
                task=external_task,
                error_code="qhana-unprocessable-entity-error",
                error_message="Plugin invocation received unprocessable entities and could not proceed.",
            )

    def complete_qhana_task(
        self,
        camunda_client: CamundaClient,
        outputs: Sequence[QhanaOutput],
        external_task: ExternalTask,
    ):
        """
        Submits the result for a corresponding external task to Camunda
        :param camunda_client: Client to be used
        :param qhana_result: Result from finished QHAna plugin
        """
        result = {
            "output": {
                "value": [
                    {
                        "name": output.name,
                        "contentType": output.content_type,
                        "dataType": output.data_type,
                        "href": output.href,
                    }
                    for output in outputs
                ]
            }
        }
        camunda_client.complete_task(external_task, result)

    def get_plugins_from_endpoints(self):
        """
        Retrieves the hosted plugins from the specified QHAna endpoints
        """
        for endpoint in self.plugin_runner_endpoints:
            response = request_json(f"{endpoint}/plugins/")
            for plugin in response["plugins"]:
                try:
                    response = request_json(plugin["apiRoot"])
                    href: Optional[str] = response.get("entryPoint", {}).get("href", None)
                    if href:
                        if href.startswith(("http://", "https://")):
                            process_endpoint = href
                        else:
                            process_endpoint = (
                                f"{endpoint.rstrip('/')}/{href.lstrip('/')}"
                            )
                    else:
                        process_endpoint = f'{plugin["apiRoot"]}/process/'

                    self.plugins.append(
                        QhanaPlugin.deserialize(plugin, endpoint, process_endpoint)
                    )
                except Exception:
                    TASK_LOGGER.info(f"Failed to load plugin {plugin}")

    def resolve(self, plugin_name):
        """
        Retrieves the plugin from the provided plugin name
        :param plugin_name: Name of the plugin
        :return: Plugin
        """
        plugin = next((pl for pl in self.plugins if pl.name == plugin_name), None)

        if plugin is None:
            TASK_LOGGER.warning(f"Could not find plugin {plugin_name} in plugin list")

        return plugin

    def get_micro_frontend(self, plugin: QhanaPlugin):
        """
        Retrieves the micro frontend of a plugin
        :param plugin: Plugin for retrieving the micro frontend
        :return:
        """
        response = requests.get(f"{plugin.api_root}/ui/")
        if endpoint_found_simple(response):
            return response.text

    def collect_input(
        self, task: ExternalTask, camunda_client: CamundaClient, local_variables: dict
    ):
        """
        TODO: Multistep plugins
        :param task: The task to use for input collection
        :param camunda_client: Client to be used
        :param local_variables: Variables which may contain input for the QHAna plugin
        :return:
        """
        prefix = config["qhana_input"]["prefix"]
        workflow_result_prefix = config["workflow_out"]["prefix"]
        input_mode_text = config["qhana_input"]["mode_text"]
        input_mode_filename = config["qhana_input"]["mode_filename"]
        input_mode_datatype = config["qhana_input"]["mode_datatype"]
        plugin_inputs = {}

        for key, item in local_variables.items():
            # Cut off result prefix if found
            if key.startswith(workflow_result_prefix):
                key = key[len(workflow_result_prefix) + 1 : len(key)]

            # Check if variable is a qhana input
            if key.startswith(prefix):
                input_parameter = key.split(".")[-1]
                output_name, select = list(item["value"].items())[0]

                # Retrieves the contents of an output that is used as input
                retrieved_output = camunda_client.get_global_variable(output_name, task)

                # Treat output as plain text
                if select == input_mode_text:
                    plugin_inputs[input_parameter] = retrieved_output
                    continue

                # If output type is not plain text, e.g., enum or choice
                if type(retrieved_output) == str and select != input_mode_text:
                    retrieved_output = [json.loads(retrieved_output)]

                deserialized_outputs = [
                    QhanaOutput.deserialize(output) for output in retrieved_output
                ]
                mode = select.split(":")[0]
                mode_val = select.split(":")[1].strip()

                for output in deserialized_outputs:
                    if mode != input_mode_filename and mode != input_mode_datatype:
                        TASK_LOGGER.warning("mode is not name or dataType")

                        camunda_client.external_task_bpmn_error(
                            task=task,
                            error_code="qhana-mode-error",
                            error_message="Input mode is not name or dataType!",
                        )

                        raise ValueError

                    if (mode == "name" and output.name == mode_val) or (
                        mode == "dataType" and output.data_type == mode_val
                    ):
                        plugin_inputs[input_parameter] = output.href

        return plugin_inputs

    def get_plugin_inputs(self, plugin: QhanaPlugin):
        """
        Gets the list of inputs for a given plugin
        :param plugin: The plugin to get inputs for
        :return:
        """
        response = request_json(f"{plugin.api_root}/")
        inputs = response["entryPoint"]["dataInput"]
        result = []

        for input in inputs:
            result.append(QhanaInput.deserialize(input))

        return result
