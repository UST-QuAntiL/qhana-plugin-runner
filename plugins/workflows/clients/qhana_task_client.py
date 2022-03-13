from __future__ import annotations

import json
import logging
from typing import List, TYPE_CHECKING

import requests

from ..datatypes.camunda_datatypes import ExternalTask
from ..datatypes.qhana_datatypes import QhanaResult, QhanaPlugin, QhanaInput, QhanaOutput
from ..util.result_store import ResultStore
from ..util.helper import endpoint_found, endpoint_found_simple

logger = logging.getLogger(__name__)

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
            result_store: ResultStore
    ):
        self.plugin_runner_endpoints = plugin_runner_endpoints
        self.plugins: List[QhanaPlugin] = []
        self.result_store = result_store
        self.processed: List[ExternalTask] = []
        self.get_plugins_from_endpoints()

    def create_qhana_plugin_instances(self, camunda_client: CamundaClient, external_tasks: List[ExternalTask]):
        """
        Console specific qhana plugin instance creator
        :param camunda_client: The camunda_client to be used
        :param external_tasks: External task to use for creating the qhana plugin instance
        :return:
        """
        for external_task in external_tasks:
            if external_task in self.processed:
                continue

            plugin_name = ""
            if "." in external_task.topic_name:
                plugin_name = external_task.topic_name.split('.', 1)[1]
            else:
                logger.warning("No plugin name found")

            plugin = self.resolve(plugin_name)
            if plugin:
                local_variables = camunda_client.get_task_local_variables(external_task)
                try:
                    parameters = self.collect_input(external_task, camunda_client, local_variables)
                except ValueError:
                    continue

                self.processed.append(external_task)
                camunda_client.qhana_listener.add_qhana_task(external_task, plugin, parameters)

    def complete_qhana_task(self, camunda_client: CamundaClient, qhana_results: List[QhanaResult]):
        """
        Submits the result for a corresponding external task to Camunda
        :param camunda_client: Client to be used
        :param qhana_results: Results from finished QHAna plugins
        :return:
        """
        self.qhana_results_store(qhana_results)
        for qhana_result in qhana_results:
            result = {"output":
                {"value":
                    [
                        {"name": output.name,
                         "contentType": output.content_type,
                         "dataType": output.data_type,
                         "href": output.href} for output in qhana_result.output_list
                    ]
                }
            }
            camunda_client.complete_task(qhana_result.qhana_task.external_task, result)

    def qhana_results_store(self, qhana_results: List[QhanaResult]):
        """
        Store result after a qhana plugin instance has finished
        :param qhana_results: The results to be stored
        :return:
        """
        for qhana_result in qhana_results:
            self.result_store.store_result(qhana_result)

    def get_plugins_from_endpoints(self):
        """
        Retrieves the hosted plugins from the specified QHAna endpoints
        :return:
        """
        for endpoint in self.plugin_runner_endpoints:
            response = requests.get(f"{endpoint}/plugins/")
            if endpoint_found(response):
                for plugin in response.json()["plugins"]:
                    response = requests.get(plugin["apiRoot"]).json()
                    href = response.get("entryPoint", {}).get("href", None)
                    if href:
                        process_endpoint = f"{endpoint[:-1]}{href}"
                    else:
                        process_endpoint = f'{plugin["apiRoot"]}/process/'
                    self.plugins.append(QhanaPlugin.deserialize(plugin, endpoint, process_endpoint))

    def resolve(self, plugin_name):
        """
        Retrieves the plugin from the provided plugin name
        :param plugin_name: Name of the plugin
        :return:
        """
        plugin = next((pl for pl in self.plugins if pl.name == plugin_name), None)
        if plugin is None:
            logger.warning(f"Could not find plugin {plugin_name} in plugin list")

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

    def collect_input(self, task: ExternalTask, camunda_client: CamundaClient, local_variables: dict):
        """
        TODO: Multistep plugins
        :param task: The task to use for input collection
        :param camunda_client: Client to be used
        :param local_variables: Variables which may contain input for the QHAna plugin
        :return:
        """
        # TODO: Move constants to config file
        qhana_input_prefix = "qinput"
        plugin_inputs = {}

        for key, item in local_variables.items():
            if key.startswith(qhana_input_prefix):
                input_parameter = key.split(".")[-1]
                output_name, select = list(item["value"].items())[0]
                retrieved_output = camunda_client.get_global_variable(output_name)

                if select == "plain":
                    plugin_inputs[input_parameter] = retrieved_output
                    continue

                if type(retrieved_output) == str and select != "plain":
                    retrieved_output = [json.loads(retrieved_output)]

                deserialized_outputs = [QhanaOutput.deserialize(output) for output in retrieved_output]
                mode = select.split(":")[0]
                mode_val = select.split(":")[1].strip()
                for output in deserialized_outputs:
                    if mode != "name" and mode != "dataType":
                        logger.warning("mode is not name or dataType")
                        camunda_client.external_task_bpmn_error(task, "qhana-mode-error",
                                                                "Input mode is not name or dataType!")
                        raise ValueError

                    if (mode == "name" and output.name == mode_val) or \
                            (mode == "dataType" and output.data_type == mode_val):
                        plugin_inputs[input_parameter] = output.href

        return plugin_inputs

    def get_plugin_inputs(self, plugin: QhanaPlugin):
        """
        Gets the list of inputs for a given plugin
        :param plugin: The plugin to get inputs for
        :return:
        """
        response = requests.get(f"{plugin.api_root}/")
        if endpoint_found(response):
            inputs = response.json()["entryPoint"]["dataInput"]
            result = []
            for input in inputs:
                result.append(QhanaInput.deserialize(input))
            return result
