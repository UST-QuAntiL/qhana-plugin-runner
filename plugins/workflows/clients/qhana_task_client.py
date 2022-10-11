from __future__ import annotations

import http.client
import json
from typing import TYPE_CHECKING, List, Optional, Sequence

import requests
from celery.utils.log import get_task_logger
from requests import HTTPError

from .. import Workflows
from ..datatypes.qhana_datatypes import QhanaInput, QhanaOutput, QhanaPlugin

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)

if TYPE_CHECKING:
    from .camunda_client import CamundaClient


class ParameterParsingError(ValueError):
    def __init__(self, *args: object, **kwargs) -> None:
        self.parameter = kwargs.pop("parameter", "UNKNOWN")
        self.mode = kwargs.pop("mode", "UNKNOWN")
        super().__init__(*args, **kwargs)


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
        self._plugins: Optional[List[QhanaPlugin]] = None
        self.timeout: int = config.get("request_timeout", 5 * 60)

    def call_qhana_plugin(self, plugin: QhanaPlugin, params):
        response = requests.post(
            plugin.process_endpoint, data=params, timeout=self.timeout
        )

        response.raise_for_status()

        if response.status_code != http.client.OK:
            raise HTTPError("Unknown status code.", response=response)

        return response.url

    def call_plugin_step(self, href: str, params):
        response = requests.post(href, data=params, timeout=self.timeout)
        response.raise_for_status()

    def _get_plugins_from_endpoints(self):
        """
        Retrieves the hosted plugins from the specified QHAna endpoints
        """
        plugin_list: List[QhanaPlugin] = []
        for endpoint in self.plugin_runner_endpoints:
            response = requests.get(f"{endpoint}/plugins/", timeout=self.timeout)
            response.raise_for_status()
            for plugin in response.json()["plugins"]:
                try:
                    plugin_response = requests.get(
                        plugin["apiRoot"], timeout=self.timeout
                    )
                    href: Optional[str] = (
                        plugin_response.json().get("entryPoint", {}).get("href", None)
                    )
                    if href:
                        if href.startswith(("http://", "https://")):
                            process_endpoint = href
                        else:
                            process_endpoint = (
                                f"{endpoint.rstrip('/')}/{href.lstrip('/')}"
                            )
                    else:
                        process_endpoint = f'{plugin["apiRoot"]}/process/'

                    plugin_list.append(
                        QhanaPlugin.deserialize(plugin, endpoint, process_endpoint)
                    )
                except Exception:
                    TASK_LOGGER.info(f"Failed to load plugin {plugin}")
        self._plugins = plugin_list

    def get_plugins(self) -> Sequence[QhanaPlugin]:
        if self._plugins is not None:
            return self._plugins
        self._get_plugins_from_endpoints()
        assert self._plugins is not None
        return self._plugins if self._plugins is not None else []

    def resolve(self, plugin_name):
        """
        Retrieves the plugin from the provided plugin name
        :param plugin_name: Name of the plugin
        :return: Plugin
        """

        def match_plugin(plugin: QhanaPlugin) -> bool:
            # TODO replace this matcher with a more sophisticated version once
            # the plugin registry is integrated
            return plugin.name == plugin_name or plugin.identifier == plugin_name

        self.get_plugins()
        plugin = next(filter(match_plugin, self.get_plugins()), None)

        return plugin

    def get_micro_frontend(self, plugin: QhanaPlugin):
        """
        Retrieves the micro frontend of a plugin
        :param plugin: Plugin for retrieving the micro frontend
        :return:
        """
        response = requests.get(f"{plugin.api_root}/ui/", timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def collect_input(
        self,
        local_variables: dict,
        process_instance_id: str,
        camunda_client: CamundaClient,
    ):
        """
        :param local_variables: Variables which may contain input for the QHAna plugin
        :param process_instance_id: The instance id of the running process
        :param camunda_client: Client to be used
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
                retrieved_output = camunda_client.get_global_variable(
                    output_name, process_instance_id=process_instance_id
                )

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

                if mode != input_mode_filename and mode != input_mode_datatype:
                    raise ParameterParsingError(parameter=input_parameter, mode=mode)

                for output in deserialized_outputs:
                    if (mode == input_mode_filename and output.name == mode_val) or (
                        mode == input_mode_datatype and output.data_type == mode_val
                    ):
                        plugin_inputs[input_parameter] = output.href

        return plugin_inputs

    def get_plugin_inputs(self, plugin: QhanaPlugin):
        """
        Gets the list of inputs for a given plugin
        :param plugin: The plugin to get inputs for
        :return:
        """
        url = plugin.api_root if plugin.api_root.endswith("/") else f"{plugin.api_root}/"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        inputs = response.json()["entryPoint"]["dataInput"]

        return [QhanaInput.deserialize(i) for i in inputs]
