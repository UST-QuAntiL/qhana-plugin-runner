from __future__ import annotations

import http.client
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests
from celery.utils.log import get_task_logger
from requests import HTTPError

from .. import DeployWorkflow
from ..config import separate_prefixes
from ..datatypes.qhana_datatypes import QhanaInput, QhanaOutput

config = DeployWorkflow.instance.config

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
    A client for executing QHAna plugins and retreiving their results.
    Completes camunda external tasks and forwards results to the result store
    """

    def __init__(self):
        self.timeout: float = config["request_timeout"]

    def call_qhana_plugin(self, plugin: Dict[str, Any], params):
        process_endpoint: Optional[str] = None
        href: Optional[str] = plugin.get("entryPoint", {}).get("href", None)
        if href:
            if href.startswith(("http://", "https://")):
                process_endpoint = href
            else:
                process_endpoint = (
                    f"{plugin.get('url', '').rstrip('/')}/{href.lstrip('/')}"
                )

        if process_endpoint is None:
            raise ValueError(
                "The plugin does not contain a valid URL for the processing endpoint!"
            )

        response = requests.post(process_endpoint, data=params, timeout=self.timeout)

        response.raise_for_status()

        if response.status_code != http.client.OK:
            raise HTTPError("Unknown status code.", response=response)

        return response.url

    def call_plugin_step(self, href: str, params):
        response = requests.post(href, data=params, timeout=self.timeout)
        response.raise_for_status()

    def get_micro_frontend(self, plugin: Dict[str, Any]):
        """
        Retrieves the micro frontend of a plugin
        :param plugin: Plugin for retrieving the micro frontend
        :return:
        """
        ui_endpoint: Optional[str] = None
        href: Optional[str] = plugin.get("entryPoint", {}).get("uiHref", None)
        if href:
            if href.startswith(("http://", "https://")):
                ui_endpoint = href
            else:
                ui_endpoint = f"{plugin.get('url', '').rstrip('/')}/{href.lstrip('/')}"

        if ui_endpoint is None:
            raise ValueError(
                "The plugin does not contain a valid URL for the user interface endpoint!"
            )

        response = requests.get(ui_endpoint, timeout=self.timeout)
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

        input_prefix = config["workflow_conf"]["task_input_variable_prefix"]

        input_config = config["workflow_conf"]["form_conf"]
        input_mode_text = input_config["text_input_mode"]
        input_mode_filename = input_config["filename_input_mode"]
        input_mode_datatype = input_config["datatype_input_mode"]

        plugin_inputs = {}

        for key, item in local_variables.items():
            input_parameter, prefixes = separate_prefixes(key, config["workflow_conf"])

            # Check if variable is a qhana input
            if input_prefix in prefixes:
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

    def get_plugin_inputs(self, plugin: Dict[str, Any]):
        """
        Gets the list of inputs for a given plugin
        :param plugin: The plugin to get inputs for
        :return:
        """
        inputs = plugin["entryPoint"]["dataInput"]

        return [QhanaInput.deserialize(i) for i in inputs]
