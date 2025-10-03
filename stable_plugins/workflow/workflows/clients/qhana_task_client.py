from __future__ import annotations

import http.client
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import requests
from celery.utils.log import get_task_logger
from requests import HTTPError

from .. import DeployWorkflow
from ..config import FormInputConfig, separate_prefixes
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


def _mimetype_like_to_regex(mimetype: str) -> re.Pattern:
    if not mimetype or mimetype == "*":
        return re.compile(r"[^/]*/[^/]*")

    if "/" not in mimetype:
        return re.compile(mimetype + r"/[^/]*")

    first, second = mimetype.split("/", maxsplit=1)
    if not first:
        first = r"[^/]*"
    if not second:
        second = r"[^/]*"

    return re.compile(f"{first}/{second}")


_GLOB_TO_REGEX = str.maketrans(
    {
        "*": ".*",
        "?": ".",
        ".": "\\.",
        "+": "\\+",
        "(": "\\(",
        ")": "\\)",
        "[": "\\[",
        "]": "\\]",
        "{": "\\{",
        "}": "\\}",
    }
)


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

        try:
            response.raise_for_status()
        except Exception:
            TASK_LOGGER.error(
                f"Response to {process_endpoint} failed with the following parameters:\n{params}\n\n---response---\n{response.text}"
            )
            raise

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

        plugin_inputs = {}

        for key, item in local_variables.items():
            input_parameter, prefixes = separate_prefixes(key, config["workflow_conf"])

            # Check if variable is a qhana input
            if input_prefix in prefixes:
                match item:
                    case {"type": "String", "value": str(value)}:
                        plugin_inputs[input_parameter] = value
                    case {"type": "Object", "value": {"from": _}}:
                        value = self._parse_map_input(
                            item["value"],
                            input_parameter,
                            process_instance_id,
                            camunda_client,
                        )

                        plugin_inputs[input_parameter] = value
                    case {"type": "Object", "value": {}} if len(item["value"]) == 1:
                        output_name, select = list(item["value"].items())[0]

                        value = self._parse_legacy_map_input(
                            output_name,
                            select,
                            input_parameter,
                            process_instance_id,
                            camunda_client,
                            input_config,
                        )

                        plugin_inputs[input_parameter] = value
                    case _:
                        TASK_LOGGER.error(
                            f"Could not parse {item} as an input parameter definition."
                        )
                        raise ParameterParsingError(parameter=input_parameter)

        return plugin_inputs

    def _parse_map_input(
        self,
        input_def: dict[str, str],
        input_parameter: str,
        process_instance_id: str,
        camunda_client: CamundaClient,
    ) -> str:
        input_def = dict(input_def)  # copy to allow deleting keys
        target_variable = input_def.pop("from")

        retrieved_output = camunda_client.get_global_variable(
            target_variable, process_instance_id=process_instance_id
        )

        # Treat output as plain text
        if not input_def or input_def.get("type") == "plain":
            return retrieved_output

        if input_def.get("type", "data") == "data":
            # assume input specifiec data
            matchers: list[Callable[[QhanaOutput], bool]] = []
            if "name" in input_def:
                # ensure only filenames
                name = Path(input_def.pop("name").strip()).name
                name_regex = re.compile(name.translate(_GLOB_TO_REGEX))

                matchers.append(lambda o: name_regex.fullmatch(o.name))

            if "dataType" in input_def:
                data_type = input_def.pop("dataType").strip()
                data_type_regex = _mimetype_like_to_regex(data_type)

                matchers.append(lambda o: data_type_regex.fullmatch(o.data_type))

            if "contentType" in input_def:
                content_type = input_def.pop("contentType").strip()
                try:
                    content_types = json.loads(content_type)
                except json.JSONDecodeError:
                    content_types = content_type.split()

                # filter empty values
                content_type_regexes = [
                    _mimetype_like_to_regex(c) for c in content_types if c
                ]

                if content_type_regexes:
                    matchers.append(
                        lambda o: any(
                            r.fullmatch(o.content_type) for r in content_type_regexes
                        )
                    )

            if not matchers:
                raise ParameterParsingError(parameter=input_parameter)
            deserialized_outputs = [
                QhanaOutput.deserialize(output) for output in retrieved_output
            ]
            for output in deserialized_outputs:
                if all(m(output) for m in matchers):
                    return output.href

        raise ParameterParsingError(parameter=input_parameter)

    def _parse_legacy_map_input(
        self,
        output_name: str,
        select: str,
        input_parameter: str,
        process_instance_id: str,
        camunda_client: CamundaClient,
        input_config: FormInputConfig,
    ) -> str:
        input_mode_text = input_config["text_input_mode"]
        input_mode_filename = input_config["filename_input_mode"]
        input_mode_datatype = input_config["datatype_input_mode"]

        # Retrieves the contents of an output that is used as input
        retrieved_output = camunda_client.get_global_variable(
            output_name, process_instance_id=process_instance_id
        )

        # Treat output as plain text
        if select == input_mode_text:
            return retrieved_output

        # If output type is not plain text, e.g., enum or choice
        if type(retrieved_output) is str and select != input_mode_text:
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
                return output.href

        raise ParameterParsingError(parameter=input_parameter)

    def get_plugin_inputs(self, plugin: Dict[str, Any]):
        """
        Gets the list of inputs for a given plugin
        :param plugin: The plugin to get inputs for
        :return:
        """
        inputs = plugin["entryPoint"]["dataInput"]

        return [QhanaInput.deserialize(i) for i in inputs]
