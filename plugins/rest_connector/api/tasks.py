# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import re
from http.client import parse_headers
from io import BytesIO
from re import Match
from tempfile import NamedTemporaryFile, SpooledTemporaryFile
from typing import Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urljoin

import requests
from celery.utils.log import get_task_logger
from flask import current_app
from requests import request
from requests.exceptions import ConnectionError, HTTPError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.virtual_plugins import PluginState, VirtualPlugin
from qhana_plugin_runner.storage import STORE

from .jinja_utils import render_template_sandboxed
from ..plugin import RESTConnector

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{RESTConnector.instance.identifier}.perform_request", bind=True)
def perform_request(self, connector_id: str, db_id: int) -> str:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(f"Could not load task data for task id '{db_id}'!")
        raise ValueError(f"Task {db_id} was not found!")

    parent_plugin = RESTConnector.instance
    connector = PluginState.get_value(parent_plugin.identifier, connector_id, default={})
    assert isinstance(connector, dict), "Type assertion"

    request_variables = json.loads(task_data.parameters)

    assert isinstance(request_variables, dict)

    request_header_template = connector["request_headers"]
    request_body_template = connector["request_body"]
    request_file_descriptors = connector.get("request_files", [])
    response_mapping = connector.get("response_mapping", [])

    assert isinstance(request_header_template, str)
    assert isinstance(request_body_template, str)

    headers = render_template_sandboxed(request_header_template, request_variables)
    body = render_template_sandboxed(request_body_template, request_variables)

    # TODO: remove prints later
    print(request_variables)
    print(headers)
    print(body)

    parsed_headers = parse_headers(BytesIO(headers.encode()))
    headers_dict = {}

    for k, v in parsed_headers.items():
        if k in headers_dict:
            headers_dict[k] += f", {v}"
        else:
            headers_dict[k] = v

    endpoint_url = render_endpoint(
        connector["endpoint_url"],
        connector.get("endpoint_variables", {}),
        request_variables,
    )

    query_variables = {
        k: render_template_sandboxed(v, request_variables)
        for k, v in cast(dict, connector.get("endpoint_query_variables", {})).items()
    }

    request_files = _download_files(request_file_descriptors)

    response = request(
        method=connector["endpoint_method"],
        url=urljoin(connector["base_url"], endpoint_url),
        params=query_variables,
        headers=headers_dict,
        data=body,
        files=request_files,
        timeout=20,  # TODO allow different timeouts?
    )

    # close request files
    for file in request_files.values():
        file[1].close()

    template_context: Dict[str, Any] = request_variables | {
        "request": response.request,
        "response": response.json(),
    }

    for response_map in response_mapping:
        with SpooledTemporaryFile(mode="w") as output:
            content = render_template_sandboxed(response_map["data"], template_context)

            output.write(content)
            STORE.persist_task_result(
                task_db_id=db_id,
                file_=output,
                file_name=response_map["name"],
                file_type=response_map["data_type"],
                mimetype=response_map["content_type"],
            )

    task_data.add_task_log_entry(
        f"Request:\n\n{response.request!r}\n{response.request.headers}\n{response.request.body}\n\nResponse:\n\n{response!r}\n{response.headers}\n{response.text}",
        commit=True,
    )

    return "finished"


def render_endpoint(
    endpoint_url: str, endpoint_variables: dict[str, str], variables: dict
) -> str:
    def get_variable(var: Match[str]) -> str:
        var_template = endpoint_variables.get(var.group(0), "")
        return render_template_sandboxed(var_template, variables)

    return re.sub(r"\{([^}]+)\}", get_variable, endpoint_url)


def _download_files(
    request_file_descriptors: List[Dict],
) -> Dict[str, Tuple[str, BytesIO, str]]:
    # TODO: cache downloaded files for subsequent requests?
    request_files = {}

    for i, req_file in enumerate(request_file_descriptors):
        file_url = req_file["source"]
        response = requests.get(file_url)

        file = NamedTemporaryFile(mode="w+b")
        file.write(response.content)
        request_files[req_file["form_field_name"]] = (
            req_file["form_field_name"],
            file,
            req_file["content_type"],
        )

    return request_files
