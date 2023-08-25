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
from collections import ChainMap
from http.client import parse_headers
from io import BytesIO
from re import Match
from tempfile import NamedTemporaryFile, SpooledTemporaryFile
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast
from urllib.parse import parse_qs, urljoin, urlsplit

import requests
from celery.utils.log import get_task_logger
from requests import PreparedRequest, Response, request

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.virtual_plugins import PluginState
from qhana_plugin_runner.storage import STORE

from .jinja_utils import render_template_sandboxed
from .response_handling import ResponseHandlingStrategy
from .schemas import ConnectorVariable
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

    request_variables = json.loads(task_data.parameters)  # FIXME
    assert isinstance(request_variables, dict)
    request_variables = prepare_variables(
        request_variables, connector.get("variables", [])
    )

    request_header_template = connector["request_headers"]
    request_body_template = connector["request_body"]
    request_file_descriptors = connector.get("request_files", [])
    response_mapping = connector.get("response_mapping", [])

    assert isinstance(request_header_template, str)
    assert isinstance(request_body_template, str)

    task_data.add_task_log_entry("Preparing request...", commit=True)

    headers = render_template_sandboxed(request_header_template, request_variables)
    body = render_template_sandboxed(request_body_template, request_variables)

    parsed_headers = parse_headers(BytesIO(headers.encode()))
    headers_dict = {}

    for k, v in parsed_headers.items():
        if k in headers_dict:
            headers_dict[k] += f", {v}"
        else:
            headers_dict[k] = v

    base_url = resolve_base_url(connector["base_url"])
    endpoint_url = render_endpoint(
        connector["endpoint_url"],
        connector.get("endpoint_variables", {}),
        request_variables,
    )
    request_url = urljoin(base_url, endpoint_url)
    request_method = connector["endpoint_method"]

    query_variables = {
        k: render_template_sandboxed(v, request_variables)
        for k, v in cast(dict, connector.get("endpoint_query_variables", {})).items()
    }

    response_strategy = ResponseHandlingStrategy.get(
        connector.get("response_handling", "default")
    )

    request_files = _download_files(request_file_descriptors)

    task_data.add_task_log_entry(
        f"Sending {request_method} request to {request_url}", commit=True
    )

    try:
        response = request(
            method=request_method,
            url=request_url,
            params=query_variables,
            headers=headers_dict,
            data=body,
            files=request_files,
            timeout=response_strategy.timeout,
            allow_redirects=response_strategy.follow_redirects,
            stream=response_strategy.stream_response,
        )
    finally:
        # close request files
        for file in request_files.values():
            try:
                file[1].close()
            except:
                pass  # ensure other files are tried too

    task_data.add_task_log_entry("Handle response...", commit=True)

    original_request = response.request
    response = response_strategy.handle_response(response, task_data, TASK_LOGGER)

    task_data.add_task_log_entry("Write output to disk...", commit=True)

    template_context: Mapping[str, Any] = ChainMap(
        {
            "request": RequestProxy(original_request),
            "response": ResponseProxy(response),
        },
        request_variables,
    )

    for response_map in response_mapping:
        # FIXME: Files with dereference_url=True in the response output descriptor
        # will have a URL in the content that points to the actual content.
        # Use the URL file store for these files!
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

    task_data.save(commit=True)

    return "finished"


def resolve_base_url(base_url: str) -> str:
    scheme, netloc, *rest = urlsplit(base_url)
    if scheme == "service":
        # TODO: lookup serice url and replace scheme/netloc and base path before returning url!
        raise NotImplementedError(
            "TODO: implemen service url lookup using plugin registry!"
        )
    return base_url


def render_endpoint(
    endpoint_url: str, endpoint_variables: dict[str, str], variables: dict
) -> str:
    def get_variable(var: Match[str]) -> str:
        var_template = endpoint_variables.get(var.group(0), "")
        return render_template_sandboxed(var_template, variables)

    return re.sub(r"\{([^}]+)\}", get_variable, endpoint_url)


def prepare_variables(
    request_variables: dict, variable_descriptions: List[ConnectorVariable]
) -> dict:
    for var in variable_descriptions:
        if var.get("data_type") or var.get("content_type"):
            # TODO implement a proxy class that  allows for easy retrieval of file based
            # data like entities and replace the variable in request_variables
            # with the proxy (see DataProxy)
            raise NotImplementedError("File variables are not implemented yet!")
    return request_variables


def _download_files(
    request_file_descriptors: List[Dict],
) -> Dict[str, Tuple[str, BytesIO, str]]:
    request_files = {}

    for req_file in request_file_descriptors:
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


class DataProxy:
    def __init__(self, variable, variable_descriptor: ConnectorVariable):
        raise NotImplementedError


class RequestProxy:
    def __init__(self, request: PreparedRequest):
        self._request = request

    def __str__(self) -> str:
        return str(self._request)

    def __repr__(self) -> str:
        return repr(self._request)

    @property
    def url(self):
        return self._request.url

    @property
    def params(self):
        query = urlsplit(self._request.url)[3]
        if isinstance(query, bytes):
            query = query.decode()
        parsed = parse_qs(query, keep_blank_values=True)
        self.__dict__["params"] = parsed
        return parsed

    @property
    def headers(self):
        return self._request.headers

    @property
    def body(self):
        return self._request.body

    @property
    def json(self):
        parsed = json.loads(self._request.body)
        self.__dict__["json"] = parsed
        return parsed


class ResponseProxy:
    def __init__(self, response: Response):
        self._response = response

    def __str__(self) -> str:
        return str(self._response)

    def __repr__(self) -> str:
        return repr(self._response)

    @property
    def url(self):
        return self._response.url

    @property
    def status_code(self):
        return self._response.status_code

    @property
    def headers(self):
        return self._response.headers

    @property
    def body(self):
        content = self._response.text
        self.__dict__["body"] = content
        return content

    @property
    def body_raw(self):
        content = self._response.content
        self.__dict__["body_raw"] = content
        return content

    @property
    def json(self):
        json = self._response.json()
        self.__dict__["json"] = json
        return json
