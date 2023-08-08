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
from http.client import parse_headers
from io import BytesIO
from typing import Mapping, Optional
from urllib.parse import urljoin

from celery.utils.log import get_task_logger
from flask import current_app
from requests import request
from requests.exceptions import ConnectionError, HTTPError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.virtual_plugins import PluginState, VirtualPlugin

from .jinja_utils import render_template_sandboxed
from ..plugin import RESTConnector

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{RESTConnector.instance.identifier}.perform_request", bind=True)
def perform_request(self, connector_id: str, db_id: int) -> None:
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
    request_files = connector.get("request_files", [])

    assert isinstance(request_header_template, str)
    assert isinstance(request_body_template, str)

    headers = render_template_sandboxed(request_header_template, request_variables)
    body = render_template_sandboxed(request_body_template, request_variables)

    # TODO: remove prints later
    print(request_variables)
    print(headers)
    print(body)

    parsed_headers = parse_headers(BytesIO(headers.encode()))

    response = request(
        method=connector["endpoint_method"],
        url=urljoin(
            connector["base_url"], connector["endpoint_url"]
        ),  # FIXME allow variables in endpoint
        headers={k: v for k, v in parsed_headers.items()},  # FIXME duplicate headers...
        data=body,
        files=[],  # FIXME allow uploading of files
        timeout=20,  # TODO allow different timeouts?
    )

    # FIXME response handling...

    task_data.add_task_log_entry(
        f"Request:\n\n{response.request!r}\n{response.request.headers}\n{response.request.body}\n\nResponse:\n\n{response!r}\n{response.headers}\n{response.text}",
        commit=True,
    )
