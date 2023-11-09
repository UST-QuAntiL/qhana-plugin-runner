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

import re
from typing import List, Optional

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import PluginState
from qhana_plugin_runner.requests import open_url

from ..openapi import (
    get_endpoint_methods,
    get_endpoint_paths,
    get_example_body,
    get_query_variables,
    get_upload_files,
    parse_spec,
)
from ..plugin import RESTConnector
from .schemas import ConnectorKey

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{RESTConnector.instance.identifier}.unlock_connector", ignore_result=True
)
def unlock_connector(connector_id: str):
    parent_plugin = RESTConnector.instance
    connector = PluginState.get_value(
        parent_plugin.identifier, connector_id, default=None
    )
    if connector is None:
        return
    assert isinstance(connector, dict)
    if connector.get("is_deployed", False):
        return
    new_data = dict(connector)
    del new_data["is_loading"]
    PluginState.set_value(parent_plugin.identifier, connector_id, new_data, commit=True)


@CELERY.task(
    name=f"{RESTConnector.instance.identifier}.prefill_values", ignore_result=True
)
def prefill_values(connector_id: str, last_step: str):
    parent_plugin = RESTConnector.instance
    connector = PluginState.get_value(
        parent_plugin.identifier, connector_id, default=None
    )
    if connector is None:
        return
    assert isinstance(connector, dict)
    new_data = dict(connector)

    changed = False

    if last_step == ConnectorKey.OPENAPI_SPEC.value:
        spec = connector.get("openapi_spec_url")
        if spec and isinstance(spec, str):
            new_data["autocomplete_paths"] = list(get_endpoint_paths(spec))
            changed = True
    elif last_step == ConnectorKey.ENDPOINT_URL.value:
        path = connector.get("endpoint_url")
        if path and isinstance(path, str):
            # extract path variables
            matches = re.findall(r"\{([\w\d]+)\}", path)
            new_data["endpoint_variables"] = {var: "" for var in matches}
            if not new_data["endpoint_variables"]:
                finished = new_data.get("finished_steps", [])
                assert isinstance(finished, list)
                finished.append(ConnectorKey.ENDPOINT_VARIABLES.value)
                new_data["finished_steps"] = finished
            changed = True

        spec = connector.get("openapi_spec_url")
        if path and isinstance(path, str) and spec and isinstance(spec, str):
            # find path methods
            new_data["autocomplete_methods"] = list(get_endpoint_methods(spec, path))
            changed = True
    elif last_step == ConnectorKey.ENDPOINT_METHOD.value:
        spec = connector.get("openapi_spec_url")
        path = connector.get("endpoint_url")
        method = connector.get("endpoint_method")
        if all(x and isinstance(x, str) for x in (spec, path, method)):
            assert (
                isinstance(spec, str)
                and isinstance(path, str)
                and isinstance(method, str)
            )
            parsed_spec = parse_spec(spec)

            # prefill query variables
            new_data["endpoint_query_variables"] = dict(
                get_query_variables(parsed_spec, path, method)
            )

            # TODO prefill header
            pass

            # prefill body
            new_data["request_body"] = get_example_body(parsed_spec, path, method)
            if method.lower() in ("get", "delete"):
                # TODO mark body as filled out if it is expected to be empty for POST/PUT/PATCH!
                finished = connector.get("finished_steps", [])
                finished.append(ConnectorKey.REQUEST_BODY.value)
                connector["finished_steps"] = finished

            # TODO prefill request files?
            files = get_upload_files(parsed_spec, path, method)
            if files:
                new_data["request_files"] = [
                    {
                        "data": "",
                        "dereference_url": False,
                        "content_type": f,
                    }
                    for f in files
                ]

            changed = True

    if changed:
        PluginState.set_value(
            parent_plugin.identifier, connector_id, new_data, commit=True
        )

    unlock_connector(connector_id=connector_id)
