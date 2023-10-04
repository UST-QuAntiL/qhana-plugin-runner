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

from ..openapi import get_endpoint_paths
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
    if last_step == ConnectorKey.OPENAPI_SPEC.value:
        spec = connector.get("openapi_spec_url")
        if spec and isinstance(spec, str):
            new_data["autocomplete_paths"] = list(get_endpoint_paths(spec))
            PluginState.set_value(
                parent_plugin.identifier, connector_id, new_data, commit=True
            )
    if last_step == ConnectorKey.ENDPOINT_URL.value:
        path = connector.get("endpoint_url")
        if path and isinstance(path, str):
            matches = re.findall(r"\{([\w\d]+)\}", path)
            new_data["endpoint_variables"] = {var: "" for var in matches}
            if not new_data["endpoint_variables"]:
                finished = new_data.get("finished_steps", [])
                assert isinstance(finished, list)
                finished.append(ConnectorKey.ENDPOINT_VARIABLES.value)
                new_data["finished_steps"] = finished
            PluginState.set_value(
                parent_plugin.identifier, connector_id, new_data, commit=True
            )

    unlock_connector(connector_id=connector_id)
