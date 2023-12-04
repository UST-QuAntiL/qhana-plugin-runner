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
import os

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import PluginState
from .prefill_tasks import unlock_connector

from ..plugin import RESTConnector
from .schemas import ConnectorKey
from ..prompts import (
    get_relevant_endpoints_from_llm,
    get_relevant_endpoints_from_vector_store,
    get_chat_model,
)

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{RESTConnector.instance.identifier}.ai_assistance")
def ai_assistance(connector_id: str, last_step: str, user_request: str):
    parent_plugin = RESTConnector.instance

    # connector definition data
    connector = PluginState.get_value(
        parent_plugin.identifier, connector_id, default=None
    )

    # extra data that is not strictly required for the connector definition
    # e.g., autocomplete info, extra documentation, etc.
    connector_extra = PluginState.get_value(
        parent_plugin.identifier, f"{connector_id}__extra", default={}
    )
    if connector is None:
        return
    assert isinstance(connector, dict)
    assert isinstance(connector_extra, dict)
    new_data = dict(connector)
    new_extra = dict(connector_extra)

    changed = False

    if last_step == ConnectorKey.ENDPOINT_URL.value:
        chat_ai = get_chat_model()
        relevant_endpoints = get_relevant_endpoints_from_vector_store(
            connector["openapi_spec_url"], user_request
        )
        relevant_endpoints = get_relevant_endpoints_from_llm(
            chat_ai, relevant_endpoints, user_request
        )

        TASK_LOGGER.info(f"found these relevant endpoints: {relevant_endpoints}")

        if len(relevant_endpoints) > 0:
            # choose first relevant endpoint
            new_data["endpoint_url"] = relevant_endpoints[0].path
            new_data["endpoint_method"] = relevant_endpoints[0].method

            finished = set(connector.get("finished_steps", []))
            finished.add(ConnectorKey.ENDPOINT_URL.value)
            finished.add(ConnectorKey.ENDPOINT_METHOD.value)
            new_data["finished_steps"] = list(
                finished.intersection(
                    {
                        ConnectorKey.BASE_URL.value,
                        ConnectorKey.OPENAPI_SPEC.value,
                        ConnectorKey.ENDPOINT_URL.value,
                        ConnectorKey.ENDPOINT_METHOD.value,
                    }
                )
            )

            changed = True

    if changed:
        PluginState.set_value(
            parent_plugin.identifier, f"{connector_id}__extra", new_extra
        )
        PluginState.set_value(
            parent_plugin.identifier, connector_id, new_data, commit=True
        )

    unlock_connector(connector_id=connector_id)
