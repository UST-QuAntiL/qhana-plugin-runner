from typing import Any, Dict, cast
from uuid import uuid4

from qhana_plugin_runner.db.models.virtual_plugins import PluginState

from .plugin import RESTConnector

_WIP_CONNECTORS = "wip_connectors"


def start_new_connector(name: str, commit: bool = False) -> str:
    plugin = RESTConnector.instance
    connectors = PluginState.get_value(plugin.identifier, _WIP_CONNECTORS, default=None)
    if connectors is None:
        connectors = {}
    assert isinstance(connectors, dict)

    connector_id = str(uuid4())

    connectors[connector_id] = name

    PluginState.set_value(plugin.identifier, _WIP_CONNECTORS, connectors, commit=commit)

    return connector_id


def remove_wip_connector(connector_id: str, commit: bool = False):
    plugin = RESTConnector.instance
    connectors = PluginState.get_value(plugin.identifier, _WIP_CONNECTORS, default=None)
    if not connectors:
        return  # already removed
    assert isinstance(connectors, dict)

    try:
        del connectors[connector_id]
    except KeyError:
        return  # already removed

    PluginState.set_value(plugin.identifier, _WIP_CONNECTORS, connectors, commit=commit)


def save_wip_connectors(connectors: Dict[str, str], commit: bool = False):
    plugin = RESTConnector.instance
    connectors_ = cast(Dict[str, Any], connectors)
    PluginState.set_value(plugin.identifier, _WIP_CONNECTORS, connectors_, commit=commit)


def get_wip_connectors() -> Dict[str, str]:
    plugin = RESTConnector.instance
    connectors = PluginState.get_value(plugin.identifier, _WIP_CONNECTORS, default=None)
    if not connectors:
        return {}
    assert isinstance(connectors, dict)
    connectors = cast(Dict[str, str], connectors)

    # return a copy of the dict to not affect DB on mutations
    return dict(connectors)
