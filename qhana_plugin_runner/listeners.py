"""Module coontaining signal listeners."""

from flask import Flask

from .registry_client import PLUGIN_REGISTRY_CLIENT


def on_virtual_plugin_create(app, *, plugin_url, **extra):
    if not PLUGIN_REGISTRY_CLIENT.ready:
        return  # Cannot notify registry of this change
    PLUGIN_REGISTRY_CLIENT.fetch_by_rel(
        ["plugin", ["plugin", "post"]], query_params={"url": plugin_url}
    )


def on_virtual_plugin_remove(app, *, plugin_url, **extra):
    if not PLUGIN_REGISTRY_CLIENT.ready:
        return  # Cannot notify registry of this change
    PLUGIN_REGISTRY_CLIENT.fetch_by_rel(
        ["plugin", ["plugin", "post"]], query_params={"url": plugin_url}
    )


def register_signal_listeners(app: Flask):
    from .db.models.virtual_plugins import (
        VIRTUAL_PLUGIN_CREATED,
        VIRTUAL_PLUGIN_REMOVED,
    )

    VIRTUAL_PLUGIN_CREATED.connect(on_virtual_plugin_create, app)
    VIRTUAL_PLUGIN_REMOVED.connect(on_virtual_plugin_remove, app)
