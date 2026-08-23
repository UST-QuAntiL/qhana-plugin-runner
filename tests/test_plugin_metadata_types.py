# Copyright 2026 QHAna plugin runner contributors.
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

"""
Tests for plugin metadata type validation.

Verifies that all plugins use allowed data_types and content_types
as defined in tests.allowed_types.
"""

from http import HTTPStatus

import pytest

from qhana_plugin_runner.util.plugins import QHAnaPluginBase

from .allowed_types import (
    AllowedContentTypes,
    AllowedDataTypesNoFormat,
    AllowedDataTypesWithFormat,
)


def _is_valid_data_type(value: str) -> bool:
    """Check if a data_type value is in allowed lists."""
    return any(e.value == value for e in AllowedDataTypesWithFormat) or any(
        e.value == value for e in AllowedDataTypesNoFormat
    )


def _is_valid_content_type(value: str) -> bool:
    """Check if a content_type value is in allowed lists."""
    return any(e.value == value for e in AllowedContentTypes)


def _get_plugin_entry_point(plugin_id, plugin, client) -> dict | None:
    """
    Validates a plugin's blueprint, fetches its metadata entry point and validates if it
    contains the input and output data types. Returns the metadata entry_point dict if
    successful, or None if the plugin has no API to validate.
    Raises AssertionError if a plugin with an API(``plugin.has_api=True``) is broken.
    """
    if not plugin.has_api:
        # Not every registered plugin is a QHAna plugin with its own API (e.g. the
        # minio-storage plugin only provides a storage backend).
        # ``has_api`` is set by ``register_plugins`` once ``get_api_blueprint`` is
        # confirmed to succeed.
        return None

    try:
        blueprint = plugin.get_api_blueprint()
        if not blueprint or not blueprint.deferred_functions:
            raise AssertionError(
                f"Blueprint of plugin {plugin_id} should not be undefined or should have deferred functions."
            )
    except NotImplementedError:
        raise AssertionError(
            f"Plugin {plugin_id} has has_api=True but get_api_blueprint() raised NotImplementedError."
        )

    try:
        response = client.get(f"/plugins/{plugin.identifier}/")
    except Exception as err:
        raise AssertionError(
            f"Response of default route of plugin {plugin_id} (get metadata) crashed: {err}"
        )

    if response.status_code != HTTPStatus.OK:
        raise AssertionError(
            f"Response of default route of plugin {plugin_id} (get metadata) did not return HTTP status code 200."
        )

    metadata = response.get_json()
    assert metadata, f"Metadata for plugin {plugin_id} is empty or invalid."

    entry_point = metadata.get("entryPoint", {})
    assert (
        "dataInput" in entry_point
    ), f"Plugin {plugin_id} entry_point is missing 'dataInput'"
    assert (
        "dataOutput" in entry_point
    ), f"Plugin {plugin_id} entry_point is missing 'dataOutput'"

    return entry_point


def test_all_plugins_use_allowed_data_types(app, client):
    """
    Verify all plugins use data_types from AllowedDataTypesWithFormat or AllowedDataTypesNoFormat.

    Iterates through all registered plugins, retrieves their metadata via GET,
    and validates that all data_input and data_output data_types are allowed.
    """
    with app.app_context():
        invalid_types = []

        plugins = QHAnaPluginBase.get_plugins()
        assert len(plugins) > 0, "No plugins found - ensure PLUGIN_FOLDERS are configured"

        for plugin_id, plugin in plugins.items():
            entry_point = _get_plugin_entry_point(plugin_id, plugin, client)
            if not entry_point:
                continue  # Plugin has no API to validate

            # Check data_input types
            for input_item in entry_point["dataInput"]:
                data_type = input_item["dataType"]
                if data_type and not _is_valid_data_type(data_type):
                    invalid_types.append(
                        {
                            "plugin": plugin_id,
                            "location": "dataInput.dataType",
                            "value": data_type,
                        }
                    )

            # Check data_output types
            for output_item in entry_point["dataOutput"]:
                data_type = output_item["dataType"]
                if data_type and not _is_valid_data_type(data_type):
                    invalid_types.append(
                        {
                            "plugin": plugin_id,
                            "location": "dataOutput.dataType",
                            "value": data_type,
                        }
                    )

        if invalid_types:
            error_lines = ["Invalid data_types found in plugins:"]
            for item in invalid_types:
                error_lines.append(
                    f"  {item['plugin']}: {item['location']} = '{item['value']}'"
                )
            error_msg = "\n ".join(error_lines)
            pytest.fail(error_msg)


def test_all_plugins_use_allowed_content_types(app, client):
    """
    Verify all plugins use content_types from AllowedContentTypes.

    Iterates through all registered plugins, retrieves their metadata via GET,
    and validates that all data_input and data_output content_types are allowed.
    """
    with app.app_context():
        invalid_types = []

        plugins = QHAnaPluginBase.get_plugins()
        assert len(plugins) > 0, "No plugins found - ensure PLUGIN_FOLDERS are configured"

        for plugin_id, plugin in plugins.items():
            entry_point = _get_plugin_entry_point(plugin_id, plugin, client)
            if not entry_point:
                continue  # Plugin has no API to validate

            # Check data_input content types
            for input_item in entry_point["dataInput"]:
                for content_type in input_item["contentType"]:
                    if content_type and not _is_valid_content_type(content_type):
                        invalid_types.append(
                            {
                                "plugin": plugin_id,
                                "location": "dataInput.contentType",
                                "value": content_type,
                            }
                        )

            # Check data_output content types
            for output_item in entry_point["dataOutput"]:
                for content_type in output_item["contentType"]:
                    if content_type and not _is_valid_content_type(content_type):
                        invalid_types.append(
                            {
                                "plugin": plugin_id,
                                "location": "dataOutput.contentType",
                                "value": content_type,
                            }
                        )

        if invalid_types:
            error_lines = ["Invalid content_types found in plugins:"]
            for item in invalid_types:
                error_lines.append(
                    f"  {item['plugin']}: {item['location']} = '{item['value']}'"
                )
            error_msg = "\n".join(error_lines)
            pytest.fail(error_msg)
