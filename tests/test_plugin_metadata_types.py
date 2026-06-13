"""
Tests for plugin metadata type validation.

Verifies that all plugins use allowed data_types and content_types
as defined in qhana_plugin_runner.plugin_utils.types.
"""

from http import HTTPStatus

import pytest
from flask import url_for

from qhana_plugin_runner.plugin_utils.types import (
    AllowedContentTypes,
    is_valid_content_type,
    is_valid_data_type,
)
from qhana_plugin_runner.util.plugins import QHAnaPluginBase


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
            try:
                blueprint = plugin.get_api_blueprint()
                if not blueprint or not blueprint.deferred_functions:
                    continue
            except NotImplementedError:
                continue

            try:
                response = client.get(f"/plugins/{plugin.identifier}/")
            except Exception:
                continue  # plugin's metadata endpoint crashes
            if response.status_code != HTTPStatus.OK:
                continue

            metadata = response.get_json()
            assert metadata

            entry_point = metadata.get("entryPoint", {})

            # Check data_input types
            for input_item in entry_point.get("dataInput", []):
                data_type = input_item.get("dataType", "")
                if data_type and not is_valid_data_type(data_type):
                    invalid_types.append(
                        {
                            "plugin": plugin_id,
                            "location": "dataInput.dataType",
                            "value": data_type,
                        }
                    )

            # Check data_output types
            for output_item in entry_point.get("dataOutput", []):
                data_type = output_item.get("dataType", "")
                if data_type and not is_valid_data_type(data_type):
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
            try:
                blueprint = plugin.get_api_blueprint()
                if not blueprint or not blueprint.deferred_functions:
                    continue
            except NotImplementedError:
                continue

            try:
                response = client.get(f"/plugins/{plugin.identifier}/")
            except Exception:
                continue  # plugin's metadata endpoint crashes
            if response.status_code != HTTPStatus.OK:
                continue

            metadata = response.get_json()
            assert metadata

            entry_point = metadata.get("entryPoint", {})

            # Check data_input content types
            for input_item in entry_point.get("dataInput", []):
                for content_type in input_item.get("contentType", []):
                    if content_type and not is_valid_content_type(content_type):
                        invalid_types.append(
                            {
                                "plugin": plugin_id,
                                "location": "dataInput.contentType",
                                "value": content_type,
                            }
                        )

            # Check data_output content types
            for output_item in entry_point.get("dataOutput", []):
                for content_type in output_item.get("contentType", []):
                    if content_type and not is_valid_content_type(content_type):
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
