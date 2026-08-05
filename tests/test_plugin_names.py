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

"""Tests ensuring that plugin registration rejects malformed names and versions.

Plugin names must match ``[a-z][a-zA-Z0-9_-]*`` because the name is part of
the plugin identifier used in REST URLs, blueprint names, and Celery task
names. Plugin versions must be numeric semantic versioning style versions of
the form ``MAJOR[.MINOR[.PATCH]]`` with an optional ``v`` prefix and at
least one nonzero version part.

The tests cover the rules, not the plugins in this repository. Checking every
plugin requires either all plugin dependencies installed in CI or a static
scan of the plugin sources, see the follow up issue on repository wide checks.
"""

import pytest

from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

VALID_NAME = "valid-test-plugin-name"
VALID_VERSION = "v1.0.0"


def _registers(name: str, version: str) -> bool:
    """Report whether a plugin class with this name and version is registered.

    Registration is the only public API that validates plugin names and
    versions, so the checks go through a throwaway plugin class. The global
    plugin registry is restored afterwards to keep the probe invisible to
    other tests.
    """
    plugins = QHAnaPluginBase.get_plugins()
    identifier = plugin_identifier(name, version)
    registered_before = plugins.get(identifier)

    probe = type("_ProbePlugin", (QHAnaPluginBase,), {"name": name, "version": version})
    accepted = hasattr(probe, "instance")

    if registered_before is not None:
        plugins[identifier] = registered_before
    elif accepted:
        plugins.pop(identifier, None)
    return accepted


@pytest.mark.parametrize(
    "name",
    [
        "a",
        "my-plugin",
        "my_plugin",
        "plugin2",
        "amazon-braket-local-simulator",
        "pytket_qulacsBackend-simulator",
        "element_sim-to-element_dist-transformers",
    ],
)
def test_plugin_with_valid_name_is_registered(name: str):
    assert _registers(name, VALID_VERSION)


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "my plugin",
        " my-plugin",
        "my-plugin ",
        "my-plugin\n",
        "My-Plugin",
        "AmazonBraket_LocalSimulator",
        "2plugin",
        "-plugin",
        "_plugin",
        "my.plugin",
        "my-plugin@v1",
        "my/plugin",
        "my:plugin",
        "my+plugin",
        "plugin(1)",
        "plüg-in",
        "\tmy-plugin",
    ],
)
def test_plugin_with_invalid_name_is_not_registered(name: str):
    assert not _registers(name, VALID_VERSION)


@pytest.mark.parametrize(
    "version",
    [
        "1",
        "0.0.1",
        "0.1",
        "1.0",
        "v1",
        "v0.1",
        "v1.0.0",
        "v0.1.1",
        "2026.7.17",
        "10.20.30",
    ],
)
def test_plugin_with_valid_version_is_registered(version: str):
    assert _registers(VALID_NAME, version)


@pytest.mark.parametrize(
    "version",
    [
        "",
        " ",
        "v",
        "0",
        "0.0",
        "0.0.0",
        "v0.0.0",
        "abc1.0.0",
        "version one",
        "1..0",
        "1.",
        ".1",
        "1.0 beta",
        " 1.0.0",
        "1.0.0 ",
        "1.0.0\n",
        "1.0.0rc1",
        "1.0.0-alpha.1",
        "1.0.0.0",
        "1.0.0+build.5",
        "1.0.0.post1",
        "1.0.0.dev3",
        "V1.0.0",
        "vv1.0.0",
        "1-0-0",
        "-1.0.0",
    ],
)
def test_plugin_with_invalid_version_is_not_registered(version: str):
    assert not _registers(VALID_NAME, version)


@pytest.mark.parametrize(
    "attribute",
    ["name", "version"],
)
def test_plugin_without_name_or_version_is_not_registered(attribute: str):
    """A plugin class must define both attributes to be registered."""
    attributes = {"name": VALID_NAME, "version": VALID_VERSION}
    del attributes[attribute]

    probe = type("_ProbePlugin", (QHAnaPluginBase,), attributes)

    assert not hasattr(probe, "instance")
