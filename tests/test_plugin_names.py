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

"""Tests ensuring that plugin names and versions are well formed.

Plugin names must match ``[a-z][a-zA-Z0-9_-]*`` because the name is part of
the plugin identifier used in REST URLs, blueprint names, and Celery task
names. Plugin versions must be numeric semantic versioning style versions of
the form ``MAJOR[.MINOR[.PATCH]]`` with an optional ``v`` prefix and at
least one nonzero version part.
"""

import ast
from pathlib import Path
from typing import Iterator, Optional, Tuple

import pytest

from qhana_plugin_runner.util.plugins import (
    PLUGIN_NAME_REGEX,
    QHAnaPluginBase,
    is_valid_plugin_name,
    is_valid_plugin_version,
)
from tests.test_plugin_imports import (
    PLUGIN_BASE_CLASS_NAME,
    PLUGIN_BASE_MODULE,
    PLUGIN_BASE_QALIFIED,
    get_plugin_roots,
)


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
def test_valid_plugin_names(name: str):
    assert is_valid_plugin_name(name)


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
    ],
)
def test_invalid_plugin_names(name: str):
    assert not is_valid_plugin_name(name)


@pytest.mark.parametrize(
    "version",
    [
        "1",
        "0.0.1",
        "0.1",
        "1.0",
        "v1.0.0",
        "v0.1.1",
        "2026.7.17",
    ],
)
def test_valid_plugin_versions(version: str):
    assert is_valid_plugin_version(version)


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
        "1.0 beta",
        "1.0.0rc1",
        "1.0.0-alpha.1",
        "1.0.0.0",
        "1.0.0+build.5",
        "1.0.0.post1",
        "1.0.0.dev3",
    ],
)
def test_invalid_plugin_versions(version: str):
    assert not is_valid_plugin_version(version)


def test_invalid_name_plugin_is_not_registered():
    """A plugin class with an invalid name must not be registered."""
    invalid_name = "Invalid Test Plugin Name"

    class _InvalidNamePlugin(QHAnaPluginBase):
        name = invalid_name
        version = "v0.0.1"

    registered_names = {p.name for p in QHAnaPluginBase.get_plugins().values()}
    assert invalid_name not in registered_names
    assert not hasattr(_InvalidNamePlugin, "instance")


def test_invalid_version_plugin_is_not_registered():
    """A plugin class with an invalid version must not be registered."""

    class _InvalidVersionPlugin(QHAnaPluginBase):
        name = "invalid-version-test-plugin"
        version = "version one"

    registered_names = {p.name for p in QHAnaPluginBase.get_plugins().values()}
    assert "invalid-version-test-plugin" not in registered_names
    assert not hasattr(_InvalidVersionPlugin, "instance")


def test_valid_name_plugin_is_registered():
    """A plugin class with a valid name must be registered."""

    class _ValidNamePlugin(QHAnaPluginBase):
        name = "valid-test-plugin-name"
        version = "v0.0.1"

    try:
        instance = _ValidNamePlugin.instance
        assert QHAnaPluginBase.get_plugins()[instance.identifier] is instance
    finally:
        # remove the test plugin to keep the global registry clean
        QHAnaPluginBase.get_plugins().pop("valid-test-plugin-name@v0-0-1", None)


def _get_base_class_aliases(tree: ast.Module) -> set:
    """Collect the local names of QHAnaPluginBase in a module."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if tuple(n.name.split(".")) == PLUGIN_BASE_QALIFIED:
                    aliases.add(n.asname if n.asname else PLUGIN_BASE_CLASS_NAME)
        elif isinstance(node, ast.ImportFrom):
            if node.module and tuple(node.module.split(".")) == PLUGIN_BASE_MODULE:
                for n in node.names:
                    if n.name == PLUGIN_BASE_CLASS_NAME:
                        aliases.add(n.asname if n.asname else PLUGIN_BASE_CLASS_NAME)
    return aliases


def _get_module_constants(tree: ast.Module) -> dict:
    """Collect module level string constants (e.g. ``_plugin_name``)."""
    constants = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value.value
    return constants


def _resolve_name_value(value: ast.expr, constants: dict) -> Optional[str]:
    """Resolve a ``name = ...`` class attribute to a string if possible."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    if isinstance(value, ast.Name):
        return constants.get(value.id)
    return None


def _iter_plugin_attr_values(
    path: Path, attr: str
) -> Iterator[Tuple[Path, int, Optional[str]]]:
    """Yield (file, lineno, resolved value) of a class attribute (e.g. ``name``)
    for every plugin class in a python file.

    The resolved value is None if the attribute cannot be resolved statically.
    """
    tree = ast.parse(path.read_text(), path)
    aliases = _get_base_class_aliases(tree)
    if not aliases:
        return
    constants = _get_module_constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_plugin = any(
            isinstance(base, ast.Name) and base.id in aliases for base in node.bases
        )
        if not is_plugin:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
                value = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets = [stmt.target]
                value = stmt.value
            else:
                continue
            if any(isinstance(t, ast.Name) and t.id == attr for t in targets):
                yield path, stmt.lineno, _resolve_name_value(value, constants)
                break
        else:
            yield path, node.lineno, None


def _scan_plugin_attr(attr: str, is_valid) -> Tuple[list, list]:
    """Scan all plugin classes and validate the given class attribute.

    Returns a tuple of (violations, unresolved) location strings.
    """
    violations = []
    unresolved = []
    found_values = 0
    for loc in get_plugin_roots():
        files = [loc] if loc.is_file() else loc.rglob("*.py")
        for file_ in files:
            if file_.name.startswith("test_") or "tests" in file_.parts:
                continue
            for path, lineno, value in _iter_plugin_attr_values(file_, attr):
                if value is None:
                    unresolved.append(f"  {path}:{lineno}")
                    continue
                found_values += 1
                if not is_valid(value):
                    violations.append(f"  {path}:{lineno}: '{value}'")

    assert found_values > 0, f"No plugin {attr}s found, the AST scan may be broken."
    return violations, unresolved


def test_plugin_names_match_regex_static():
    """Check the name attribute of all plugin classes in the plugin folders.

    Uses a static AST scan so that plugins with invalid names are found even
    though plugin registration silently drops them.
    """
    violations, unresolved = _scan_plugin_attr("name", is_valid_plugin_name)

    if violations:
        pytest.fail(
            "Found plugin names that do not match the regex "
            f"'{PLUGIN_NAME_REGEX.pattern}':\n" + "\n".join(violations)
        )
    if unresolved:
        pytest.fail(
            "Could not statically resolve the name of the following plugin "
            "classes. Assign a string literal or a module level string "
            "constant to the name attribute:\n" + "\n".join(unresolved)
        )


def test_plugin_versions_are_valid_static():
    """Check the version attribute of all plugin classes in the plugin folders.

    Uses a static AST scan so that plugins with invalid versions are found
    even though plugin registration silently drops them.
    """
    violations, unresolved = _scan_plugin_attr("version", is_valid_plugin_version)

    if violations:
        pytest.fail(
            "Found plugin versions that are not of the form "
            "'MAJOR[.MINOR[.PATCH]]' with an optional 'v' prefix:\n"
            + "\n".join(violations)
        )
    if unresolved:
        pytest.fail(
            "Could not statically resolve the version of the following plugin "
            "classes. Assign a string literal or a module level string "
            "constant to the version attribute:\n" + "\n".join(unresolved)
        )


def test_registered_plugin_names_match_regex(app):
    """Check the names of all plugins that loaded successfully."""
    with app.app_context():
        plugins = QHAnaPluginBase.get_plugins()
        assert len(plugins) > 0, "No plugins found - ensure PLUGIN_FOLDERS are configured"
        violations = [
            f"  {plugin_id}: '{plugin.name}'"
            for plugin_id, plugin in plugins.items()
            if not is_valid_plugin_name(plugin.name)
        ]
        if violations:
            pytest.fail(
                "Found registered plugins with invalid names:\n" + "\n".join(violations)
            )


def test_registered_plugin_versions_are_valid(app):
    """Check the versions of all plugins that loaded successfully."""
    with app.app_context():
        plugins = QHAnaPluginBase.get_plugins()
        assert len(plugins) > 0, "No plugins found - ensure PLUGIN_FOLDERS are configured"
        violations = [
            f"  {plugin_id}: '{plugin.version}'"
            for plugin_id, plugin in plugins.items()
            if not is_valid_plugin_version(plugin.version)
        ]
        if violations:
            pytest.fail(
                "Found registered plugins with invalid versions:\n"
                + "\n".join(violations)
            )
