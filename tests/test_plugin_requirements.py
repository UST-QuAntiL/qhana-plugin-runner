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

"""Tests ensuring that plugins declare the dependencies they import.

The checks are purely static so that they also run in an environment where the
plugin dependencies themselves are not installed.
"""

import ast
from pathlib import Path
from typing import Iterator

from test_plugin_imports import IGNORE_FOLDERS, PLUGIN_LOCATIONS

REQUIREMENTS_METHOD_NAME = "get_requirements"

# imported module -> distribution names that may satisfy it in a
# requirements line
DEPENDENCY_FOR_MODULE = {
    "qiskit_aer": ("qiskit-aer", "qiskit_aer"),
    # qiskit.qasm3.loads only works when the optional importer package is
    # present,
    # either installed directly or pulled in by the qasm3-import extra of qiskit
    "qiskit.qasm3": ("qasm3-import", "qiskit_qasm3_import", "qiskit-qasm3-import"),
}


def get_plugin_packages() -> list[Path]:
    """Return the root folder of every plugin that is a package.

    Single module plugins are returned as well, they are their own root.
    """

    def iter_plugins(path: Path) -> Iterator[Path]:
        if (path / "__init__.py").exists():
            yield path
            return
        for child in path.iterdir():
            if child.name.startswith(".") or child.name in IGNORE_FOLDERS:
                continue
            if child.is_file() and child.suffix == ".py":
                yield child
            if child.is_dir():
                yield from iter_plugins(child)

    roots: list[Path] = []
    for location in PLUGIN_LOCATIONS:
        roots.extend(iter_plugins(location.resolve()))
    return roots


def iter_source_files(plugin_root: Path) -> Iterator[Path]:
    """Yield every python source file belonging to a plugin."""
    if plugin_root.is_file():
        yield plugin_root
        return
    for path in plugin_root.rglob("*.py"):
        if any(part in IGNORE_FOLDERS for part in path.parts):
            continue
        yield path


def extract_imported_modules(tree: ast.AST) -> set[str]:
    """Return the dotted names of all absolute imports in a source file."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def extract_declared_requirements(tree: ast.AST) -> str:
    """Return the literal parts of everything ``get_requirements`` returns.

    Interpolated values of f-strings are left out because only the package names
    are of interest here, and those are always written as literals.
    """
    parts: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != REQUIREMENTS_METHOD_NAME:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                parts.append(child.value)
    return "\n".join(parts)


def collect_plugin_dependencies() -> list[tuple[Path, set[str], str]]:
    """Return the imports and declared requirements of every plugin."""
    plugins = []
    for root in get_plugin_packages():
        imports: set[str] = set()
        requirements: list[str] = []
        for source_file in iter_source_files(root):
            tree = ast.parse(source_file.read_text(encoding="utf-8"), str(source_file))
            imports.update(extract_imported_modules(tree))
            requirements.append(extract_declared_requirements(tree))
        plugins.append((root, imports, "\n".join(requirements)))
    return plugins


def assert_dependency_is_declared(module: str):
    """Assert that every plugin importing ``module`` also declares it."""
    expected_names = DEPENDENCY_FOR_MODULE[module]
    offenders = []
    for root, imports, requirements in collect_plugin_dependencies():
        if module not in imports:
            continue
        if not any(name in requirements for name in expected_names):
            offenders.append(str(root))

    assert not offenders, (
        f"These plugins import '{module}' but none of {expected_names} appears in "
        f"their {REQUIREMENTS_METHOD_NAME}(), so installing the plugin will not "
        f"install what it imports and it fails at runtime: {sorted(offenders)}"
    )


def test_openqasm3_plugins_declare_the_qasm3_importer():
    """Plugins parsing OpenQASM 3 must declare the optional qasm3 importer."""
    assert_dependency_is_declared("qiskit.qasm3")


def test_qiskit_aer_users_declare_the_qiskit_aer_dependency():
    """Plugins using an Aer simulator must declare qiskit-aer."""
    assert_dependency_is_declared("qiskit_aer")


def test_at_least_one_plugin_imports_each_checked_module():
    """Guard the checks above against silently matching no plugin at all.

    Without this a typo in a module name would turn both tests into no-ops that
    can never fail.
    """
    all_imports: set[str] = set()
    for _, imports, _ in collect_plugin_dependencies():
        all_imports.update(imports)

    unused = sorted(set(DEPENDENCY_FOR_MODULE) - all_imports)
    assert not unused, (
        f"No plugin imports {unused} any more. Either the module was renamed or the "
        f"entry is stale, and the requirement check for it silently passes."
    )
