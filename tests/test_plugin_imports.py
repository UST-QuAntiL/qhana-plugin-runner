# Copyright 2024 QHAna plugin runner contributors.
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

"""Tests enshuring that plugin imports will not cause any problems."""

import ast
import sys
from collections import deque
from pathlib import Path

# search for plugins in the following locations
PLUGIN_LOCATIONS = [
    Path("./plugins"),
    Path("./stable_plugins"),
]

# ignore these folder names during plugin search
IGNORE_FOLDERS = {"__pycache__", "node_modules"}


# Only add stdlib modules or dependencies found in pyproject.toml or poetry.lock to this set
ALLOWED_IMPORTS = {
    # python stdlib
    "typing",
    "http",
    "mimetypes",
    # plugin runner
    "qhana_plugin_runner",
    # direct plugin runner dependencies
    "flask",
    "werkzeug",
    "marshmallow",
    "celery",
    "requests",
    "sqlalchemy",
    "typing_extensions",
}


PLUGIN_BASE_MODULE = ("qhana_plugin_runner", "util", "plugins")
PLUGIN_BASE_CLASS_NAME = "QHAnaPluginBase"
PLUGIN_BASE_QALIFIED = PLUGIN_BASE_MODULE + (PLUGIN_BASE_CLASS_NAME,)

BASE_PATH = Path(sys.base_prefix)
VENV_PATH = Path(sys.prefix)
REPOSITORY_PATH = Path(".").resolve()


def get_plugin_roots():
    """Find all possible plugin packages and modules recursively."""

    def iter_plugins(path: Path):
        init_file = path / "__init__.py"
        if init_file.exists():
            yield path
            return
        for path in path.iterdir():
            if path.name.startswith("."):
                continue
            if path.name in IGNORE_FOLDERS:
                continue
            if path.is_file() and path.suffix == ".py":
                yield path
            if path.is_dir():
                yield from iter_plugins(path)

    locations: list[Path] = []
    for location in PLUGIN_LOCATIONS:
        full_path = location.resolve()
        locations.extend(iter_plugins(full_path))

    return locations


def extract_imports(code: str, path: Path):  # noqa: C901
    """Use the ast module to extract all imported modules up until the
    definition of a plugin class (or the end of the file)."""
    tree = ast.parse(code, path)

    # current alias of plugin base class
    base_class_aliases = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                module = tuple(n.name.split("."))
                if module == PLUGIN_BASE_QALIFIED:
                    if n.asname:
                        base_class_aliases.add(str(n.asname))
                    else:
                        base_class_aliases.add(PLUGIN_BASE_CLASS_NAME)
                yield module, n.lineno
        elif isinstance(node, ast.ImportFrom):
            assert (
                node.module is not None
            ), f"Bad import found in line {node.lineno} in file {path}"
            module = tuple(("." * node.level + node.module).split("."))
            if module == PLUGIN_BASE_MODULE:
                for name in node.names:
                    if name.name == PLUGIN_BASE_CLASS_NAME:
                        if name.asname:
                            base_class_aliases.add(str(name.asname))
                        else:
                            base_class_aliases.add(PLUGIN_BASE_CLASS_NAME)
            for name in node.names:
                yield (*module, str(name.name)), node.lineno
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    if str(base.id) in base_class_aliases:
                        return


def extract_all_imports(code: str, path: Path):
    """Use the ast module to extract all information about all import statements in the code."""
    tree = ast.parse(code, path)

    nodes = deque(ast.iter_child_nodes(tree))

    while nodes:
        node = nodes.popleft()
        if isinstance(node, ast.Import):
            for n in node.names:
                module = tuple(n.name.split(".", maxsplit=1))
                if module:
                    yield module[0], n.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                yield node.level, node.lineno
            else:
                assert (
                    node.module is not None
                ), f"Bad import found in line {node.lineno} in file {path}"
                module = node.module.split(".", maxsplit=1)
                if module:
                    yield module[0], node.lineno
        else:
            nodes.extend(ast.iter_child_nodes(node))


def is_not_in_project_repo(module: tuple[str, ...]) -> bool:
    if not module:
        return True

    imported_path = REPOSITORY_PATH / module[0]
    if imported_path.exists():
        return False  # module is a relative import
    if len(module) == 1:
        imported_path = REPOSITORY_PATH / f"{module[0]}.py"
        if imported_path.exists():
            return False  # module is a relative import

    # module is not found inside the repository
    return True


def is_valid_import(module: tuple[str, ...], lineno, file_) -> bool:
    """Check if a given import is considered ok."""
    if not module:
        return False
    if module[0] == "":
        return True  # allow relative imports
    if module[0] in ALLOWED_IMPORTS:
        return True  # explicitly allowed imports
    sys_module = sys.modules.get(module[0])
    if sys_module:  # check resolved module path
        if not hasattr(sys_module, "__file__") or not sys_module.__file__:
            return True  # is a system module without a file location
        module_path = Path(sys_module.__file__)
        if module_path.is_relative_to(VENV_PATH):
            return False  # is installed dependency module
        if module_path.is_relative_to(BASE_PATH):
            return True  # is stdlib module
        assert not module_path.is_relative_to(
            REPOSITORY_PATH
        ), f"Plugins must use relative imports! Found import '{'.'.join(module)}' in line {lineno} of file {file_}"

    # check if absolute import resolves to a path relative to the repository
    assert is_not_in_project_repo(
        module
    ), f"Plugins must use relative imports! Found import '{'.'.join(module)}' in line {lineno} of file {file_}"

    return False


def check_file_imports(file_: Path, visited: set, module_base: Path | None = None):
    """Check python module imports."""
    if visited is not None:
        if str(file_) in visited:
            return  # already seen
        visited.add(str(file_))
    assert (
        file_.exists() and file_.suffix == ".py"
    ), "Must be an existing python file to check!"
    imports = list(extract_imports(file_.read_text(), file_))
    for module, lineno in imports:
        assert is_valid_import(
            module, lineno, file_
        ), f"Found early import of external dependency '{'.'.join(module)}' in line {lineno} of file {file_}"
        if module_base is not None:
            assert visited is not None, "visited set must be present for recursion!"
            if module and module[0] == "":
                check_imports_recursive(module, module_base, visited, lineno, file_)


def check_folder_imports(folder: Path, module_base: Path, visited: set):
    """Check python package imports."""
    if str(folder) in visited:
        return  # already seen
    visited.add(str(folder))

    file_ = folder / "__init__.py"
    assert (
        file_.exists() and file_.suffix == ".py"
    ), "Python package must include a __init__.py file!"
    imports = list(extract_imports(file_.read_text(), file_))
    for module, lineno in imports:
        assert is_valid_import(
            module, lineno, file_
        ), f"Found early import of external dependency '{'.'.join(module)}' in line {lineno} of file {file_}"
        if module and module[0] == "":
            check_imports_recursive(module, module_base, visited, lineno, file_)


def check_imports_recursive(
    module: tuple[str, ...], module_base: Path, visited: set, lineno: int, file_: Path
):
    """Resolve relative import and check imports recursively."""
    import_loc = {".".join(module)}

    import_path = file_
    if import_path.name == "__init__.py":
        import_path = import_path.parent

    for component in module[1:]:
        if component == "":
            import_path = import_path.parent
            assert import_path.is_relative_to(
                module_base
            ), f"Attempted relative import '{import_loc}' beyond package boundary in line {lineno} of file {file_}!"
        else:
            import_path = import_path / component

    file_path = import_path.parent / f"{import_path.name}.py"
    folder_path = import_path / "__init__.py"

    if file_path.exists():
        check_file_imports(file_path, module_base=module_base, visited=visited)
        return
    if folder_path.exists():
        check_folder_imports(import_path, module_base=module_base, visited=visited)
        return

    # mybe last component was a member of the python module, check parent path
    import_path = import_path.parent
    file_path = import_path.parent / f"{import_path.name}.py"
    folder_path = import_path / "__init__.py"

    if file_path.exists():
        check_file_imports(file_path, module_base=module_base, visited=visited)
        return

    assert (
        folder_path.exists()
    ), f"Import '{import_loc}' in line {lineno} of file {file_} cannot be resolved!"
    check_folder_imports(import_path, module_base=module_base, visited=visited)


def test_plugin_imports():
    """Test all imports of plugins up to the definition of the Plugin class."""
    plugin_locations = get_plugin_roots()
    visited_files = set()
    for loc in plugin_locations:
        if loc.is_file():
            check_file_imports(loc, visited=visited_files)
        elif loc.is_dir():
            check_folder_imports(loc, module_base=loc, visited=visited_files)


def test_relative_imports():
    """Check all plugins for potential absolute imports of plugin modules or
    relative imports beyond plugin boundaries."""
    plugin_locations = get_plugin_roots()
    for loc in plugin_locations:
        for file_ in loc.rglob("**/*.py"):
            imports = extract_all_imports(file_.read_text(), file_)
            for module, lineno in imports:
                if isinstance(module, int):
                    import_path = file_
                    for _ in range(module - 1):
                        assert (
                            import_path.parent
                        ), f"Found out of bounds relative import in line {lineno} of file {file_}!"
                        import_path = import_path.parent
                    assert import_path.is_relative_to(
                        loc
                    ), f"Found out of bounds relative import in line {lineno} of file {file_}!"
                else:
                    if module == "qhana_plugin_runner":
                        continue
                    assert is_not_in_project_repo(
                        (module,)
                    ), f"Plugins must use relative imports! Found import '{module}[...]' in line {lineno} of file {file_}"
